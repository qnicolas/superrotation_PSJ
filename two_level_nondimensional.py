import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
import os;import shutil;from pathlib import Path
import warnings; import sys
import argparse

#########################################
#########  CONTROL PARAMETERS    ########
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('--Ro', type=float, required=True, help='Thermal Rossby number')
parser.add_argument('--Trad', type=float, required=True, help='Nondimensional radiative timescale')
parser.add_argument('--type', type=str, required=True, help='Sets the longitudinal structure of the forcing. Options: locked, axi, suarezp25, suarezp1, suarezp05, semilocked, semilocked2, halfcoslon')
parser.add_argument('--E', type=float, default=0.02, help='Ekman number, controls the strength of the bottom drag (default: 0.02)')
parser.add_argument('--stratification', type=float, default=0.05, help='Stratification parameter, which controls the vertical structure of the forcing (default: 0.05)')
parser.add_argument('--output_dir', type=str, default="./data/", help='Directory to save output files (default: ./data/)')

args = parser.parse_args()

# Nondimensional parameters
Ro_T = args.Ro
tau_rad_nondim = args.Trad
lontyp = args.type
E = args.E
stratification = args.stratification

# Resolution parameters
Nphi = 128; Ntheta = 64; resolution='T64'
#Nphi = 64; Ntheta = 32; resolution='T32'
dealias = (3/2, 3/2)
dtype = np.float64

# Misc run parameters
output_dir = args.output_dir
restart = False # set to True if doing a restart run.
restart_id='s1' # set if doing a restart run. If doing a second restart, then set to 's2', etc.
use_CFL = True # Whether to use adaptive timestep based on CFL condition. 
safety_CFL = 0.7 

linear = False
timestep = 1e-2 # Minimum timestep if using CFL, or fixed timestep if not using CFL. 
stop_sim_time = 1000*4*np.pi


#########################################
###########  GENERAL SETUP    ###########
#########################################

lat_forcing = lambda lat: np.cos(lat); lattyp=''
if lontyp=='locked':
    lon_forcing = lambda lon: np.cos(lon)*(np.cos(lon)>=0.)
elif lontyp=='axi':
    lon_forcing = lambda lon: 1/np.pi*lon**0
elif lontyp=='suarezp25':
    lon_forcing = lambda lon: 1/np.pi*lon**0 + 0.25 * np.cos(lon)
elif lontyp=='suarezp1':
    lon_forcing = lambda lon: 1/np.pi*lon**0 + 0.1 * np.cos(lon)
elif lontyp=='suarezp05':
    lon_forcing = lambda lon: 1/np.pi*lon**0 + 0.05 * np.cos(lon)
elif lontyp=='semilocked':
    lon_forcing = lambda lon: 1/np.pi*lon**0 + 0.5 * np.cos(lon)
elif lontyp=='semilocked2':
    lon_forcing = lambda lon: 1/np.pi*lon**0 + 0.5 * np.cos(lon) + 2/(3*np.pi) * np.cos(2*lon)
elif lontyp=='halfcoslon':
    lon_forcing = lambda lon: 0.5*np.cos(lon)
else:
    raise ValueError("wrong input argument for lontyp")

if linear:
    ext='_linear'
else:
    ext=''
ext+=''

snapshot_id = 'snapshots_2levelnondim_%s_%s%s_%.2f_%.2f_%i_%.2f%s'%(resolution,lontyp,lattyp,Ro_T,E,tau_rad_nondim,stratification,ext)
snapshot_id = snapshot_id.replace('.','p')

# Fixed parameters
kappa = 2/7
P1 = 0.25**kappa
P2 = 0.75**kappa

# Hyperdiffusion
hyperdiff_degree = 4; nu = 1.5e-7
if Ro_T>2 and tau_rad_nondim<=10:
    nu*=2.

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype,comm=MPI.COMM_WORLD)
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=1, dealias=dealias, dtype=dtype)

# cross product by zhat for Coriolis term
zcross = lambda A: d3.MulCosine(d3.skew(A))
if hyperdiff_degree==4:
    hyperdiff = lambda A : nu*d3.lap(d3.lap(A))
elif hyperdiff_degree==8:
    hyperdiff = lambda A : nu*d3.lap(d3.lap(d3.lap(d3.lap(A))))
else:
    raise ValueError('hyperdiff_degree')

###############################
###### SAVE CURRENT FILE ######
###############################
if dist.comm.rank == 0:
    Path(os.path.join(output_dir, snapshot_id)).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(output_dir, snapshot_id, os.path.basename(__file__)))


###############################
######## SETUP PROBLEM ########
###############################

# Fields
u1     = dist.VectorField(coords, name='u1', bases=full_basis)
u2     = dist.VectorField(coords, name='u2', bases=full_basis)
omega  = dist.Field(name='omega' , bases=full_basis)
Phi1   = dist.Field(name='Phi1'  , bases=full_basis)
Phi2   = dist.Field(name='Phi2'  , bases=full_basis)
theta1 = dist.Field(name='theta1', bases=full_basis)
theta2 = dist.Field(name='theta2', bases=full_basis)
theta1E = dist.Field(name='theta1E', bases=full_basis)
theta2E = dist.Field(name='theta2E', bases=full_basis)
tau = dist.Field(name='tau')

## Problem
problem = d3.IVP([u1,u2,Phi1,theta1,theta2,tau], namespace=locals())
    
if linear:
    problem.add_equation("dt(u1) + hyperdiff(u1) + grad(Phi1) + zcross(u1) = 0")
    problem.add_equation("dt(u2) + hyperdiff(u2) + grad(Phi1- (P2-P1)*(theta1+theta2)/2) + zcross(u2) + E * u2 = 0")
    problem.add_equation("dt(theta1) + hyperdiff(theta1) - Ro_T*stratification*div(u2)/2 + theta1/tau_rad_nondim = theta1E/tau_rad_nondim")
    problem.add_equation("dt(theta2) + hyperdiff(theta2) - Ro_T*stratification*div(u2)/2 + theta2/tau_rad_nondim = theta2E/tau_rad_nondim")
    problem.add_equation("div(u1+u2) + tau = 0")
    problem.add_equation("ave(Phi1) = 0")
else:
    problem.add_equation("dt(u1) + hyperdiff(u1) + grad(Phi1) + zcross(u1) = - Ro_T * (u1@grad(u1) + div(u2)/2*(u2-u1))")
    problem.add_equation("dt(u2) + hyperdiff(u2) + grad(Phi1- (P2-P1)*(theta1+theta2)/2) + zcross(u2) + E * u2 = - Ro_T * (u2@grad(u2) + div(u2)/2*(u2-u1))")
    problem.add_equation("dt(theta1) + hyperdiff(theta1) + theta1/tau_rad_nondim = - Ro_T * (u1@grad(theta1) + div(u2)/2*(theta2-theta1)) + theta1E/tau_rad_nondim")
    problem.add_equation("dt(theta2) + hyperdiff(theta2) + theta2/tau_rad_nondim = - Ro_T * (u2@grad(theta2) + div(u2)/2*(theta2-theta1)) + theta2E/tau_rad_nondim")
    problem.add_equation("div(u1+u2) + tau = 0")
    problem.add_equation("ave(Phi1) = 0")

# Solver + set time-stepping scheme
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# CFL condition for adaptive time-stepping
CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=safety_CFL, threshold=0.1,
             max_change=1.2, min_change=0.8, max_dt=1., min_dt=timestep)
CFL.add_velocity(Ro_T * u1)
CFL.add_frequency(Ro_T * d3.div(u2)*2)

###################################################
######## SETUP RESTART & INITIALIZE FIELDS ########
###################################################
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi

theta1E['g'] = (1-stratification*np.log(P1)) * lat_forcing(lat)*lon_forcing(lon)
theta2E['g'] = (1-stratification*np.log(P2)) * lat_forcing(lat)*lon_forcing(lon)

sample_lat = np.linspace(-np.pi/2,np.pi/2,201)[:,None]
sample_lon = np.linspace(-np.pi,np.pi,401)[None,:]
meantheta1E = np.mean( np.cos(sample_lat) * ((1-stratification*np.log(P1)) * lat_forcing(sample_lat)*lon_forcing(sample_lon)) ) * np.pi/2
meantheta2E = np.mean( np.cos(sample_lat) * ((1-stratification*np.log(P2)) * lat_forcing(sample_lat)*lon_forcing(sample_lon)) ) * np.pi/2

if not restart:
    theta1.fill_random('g', seed=1, distribution='normal', scale=1e-4)
    theta2.fill_random('g', seed=2, distribution='normal', scale=1e-4)
    theta1['g'] += meantheta1E
    theta2['g'] += meantheta2E
    Phi2['g'] = - (P2-P1) * (theta1['g']+theta2['g'])/2
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state(os.path.join(output_dir, snapshot_id, '%s_%s.h5' % (snapshot_id, restart_id)))
    file_handler_mode = 'append'


##########################################
######## SETUP SNAPSHOTS & DO RUN ########
##########################################
snapshots = solver.evaluator.add_file_handler(os.path.join(output_dir, snapshot_id), sim_dt=np.pi,mode=file_handler_mode)
snapshots.add_tasks(solver.state)
snapshots.add_task(Phi1- (P2-P1)*(theta1+theta2)/2, name='Phi2')
snapshots.add_task(d3.div(u2)/2, name='omega')
snapshots.add_task(-d3.div(d3.skew(u1)), name='vorticity_1')
snapshots.add_task(-d3.div(d3.skew(u2)), name='vorticity_2')

# Main loop
with warnings.catch_warnings():
    warnings.filterwarnings('error',category=RuntimeWarning)
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            if use_CFL:
                timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 20 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.info('Last dt=%e' %(timestep))
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
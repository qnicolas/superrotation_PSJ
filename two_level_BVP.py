import numpy as np
import xarray as xr
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
import argparse
import os

#########################################
#########  CONTROL PARAMETERS    ########
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('--Ro', type=float, required=True, help='Thermal Rossby number')
parser.add_argument('--Trad', type=float, required=True, help='Nondimensional radiative timescale')
parser.add_argument('--type', type=str, default='halfcoslon', help='Sets the longitudinal structure of the forcing. Default: halfcoslon. Options: locked, axi, suarezp25, suarezp1, suarezp05, semilocked, semilocked2, halfcoslon')
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
Nphi = 256; Ntheta = 128; resolution='T128'
# Nphi = 128; Ntheta = 64; resolution='T64'
# Nphi = 64; Ntheta = 32; resolution='T32'
dealias = (3/2, 3/2)
dtype = np.float64

output_dir = args.output_dir


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
    raise ValueError("wrong input argument: lontyp")

snapshot_id = 'bvp_2levelnondim_%s_%s%s_%.2f_%.2f_%i_%.2f'%(resolution,lontyp,lattyp,Ro_T,E,tau_rad_nondim,stratification)
snapshot_id = snapshot_id.replace('.','p')

# Fixed parameters
kappa = 2/7
P1 = 0.25**kappa
P2 = 0.75**kappa

# Hyperdiffusion parameters
hyperdiff_degree = 4; nu = 1e-10

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
######## SETUP PROBLEM ########
###############################

# Fields
u1     = dist.VectorField(coords, name='u1', bases=full_basis)
u2     = dist.VectorField(coords, name='u2', bases=full_basis)
omega  = dist.Field(name='omega' , bases=full_basis)
Phi1   = dist.Field(name='Phi1'  , bases=full_basis)
theta1 = dist.Field(name='theta1', bases=full_basis)
theta2 = dist.Field(name='theta2', bases=full_basis)
theta1E = dist.Field(name='theta1E', bases=full_basis)
theta2E = dist.Field(name='theta2E', bases=full_basis)
tau = dist.Field(name='tau')

# Forcings
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi

theta1E['g'] = (1-stratification*np.log(P1)) * lat_forcing(lat)*lon_forcing(lon)
theta2E['g'] = (1-stratification*np.log(P2)) * lat_forcing(lat)*lon_forcing(lon)


## Problem
problem = d3.LBVP([u1,u2,Phi1,theta1,theta2,tau], namespace=locals())
    
problem.add_equation("hyperdiff(u1) + grad(Phi1) + zcross(u1) = 0")
problem.add_equation("hyperdiff(u2) + grad(Phi1- (P2-P1)*(theta1+theta2)/2) + zcross(u2) + E * u2 = 0")
problem.add_equation("hyperdiff(theta1) - Ro_T*stratification*div(u2)/2 + theta1/tau_rad_nondim = theta1E/tau_rad_nondim")
problem.add_equation("hyperdiff(theta2) - Ro_T*stratification*div(u2)/2 + theta2/tau_rad_nondim = theta2E/tau_rad_nondim")
problem.add_equation("div(u1+u2) + tau = 0")
problem.add_equation("ave(Phi1) = 0")

# Solver
solver = problem.build_solver()
solver.solve()

omega = (d3.div(u2)/2).evaluate()
vorticity_1 = ( -d3.div(d3.skew(u1)) ).evaluate()
vorticity_2 = ( -d3.div(d3.skew(u2)) ).evaluate()

for var in (u1,u2,Phi1,theta1,theta2,theta1E,theta2E,omega,vorticity_1,vorticity_2):
    var.change_scales(1)


###############################
####### OUTPUT TO netCDF ######
###############################
def make_da(name,var,phi,theta,wind=False):
    phi   = phi[:,0]
    theta = theta[0]
    if wind:
        dims = ['component','longitude','latitude']
    else:
        dims = ['longitude','latitude']
    var_da  = xr.DataArray(var, coords={'longitude':('longitude',(phi-np.pi)*180/np.pi),
                                   'latitude':('latitude',(np.pi/2-theta)*180/np.pi),
                                   'phi':('longitude',phi),
                                   'theta':('latitude',theta)}, dims=dims,name=name)
    return var_da

def make_ds(u1g,u2g,Phi1g,theta1g,theta2g,theta1Eg,theta2Eg,omegag,vorticity_1g,vorticity_2g,phi,theta):
    ds = xr.merge([make_da('u1',u1g,phi,theta,True),
                   make_da('u2',u2g,phi,theta,True),
                   make_da('Phi1',Phi1g,phi,theta),
                   make_da('theta1',theta1g,phi,theta),
                   make_da('theta2',theta2g,phi,theta),
                   make_da('theta2E',theta2Eg,phi,theta),
                   make_da('theta1E',theta1Eg,phi,theta),
                   make_da('omega',omegag,phi,theta),
                   make_da('vorticity_1',vorticity_1g,phi,theta),
                   make_da('vorticity_2',vorticity_2g,phi,theta)
                  ])
    ds['Phi2'] = ds.Phi1- (P2-P1)*(ds.theta1+ds.theta2)/2
    return ds

# Gather global data

phi, theta = full_basis.global_grids(dist, scales=(1,1))
u1g = u1.allgather_data('g')
u2g = u2.allgather_data('g')
Phi1g = Phi1.allgather_data('g')
theta1g = theta1.allgather_data('g')
theta2g = theta2.allgather_data('g')
theta1Eg = theta1E.allgather_data('g')
theta2Eg = theta2E.allgather_data('g')
omegag = omega.allgather_data('g')
vorticity_1g = vorticity_1.allgather_data('g')
vorticity_2g = vorticity_2.allgather_data('g')

if dist.comm.rank == 0:
    make_ds(u1g,u2g,Phi1g,theta1g,theta2g,theta1Eg,theta2Eg,omegag,vorticity_1g,vorticity_2g,phi,theta).to_netcdf(os.path.join(output_dir, snapshot_id + '.nc'))

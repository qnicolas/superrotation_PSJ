import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

def calc_emfc_bvp(sim,Ro_T):
    """
    Calculate upper and lower level EMFC for the BVP/EVP solutions. 
    
    `sim` is modified in-place to add the following fields:
    - `zetav`: the vorticity flux component of the EMFC
    - `omegauhat`: the vertical momentum advection component of the EMFC
    - `emfc`: the total EMFC, which is the sum of `zetav` and `omegauhat`

    Parameters
    ----------
    sim : xarray.Dataset
        Dataset containing the fields of the BVP solution.
        Must contain the fields u1 (upper-level velocity), 
        u2 (lower-level velocity), omega (pressure velocity), 
        vorticity_1 (upper-level vorticity), and vorticity_2 (lower-level vorticity).
    Ro_T : float
        Thermal Rossby number.
    """
    u1 = sim.u1[0]
    v1 = -sim.u1[1]

    u2 = sim.u2[0]
    v2 = -sim.u2[1]

    omega = sim.omega
    zeta1 = sim.vorticity_1
    zeta2 = sim.vorticity_2

    zeta1v1 = Ro_T * v1 * zeta1 
    zeta2v2 = Ro_T * v2 * zeta2
    omegauhat = Ro_T * ((u1-u2)*omega)
    omegaubar = Ro_T * ((u1+u2)*omega)

    sim['zeta1v1'] = zeta1v1.mean('longitude')
    sim['omegauhat'] = omegauhat.mean('longitude')
    sim['omegaubar'] = omegaubar.mean('longitude')
    sim['emfc1'] = sim['zeta1v1'] + sim['omegauhat'] 

    sim['zeta2v2'] = zeta2v2.mean('longitude')
    sim['emfc2'] = sim['zeta2v2'] + sim['omegauhat'] 



def calc_emfc(sim,Ro_T):
    """
    Calculate upper-level momentum flux convergence from the mean flow and eddies for fully nonlinear solutions. 

    `sim` is modified in-place to add the following fields:
    - `mmc`: Mean flow contribution to total momentum flux convergence, i.e, "mean meridional circulation" (MMC) term,
    - `emfc`: Eddy momentum flux convergence (EMFC) term, which is the total momentum flux convergence minus the MMC term.
    """
    u1 = sim.u1[:,0]
    v1 = -sim.u1[:,1]
    omega = sim.omega
    zeta1 = sim.vorticity_1
    u2 = sim.u2[:,0]
    
    fv = v1 * np.sin(v1.latitude*np.pi/180)
    zetav = v1*Ro_T*zeta1 
    omegauhat = Ro_T*((u1-u2)*omega)
    
    fvbar = fv.mean('longitude')
    zetabarvbar = Ro_T * v1.mean('longitude') * zeta1.mean('longitude')
    omegabaruhatbar = Ro_T * ((u1-u2).mean('longitude') * omega.mean('longitude'))

    fzetav     = (fv+zetav).mean('longitude')
    omegauhat  = omegauhat.mean('longitude')

    sim['mmc']  = fvbar + zetabarvbar + omegabaruhatbar
    sim['emfc'] = omegauhat + fzetav - sim['mmc']

def calc_emfc_tmean(sim,Ro_T):
    """
    Calculate upper-level momentum flux convergence from the mean flow and eddies for fully nonlinear solutions, using time-mean fields. 

    `sim` is modified in-place to add the following fields:
    - `mmc_tmean`: Time-mean mean flow contribution to total momentum flux convergence, i.e, "mean meridional circulation" (MMC) term,
    - `emfc_tmean`: Time-mean eddy momentum flux convergence (EMFC) term, i.e. stationary eddy EMFC.
    """
    u1 = sim.u1[:,0]
    v1 = -sim.u1[:,1]
    omega = sim.omega
    zeta1 = sim.vorticity_1
    u2 = sim.u2[:,0]
    fv = v1 * np.sin(v1.latitude*np.pi/180)
    mmc = fv.mean(('t','longitude')) + v1.mean(('t','longitude'))*Ro_T*zeta1.mean(('t','longitude')) + Ro_T*((u1-u2).mean(('t','longitude'))*omega.mean(('t','longitude')))
    
    u1p = (u1 - u1.mean('longitude')).mean('t')
    v1p = (v1 - v1.mean('longitude')).mean('t')
    u2p = (u2 - u2.mean('longitude')).mean('t')
    zeta1p = (zeta1 - zeta1.mean('longitude')).mean('t')
    omegap = (omega - omega.mean('longitude')).mean('t')
    emfc = Ro_T * (v1p*zeta1p).mean('longitude') + Ro_T * ((u1p-u2p)*omegap).mean('longitude')

    sim['mmc_tmean']  = mmc
    sim['emfc_tmean'] = emfc

#################################################################################
################################ EMFC spectrum ##################################
#################################################################################

def cospec_kc_from_cospec(cospec,phase_speeds = np.linspace(-1.,1.,2001)):
    """
    Transform cospectrum data from (frequency,wavenumber,latitude) space to a (phase-speed,wavenumber,latitude) cospectrum.
    
    This function applies Gaussian smoothing to the input cospectrum data and 
    scales it by the ratio of wavenumber to cosine of latitude angle. It then 
    interpolates the scaled cospectrum onto a uniform phase speed grid.
    
    Parameters
    ----------
    cospec : xarray.DataArray
        Input cospectrum data with dimensions including
        'kphi', 'theta', and 'frequency'. A 'c' (phase speed) 
        coordinate is also required
    phase_speeds : numpy.ndarray, optional
        Array of phase speed values for interpolation output.
        Default is np.linspace(-1., 1., 2001).
    
    Returns
    -------
    xarray.DataArray
        Transformed cospectrum data interpolated onto uniform phase speed
        grid
    
    Notes
    -----
    The function applies the following transformations:
    1. Gaussian smoothing with sigma=4 along the first axis
    2. Scaling by wavenumber divided by sine of latitude angle
    3. Interpolation onto uniform phase speed grid
    
    Values outside the original phase speed range are set to zero.
    """
    cospec = gaussian_filter1d(cospec,4.,axis=0) * cospec ** 0
    cospec_scaled = cospec * cospec.kphi / np.sin(cospec.theta)

    cospec_kc = 0*xr.DataArray(phase_speeds,coords={'c':phase_speeds},dims=['c']) + cospec[0].drop('c')*0
    for k in range(len(cospec.kphi)):
        for l in range(len(cospec.latitude)):
            cospec_kc[:,k,l] = interp1d(cospec.c[:,k,l],cospec_scaled[:,k,l],axis=0,bounds_error=False,fill_value=0.)(phase_speeds)    
    return cospec_kc    

def get_emfc_spectra_nondim_zeta(sim,Ro_T):
    """Calculate spectral decomposition of the EMFC for a fully nonlinear simulation.
    See Appendix E of Nicolas and Vallis (2026) for details.

    Parameters
    ----------
    sim : xarray.Dataset
        Dataset containing the fields of the simulation.
        Must contain the fields u1 (upper-level velocity), 
        u2 (lower-level velocity), omega (pressure velocity), 
        vorticity_1 (upper-level vorticity), and vorticity_2 (lower-level vorticity).
    Ro_T : float
        Thermal Rossby number.

    Returns
    -------
    xarray.DataArray
        Spectral decomposition of the EMFC with dimensions ['frequency','kphi','latitude'] and a coordinate 'c' for phase speed.
    """
    u1_mixed    = np.fft.rfft(sim.u1[:,0],axis=1,norm='forward')
    v1_mixed    =-np.fft.rfft(sim.u1[:,1],axis=1,norm='forward')
    u2_mixed    = np.fft.rfft(sim.u2[:,0],axis=1,norm='forward')
    zeta1_mixed = np.fft.rfft(sim.vorticity_1,axis=1,norm='forward')
    omega_mixed = np.fft.rfft(sim.omega,axis=1,norm='forward')

    # correct fourier coefficients so that any field phi is given by phi(lambda) = sum(phitilde(k) * exp(i*k*lamda))
    u1_mixed   [:,1:] *= 2 
    v1_mixed   [:,1:] *= 2 
    u2_mixed   [:,1:] *= 2 
    zeta1_mixed[:,1:] *= 2 
    omega_mixed[:,1:] *= 2 
    
    freqs = -2*np.pi*np.fft.fftfreq(len(sim.t),(sim.t[1]-sim.t[0]).data)
    order = np.argsort(freqs); freqs = freqs[order]
    u1_spec    = np.fft.fft(u1_mixed,axis=0)[order]
    v1_spec    = np.fft.fft(v1_mixed,axis=0)[order]
    u2_spec    = np.fft.fft(u2_mixed,axis=0)[order]
    zeta1_spec = np.fft.fft(zeta1_mixed,axis=0)[order]
    omega_spec = np.fft.fft(omega_mixed,axis=0)[order]
    
    u1_spec = xr.DataArray(u1_spec,
                           coords={'frequency' :freqs,
                                   'kphi':np.arange(len(u1_mixed[0])),
                                   'latitude': sim.latitude,
                                   'theta': sim.theta
                                  },
                           dims=['frequency','kphi','latitude']
                          )
    v1_spec = xr.DataArray(v1_spec, dims=u1_spec.dims, coords=u1_spec.coords)
    u2_spec = xr.DataArray(u2_spec, dims=u1_spec.dims, coords=u1_spec.coords)
    zeta1_spec = xr.DataArray(zeta1_spec, dims=u1_spec.dims, coords=u1_spec.coords)
    omega_spec = xr.DataArray(omega_spec, dims=u1_spec.dims, coords=u1_spec.coords)
    
    u1_spec = u1_spec.sortby('frequency')
    v1_spec = v1_spec.sortby('frequency')
    u2_spec = u2_spec.sortby('frequency')
    zeta1_spec = zeta1_spec.sortby('frequency')
    omega_spec = omega_spec.sortby('frequency')
    
    cospec_hz = Ro_T * (np.real(zeta1_spec*np.conj(v1_spec)))
    cospec_vt = Ro_T *  np.real(omega_spec*np.conj(u1_spec-u2_spec))
    
    cospec = (cospec_hz+cospec_vt).assign_coords({'c':cospec_hz.frequency/cospec_hz.kphi * np.sin(cospec_hz.theta)})

    # Need to correct the contribution of the k=0 component by a factor 2
    cospec[:,0] = cospec[:,0]*2

    # renormalize so that integral over omega - sum over k gives full EMFC
    deltaomega = cospec.frequency[1]-cospec.frequency[0]
    cospec = cospec / deltaomega / len(cospec.frequency)**2 / 2

    return cospec    
import numpy as np
import xarray as xr 

def thetaphi_to_latlon(ds):
    return ds.assign_coords({'longitude':(ds.phi-np.pi)*180/np.pi,'latitude':(np.pi/2-ds.theta)*180/np.pi}).swap_dims({'phi':'longitude','theta':'latitude'})

def open_h5(name,sim='s1',SNAPSHOTS_DIR = "/Users/qnicolas/superrotation/data/"):
    ds = thetaphi_to_latlon(xr.open_dataset(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(name,name,sim),engine='dedalus'))
    return ds.assign_coords({'day':ds.t/24})

def open_h5s(name,sims,SNAPSHOTS_DIR = "/Users/qnicolas/superrotation/data/"):
    return xr.concat([open_h5(name,sim,SNAPSHOTS_DIR) for sim in sims],dim='t')

def open_h5s_wgauge(name,sims,gauge_names=('tau_Phi1',),SNAPSHOTS_DIR = "/Users/qnicolas/superrotation/data/"):
    return xr.concat([open_h5(name,sim,SNAPSHOTS_DIR).drop((*gauge_names,'constant')) for sim in sims],dim='t',coords='minimal',compat='override')
   
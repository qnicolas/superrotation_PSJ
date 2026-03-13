import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

def plot_map(ax, phi, wind, 
             cmap = plt.cm.viridis, add_colorbar=True,
             wind_disc=1, quiver_kwargs={}, include_qk=False, qk_scale=100, startlat=0, qkx=1.05, qky=-0.05, 
             **contourf_kwargs
            ):
    """Plot a map of a field `phi` with overlaid wind vectors `wind` on the axes `ax`. 

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the map.
    phi : xarray.DataArray
        The field to plot as a filled contour. Must have coordinates 'longitude' and 'latitude'.
    wind : xarray.DataArray
        The wind field to plot as vectors. Must have dimensions ['component','longitude','latitude']
        Uses the Dedalus convention where component 0 is the zonal wind and component 1 is MINUS the meridional wind.
    cmap : matplotlib.colors.Colormap, optional
        The colormap to use for the filled contour (default: plt.cm.viridis).
    add_colorbar : bool, optional
        Whether to add a colorbar (default: True).
    wind_disc : int, optional
        The spacing for the wind vectors (default: 1).
    quiver_kwargs : dict, optional
        Additional keyword arguments for the quiver plot (default: {}).
    include_qk : bool, optional
        Whether to include a quiver key (default: False).
    qk_scale : float, optional
        The scale for the quiver key (default: 100).
    startlat : int, optional
        The starting latitude for the wind vectors (default: 0).
    qkx : float, optional
        The x-position of the quiver key (default: 1.05).
    qky : float, optional
        The y-position of the quiver key (default: -0.05).
    **contourf_kwargs : dict
        Additional keyword arguments for the contourf plot.

    """
    c = wrap_lon(phi).plot.contourf(ax=ax,
                                    y='latitude',
                                    cmap=cmap,
                                    add_colorbar=add_colorbar,
                                    transform=ccrs.PlateCarree(),
                                    **contourf_kwargs)
    n=2*wind_disc
    m=wind_disc
    q=ax.quiver(wind.longitude.data[::n],
                wind.latitude.data[startlat::m],
                wind[0].data[::n,startlat::m].T,
               -wind[1].data[::n,startlat::m].T,
                transform=ccrs.PlateCarree(),**quiver_kwargs)
    if include_qk:
        ax.quiverkey(q, qkx, qky, qk_scale, str(qk_scale), labelpos='N',coordinates='axes',color='k')
    return c

def filter_mode(field, k, freq):
    """
    Filter the input `field` to extract the mode with zonal wavenumber `k` and frequency `freq`.
    The function performs a Fourier transform in time and longitude, selects the specified mode, 
    then applies an inverse Fourier transform to return the filtered field in physical space.

    Parameters
    ----------
    field : xarray.DataArray
        The input field to be filtered. Must have dimensions including 't' and 'longitude'.
    k : int
        The zonal wavenumber of the mode to extract.
    freq : float
        The frequency of the mode to extract.
    """
    field_mixed = np.fft.rfft(field,axis=1,norm='forward')
    freqs = -2*np.pi*np.fft.fftfreq(len(field.t),(field.t[1]-field.t[0]).data)
    order = np.argsort(freqs); freqs = freqs[order]
    field_spec = np.fft.fft(field_mixed,axis=0)[order]
    
    field_spec = xr.DataArray(field_spec,
                           coords={'frequency' :freqs,
                                   'kphi':np.arange(len(field_mixed[0])),
                                   'latitude': field.latitude,
                                   'theta': field.theta
                                  },
                           dims=['frequency','kphi','latitude']
                          )
    return np.real( field_spec.sel(kphi = k).sel(frequency=freq,method = 'nearest') * np.exp(1j*k*field.phi))

def stag_latitude(ds):
    """Stagger the latitude coordinate of a dataset `ds` by averaging adjacent latitudes"""
    ds_destag = (ds.data[:,:,:-1]+ds.data[:,:,1:])/2
    newlat_dat = (ds.latitude.data[:-1]+ds.latitude.data[1:])/2
    newlat = xr.DataArray(newlat_dat,coords={'latitude':newlat_dat},dims=['latitude'])
    return ds_destag * (ds.component**0 * ds.longitude**0 * newlat**0)

def add_ticks(ax,lons = range(-180,181,180), lats = range(-90,91,45),sz=16):
    """Add longitude and latitude ticks to a cartopy map axes `ax` with specified longitude and latitude values and label size."""
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray',alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': sz, 'color': 'k'}
    gl.ylabel_style = {'size': sz, 'color': 'k'}

def wrap_lon(ds):
    """Fill the longitude coordinate of a DataArray `da` by appending a column at longitude 180 with the same data as longitude -180. 
      This is useful for plotting filled contours without gaps at the dateline."""
    ds_wrapped = ds.pad(longitude=1, mode="wrap").assign_coords(longitude=ds.longitude.pad(longitude=1, mode="reflect", reflect_type="odd"))
    return ds_wrapped.sel(longitude = slice(-180.01,180.01))

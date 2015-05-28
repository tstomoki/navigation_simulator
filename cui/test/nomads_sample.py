# basic NOMADS OpenDAP extraction and plotting script
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import sys
# import own modules #
sys.path.append('../public')
sys.path.append('../models')
from constants   import *
from my_modules  import *
# import own modules #

# set up the figure
plt.figure()

# set up the URL to access the data server.
# See the NWW3 directory on NOMADS
# for the list of available model run dates.

end_date   = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=8)
date_range = generate_date_array(start_date, end_date)
print_with_notice("generating wave map from %s to %s" % (start_date.strftime('%Y/%m/%d'), end_date.strftime('%Y/%m/%d')))

for target_date in date_range:
    target_date_key = target_date.strftime('%Y%m%d')
    url='http://nomads.ncep.noaa.gov:9090/dods/wave/nww3/nww3'+ \
        target_date_key+'/nww3'+target_date_key+'_00z'

    # Extract the significant wave height of combined wind waves and swell
    try:
        file = netCDF4.Dataset(url)
    except RuntimeError:
        print "ERROR: Looks like data on %s does not exist!" % (target_date_key)
        continue

    lat  = file.variables['lat'][:]
    lon  = file.variables['lon'][:]
    data = file.variables['htsgwsfc'][1,:,:]
    file.close()

    # Since Python is object oriented, you can explore the contents of the NOMADS
    # data set by examining the file object, such as file.variables.

    # The indexing into the data set used by netCDF4 is standard python indexing.
    # In this case we want the first forecast step, but note that the first time
    # step in the RTOFS OpenDAP link is all NaN values.  So we start with the
    # second timestep
    
    # Plot the field using Basemap.  Start with setting the map
    # projection using the limits of the lat/lon data itself:
    # clear graph
    plt.clf()    
    m=Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(), \
              urcrnrlon=lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
              resolution='c')
    
    # convert the lat/lon values to x/y projections.
    
    x, y = m(*np.meshgrid(lon,lat))
    
    # plot the field using the fast pcolormesh routine
    # set the colormap to jet.
    
    m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.jet)
    m.colorbar(location='right')
    
    # Add a coastline and axis values.
    
    m.drawcoastlines()
    m.fillcontinents()
    m.drawmapboundary()
    m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
    
    # Add a colorbar and title, and then show the plot.
    plt.title('NWW3 Significant Wave Height from NOMADS on %s' % target_date_key)
    output_file_path = "%s/%s.png" % (WAVE_DIR_PATH, target_date_key)
    plt.savefig(output_file_path)
    print_with_notice("wave map on %s has been generated" % (target_date.strftime('%Y/%m/%d')))

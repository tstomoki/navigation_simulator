# basic NOMADS OpenDAP extraction and plotting script
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import sys
import os
from pdb import *
import datetime
# import own modules #
sys.path.append('../public')
sys.path.append('../models')
from constants   import *
from my_modules  import *
# import own modules #

min_lon              = 0
max_lon              = 150
min_lat              = -45
max_lat              = 60

def in_range(lat, lon):
    ret_flag = False
    if (lat >= min_lat) and (lat < max_lat):
        if (lon >= min_lon) and (lon < max_lon):
            ret_flag = True
    return ret_flag

def in_square(target_areas, lon, lat):
    ret_flag = False

    for target_area in target_areas:
        start_point = target_area[0]
        end_point   = target_area[1]

        if (lon <= start_point[0]) and (lon > end_point[0]):
            if (lat <= start_point[1]) and (lat > end_point[1]):
                ret_flag = True
    return ret_flag

def calc_date(time):
    base_date = datetime.datetime(1800, 1,1)
    return base_date + datetime.timedelta(hours=time)

def analyze():
    nc_file = '../data/weather_data/wspd.mon.mean.nc'
    url = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/wspd.mon.mean.nc'
    fh = netCDF4.Dataset(url)
    print 'got data'
    #fh      = netCDF4.Dataset(nc_file, mode='r')
    
    plt.clf()    
    lat  = fh.variables['lat'][:]
    lon  = fh.variables['lon'][:]
    time = fh.variables['time'][:]
    wspd = fh.variables['wspd'][:]
    m    = Basemap(projection='mill',lat_ts=10,llcrnrlon=min_lon, \
                   urcrnrlon  = max_lon,llcrnrlat=min_lat,urcrnrlat=max_lat,           \
                   resolution = 'c')
    
    # convert the lat/lon values to x/y projections.
    #x, y = m(*np.meshgrid(lon,lat))
    
    # plot the field using the fast pcolormesh routine
    # set the colormap to jet.
    #m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.jet)
    #m.colorbar(location='right')
    
    # Add a coastline and axis values.
    
    m.drawcoastlines(linewidth=0.5, color='#222222')
    m.drawmapboundary(fill_color='blue')
    m.fillcontinents(color='#71ff71',lake_color='#99ffff')
    m.drawcountries()    
    m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])

    # debug
    # draw observe points
    ## wspd[time = 814][lat = 73][lon = 144];
    '''
    for lt in lat:
        for ln in lon:
            if in_range(lt, ln):
                x,y = m(ln, lt)
                #m.plot(x, y, 'ro', markersize=5)
    '''

    # debug            
    # draw target area 
    ## target_area = (lon0, lat0), (lon1, lat1)
    target_areas = [[(143.207, 35.171), (122.992, 16.063)],
                    [(133.539, 16.063), (52.241, -11.290)],
                    [(71.225, 30.801), (48.134, 16.063)]]
    '''
    for target_area in target_areas:
        start_point = target_area[0]
        end_point   = target_area[1]
        for mode in ['start', 'end']:
            point = eval(mode + '_point')
            x,y = m(point[0], point[1])
            #m.plot(x, y, 'ro', markersize=10)
    '''

    # draw target points
    target_points = []
    for lt in lat:
        for ln in lon:
            if in_square(target_areas, ln, lt):
                x,y = m(ln, lt)
                m.plot(x, y, 'ro', markersize=5)
                target_points.append((ln, lt))

    # analyze wind speed
    result_dict = {}
    for current_time in time:
        tmp_result = []
        date_str   = calc_date(current_time).strftime("%Y/%m")
        print "analyzing ..."
        length = len(target_points)
        for index, target_point in enumerate(target_points):
            target_lon = target_point[0]
            target_lat = target_point[1]
            # calc index
            time_index = np.where(time==current_time)[0][0]
            lon_index  = np.where(lon==target_lon)[0][0]
            lat_index  = np.where(lat==target_lat)[0][0]
            wind_speed = wspd[time_index][lat_index][lon_index]
            tmp_result.append(tmp_result)
        result_dict[date_str] = np.average(tmp_result)
        print "%s done" % date_str
    output_json_path = '../results/weather_analysis_data.json'
    write_file_as_json(result_dict, output_json_path)
    
    # Add a colorbar and title, and then show the plot.
    plt.title('observed points\n'.title())
    plt.savefig('../results/weather_observed_points.png')
    print_with_notice("wave map on %s has been generated" % (target_date.strftime('%Y/%m/%d')))
    
if __name__ == '__main__':
    analyze()

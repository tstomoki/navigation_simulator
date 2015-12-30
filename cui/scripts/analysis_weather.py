# basic NOMADS OpenDAP extraction and plotting script
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import sys
import os
from pdb import *
import datetime
import pandas as pd
# import own modules #
sys.path.append('../public')
sys.path.append('../models')
from constants   import *
from my_modules  import *
# import own modules #

min_lon              = 0
max_lon              = 360
min_lat              = -60
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
        lon_flag = False
        lat_flag = False
        
        start_point = target_area[0]
        end_point   = target_area[1]

        # lon
        start_lon = start_point[0]
        end_lon   = end_point[0]
        if start_lon > end_lon:
            tmp_lon   = end_lon
            end_lon   = start_lon
            start_lon = tmp_lon

        if start_lon < 0 and end_lon > 0:
            tmp_lon = 360 + start_lon
            if ( (0 <= lon) and (lon < end_lon) ) or ( (tmp_lon <= lon) and (lon < 360) ):
                lon_flag = True
        elif start_lon < 0 and end_lon < 0:
            tmp_start_lon = 360 + start_lon
            tmp_end_lon   = 360 + end_lon
            if (tmp_start_lon <= lon) and (lon < tmp_end_lon):
                lon_flag = True                
        else:
            if (start_lon <= lon) and (lon < end_lon):
                lon_flag = True                
               
        # lat
        start_lat = start_point[1]
        end_lat   = end_point[1]
        if start_lat > end_lat:
            tmp_lat   = end_lat
            end_lat   = start_lat
            start_lat = tmp_lat

        if (start_lat <= lat) and (lat < end_lat):
            lat_flag = True                

        if lon_flag and lat_flag:
            return True

    return False

def calc_date(time):
    base_date = datetime.datetime(1800, 1,1)
    return base_date + datetime.timedelta(hours=time)

def analyze_target_areas(target_areas, area_name, lat, lon, time, wspd, m):
    # draw target points
    target_points = []
    for lt in lat:
        for ln in lon:
            if in_square(target_areas, ln, lt):
                x,y = m(ln, lt)
                m.plot(x, y, 'ro', markersize=5)
                target_points.append((ln, lt))
                
    # Add a colorbar and title, and then show the plot.
    title = "observed points in %s\n" % (area_name)
    plt.title(title.title())
    plt.savefig("../results/weather_observed_points_in_%s.png" % (area_name))

    # analyze wind speed
    result_dict = {}
    for current_time in time:
        tmp_result = np.array([])
        date_str   = calc_date(current_time).strftime("%Y/%m")
        length = len(target_points)
        for index, target_point in enumerate(target_points):
            target_lon = target_point[0]
            target_lat = target_point[1]
            # calc index
            time_index = np.where(time==current_time)[0][0]
            lon_index  = np.where(lon==target_lon)[0][0]
            lat_index  = np.where(lat==target_lat)[0][0]
            wind_speed = wspd[time_index][lat_index][lon_index]
            tmp_result = np.append(tmp_result, wind_speed)
        result_dict[date_str] = np.average(tmp_result)
        sys.stdout.write("\r %s done" % date_str)
    return result_dict

def init_basemap():
    plt.clf()    
    m    = Basemap(projection='mill',lat_ts=10,llcrnrlon=min_lon, \
                   urcrnrlon  = max_lon,llcrnrlat=min_lat,urcrnrlat=max_lat,           \
                   resolution = 'c')
    # Add a coastline and axis values.
    
    m.drawcoastlines(linewidth=0.5, color='#222222')
    m.drawmapboundary(fill_color='blue')
    m.fillcontinents(color='#71ff71',lake_color='#99ffff')
    m.drawcountries()    
    m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
    return m

def analyze():
    nc_file = '../data/weather_data/wspd.mon.mean.nc'
    url     = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/wspd.mon.mean.nc'
    fh      = netCDF4.Dataset(url)
    print 'got data'
    #fh     = netCDF4.Dataset(nc_file, mode='r')
    
    lat  = fh.variables['lat'][:]
    lon  = fh.variables['lon'][:]
    time = fh.variables['time'][:]
    wspd = fh.variables['wspd'][:]

    # second target area
    m = init_basemap()
    target_areas     = [[(60.4583, 26.2895), (50.2629, -34.1206)],
                       [(50.2629, -34.1206), (-56.6120, -50.1591)]]
    area_name        = 'arabia_us'
    result_dict      = analyze_target_areas(target_areas, area_name, lat, lon, time, wspd, m)
    output_json_path = "../results/weather_analysis_data_in_%s.json" % (area_name)
    write_file_as_json(result_dict, output_json_path)
    
    # third target area
    m = init_basemap()
    target_areas = [[(60.4583, 26.2895), (50.2629, -34.1206)],
                    [(50.2629, -34.1206), (-36.6120, -38.1591)],
                    [(-36.6120, -38.1591), (-46.6120, 26.2895)],
                    [(-46.6120, 26.2895), (-96.6120, 20.2895)]]
    
    area_name        = 'arabia_mid_us'
    result_dict      = analyze_target_areas(target_areas, area_name, lat, lon, time, wspd, m)
    output_json_path = "../results/weather_analysis_data_in_%s.json" % (area_name)
    write_file_as_json(result_dict, output_json_path)
    
    # first target area 
    ## target_area = (lon0, lat0), (lon1, lat1)
    m = init_basemap()
    target_areas = [[(143.207, 35.171), (122.992, 16.063)],
                    [(133.539, 16.063), (52.241, -11.290)],
                    [(71.225, 30.801), (48.134, 16.063)]]

    area_name        = 'japan_arabia'
    result_dict      = analyze_target_areas(target_areas, area_name, lat, lon, time, wspd, m)
    output_json_path = "../results/weather_analysis_data_in_%s.json" % (area_name)
    write_file_as_json(result_dict, output_json_path)

    return
    
def draw_incident_rate(json_file_path):
    # load json
    data = load_json_file(json_file_path)

    bf_data = {}
    for date, wspd in data.items():
        bf = calc_BF_from_wspd(wspd)
        if not bf_data.has_key(bf):
            bf_data[bf] = 0
        bf_data[bf] += 1
    data_num = sum(bf_data.values())
        
    # draw incident rate
    start_date = datetime.datetime(1995, 1,1,0,0)
    target_data = [ _v for _d, _v in data.items() if datetime.datetime.strptime(_d, "%Y/%m") > start_date]
    '''
    panda_frame = pd.DataFrame({'date': [datetime.datetime.strptime(_str, "%Y/%m") for _str in target_data.keys()],
                                
    '''
    panda_frame = pd.DataFrame({'wspd': target_data})
    # hist
    filepath = "../results/weather/wspd_in_%s.png" % (re.compile(r'in_(.+).json').search(json_file_path).groups()[0])
    plt.figure()
    graphInitializer('wind speed'.upper(), 'wind speed'.upper() + ' [m/s]', 'probability'.upper())
    panda_frame['wspd'].hist(color="#5F9BFF", alpha=.5)
    plt.xlim(7.0, 11.0)
    plt.ylim(0.0, 60.0)
    plt.legend(shadow=True)
    plt.legend(loc='upper right')        
    plt.savefig(filepath)
    plt.clf()
    plt.close()


    # bf
    '''
    bf_frame    = pd.DataFrame({'BF': bf_data.keys(),
                                'probability': num in bf_data.values()})
    filepath = "../results/weather/incident_in_%s.png" % (re.compile(r'in_(.+).json').search(json_file_path).groups()[0])
    plt.figure()
    graphInitializer('incident'.upper(), 'beaufort scale'.upper(), 'probability'.upper())
    bf_frame['probability'].hist(color="#5F9BFF", alpha=.5)
    plt.legend(shadow=True)
    plt.legend(loc='upper right')        
    plt.savefig(filepath)
    plt.clf()
    plt.close()    
    '''

    return 
    
if __name__ == '__main__':
    analyze()
    for loc in ['japan_arabia', 'arabia_us', 'arabia_mid_us']:
        json_path = "../results/weather_analysis_data_in_%s.json" % loc
        draw_incident_rate(json_path)

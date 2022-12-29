import pandas as pd
import numpy as np
import re
import difflib
import SQL_connect
from geopy.distance import geodesic
from glspred import preprocess, distcal, extraction
import time

start = time.time()
dbconn = SQL_connect.DBConnectionRS()

# read in predicting data and master table
pred_raw = pd.read_csv(r'G:\REA\Working files\land-bidding\pipeline\Parcel for land bidding prediction.csv')
master = pd.read_csv(r'G:\REA\Working files\land-bidding\pipeline\gls_origin.csv')

poi_df = dbconn.read_data('''select poi_name, poi_type, poi_subtype, poi_lat, poi_long, location_poi_dwid, town
                             from masterdata_sg.poi''')

mrt_dist_master = dbconn.read_data('''select * from data_science.sg_land_parcel_mrt_distance''')
bus_dist_master = dbconn.read_data('''select * from data_science.sg_land_parcel_bus_stop_distance''')
sch_dist_master = dbconn.read_data('''select * from data_science.sg_land_parcel_school_distance''')
infra_dist_master = dbconn.read_data('''select * from data_science.sg_land_parcel_distance_to_infrastructure''')

parcel_dist_master = dbconn.read_data('''select * from data_science.sg_land_parcel_pairwise_distance''')
project = dbconn.read_data('''select project_dwid, project_name, project_type_code, completion_year, location_marker 
                                from masterdata_sg.project ''')
proj_dist_master = dbconn.read_data('''select * from data_science.sg_land_parcel_distance_to_project''')
proj_dist_info_master = dbconn.read_data(
    '''select * from data_science.sg_land_parcel_filled_info_distance_to_project''')

print('Read-in completed: {:.3f}s'.format(time.time() - start))

# preprocessing: align header and concat data
master['predicting'] = 0
pred_raw['predicting'] = 1
pred = pred_raw.copy()
num_pred_parcels = pred.shape[0]
processor = preprocess.Preprocess()
pred_processed = processor.process(pred, master)
master_new = pd.concat([master, pred_processed], ignore_index=True)

print('Preprocess completed: {:.3f}s'.format(time.time() - start))

# calculate pairwise distances
# infrastructure
distance_limit = 5000
poi_df = poi_df.rename(columns={'poi_lat': 'latitude', 'poi_long': 'longitude'})
poi_mrt = poi_df[poi_df.poi_subtype == 'mrt station'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_bus = poi_df[poi_df.poi_subtype == 'bus stop'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_sch = poi_df[poi_df.poi_type == 'school'].drop_duplicates(subset='poi_name').reset_index(drop=True)

# special wrangling for mrt poi
poi_mrt['mrt_line'] = poi_mrt.poi_name.apply(extraction.extract_bracketed)\
    .apply(lambda x: re.sub(r' ?/ ?', '/', x[0]))\
    .apply(lambda x: ''.join([w for w in list(x) if not w.isnumeric()]))\
    .apply(lambda x: x.split('/'))
# by right the line code should be 2-char (cc, ne, dt...), if not, means problematic
poi_mrt['line_code_check'] = poi_mrt.mrt_line.apply(lambda x: 0 if any(len(code) != 2 for code in x) else 1)

# # this can match the closest name, use when necessary
# mrt_correct = poi_mrt[poi_mrt.line_code_check == 1].poi_name
# difflib.get_close_matches('ocbc buona vista mrt station', mrt_correct, n=1, cutoff=0.7)

# here just drop those with code issues
poi_mrt = poi_mrt[poi_mrt.line_code_check == 1]
poi_mrt['mrt_station_name'] = poi_mrt.poi_name.apply(extraction.remove_brackets, remove_content=True)
poi_mrt['num_lines_raw'] = poi_mrt.mrt_line.str.len()
mrt = poi_mrt[['poi_name', 'mrt_station_name', 'num_lines_raw', 'mrt_line']]
mrt_num_lines = mrt.groupby('mrt_station_name').sum('num_lines_raw').reset_index().rename(columns={'num_lines_raw': 'num_lines'})
mrt_line_linkage = mrt.merge(mrt_num_lines, how='left', on='mrt_station_name')\
    .rename(columns={'mrt_station_name': 'mrt_station'})\
    .drop_duplicates(subset=['mrt_station'])

# transform for land parcels
master_new['coordinates'] = list(zip(list(master_new.latitude), list(master_new.longitude)))
master_new['year_launch'] = master_new.date_launch.apply(lambda date: int(date.split('/')[-1]))
master_new['month_launch'] = master_new.date_launch.apply(lambda date: int(date.split('/')[1]))
master_new['day_launch'] = master_new.date_launch.apply(lambda date: int(date.split('/')[0]))
master_new['date_launch_index'] = master_new.year_launch.astype(str) + \
                                  master_new.month_launch.astype(str).apply(lambda x: x.zfill(2)) + \
                                  master_new.day_launch.astype(str).apply(lambda x: x.zfill(2))
master_new['date_launch_index'] = master_new['date_launch_index'].astype(int)

master_new = master_new.sort_values(by=['year_launch', 'month_launch', 'day_launch'])
master_unique_land_parcels = master_new.drop_duplicates(subset=['land_parcel_id'], keep='last')

poi_cols = ['land_parcel_id',
            'land_parcel_name',
            'latitude',
            'longitude',
            'coordinates',
            'year_launch',
            'month_launch',
            'day_launch',
            'date_launch_index']

land_parcel_poi = master_new[poi_cols].drop_duplicates(subset='land_parcel_id').reset_index(drop=True)
pred = master_new[master_new.predicting == 1]
pred_parcels_poi = pred[poi_cols]

distcalculator = distcal.PairwiseDistCalculator(poi_master=pred_parcels_poi, master_id='land_parcel_id')
# execute function for infrastructure
mrt_distance = distcalculator.calculate_distance(poi_ref=poi_mrt, ref_id='poi_name', distance_limit=distance_limit) \
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'mrt_station'})
bus_distance = distcalculator.calculate_distance(poi_ref=poi_bus, ref_id='poi_name', distance_limit=distance_limit) \
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'bus_stop'})
sch_distance = distcalculator.calculate_distance(poi_ref=poi_sch, ref_id='poi_name', distance_limit=distance_limit) \
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'school'})

# find the nearest poi to parcels
nearby_mrt_pred = distcalculator.find_nearby(mrt_distance, {'distance_m': 'dist_to_mrt'})
nearby_bus_pred = distcalculator.find_nearby(bus_distance, {'distance_m': 'dist_to_bus_stop'})
nearby_sch_pred = distcalculator.find_nearby(sch_distance, {'distance_m': 'dist_to_school'})

# calculate distance to cbd
sg_cbd_coord = (1.2884113726733633, 103.85252198698596)
dist_to_cbd = pd.DataFrame(
    {'dist_to_cbd': pred_parcels_poi.coordinates.apply(lambda x: round(geodesic(x, sg_cbd_coord).m), 2)})

# merge into distance-to-infrastructure table
dist_to_infra = pd.concat([nearby_mrt_pred,
                           nearby_bus_pred,
                           nearby_sch_pred,
                           dist_to_cbd], axis=1).rename_axis('land_parcel_id').reset_index()

# calculate parcel-parcel distances and project-parcel distances
# parcel-parcel distances
parcel_distance = distcalculator.calculate_distance(poi_ref=land_parcel_poi, ref_id='land_parcel_id',
                                                    distance_limit=distance_limit)
# add timings
parcel_distance_with_time = parcel_distance.merge(pred[['land_parcel_id', 'date_launch_index']],
                                                  how='left',
                                                  left_on='poi_a',
                                                  right_on='land_parcel_id') \
    .rename(columns={'date_launch_index': 'date_a'}) \
    .drop('land_parcel_id', axis=1) \
    .merge(master_new[['land_parcel_id', 'date_launch_index']],
           how='left',
           left_on='poi_b',
           right_on='land_parcel_id') \
    .rename(columns={'date_launch_index': 'date_b'}) \
    .drop('land_parcel_id', axis=1) \
    .rename(columns={'poi_a': 'land_parcel_id_a',
                     'poi_b': 'land_parcel_id_b',
                     'date_a': 'land_parcel_a_launch_date',
                     'date_b': 'land_parcel_b_launch_date'})

timer = distcal.TimeMaster()
parcel_distance_with_time['launch_time_diff_days'] = timer.time_diff_by_index(parcel_distance_with_time,
                                                                              'land_parcel_a_launch_date',
                                                                              'land_parcel_b_launch_date',
                                                                              time_format='%Y%m%d',
                                                                              time_unit='D')
parcel_distance_with_time[['land_parcel_a_launch_date',
                           'land_parcel_b_launch_date']] = parcel_distance_with_time[['land_parcel_a_launch_date',
                                                                                      'land_parcel_b_launch_date']].astype(int)

# parcel-project distances
project['latitude'] = project.location_marker.apply(extraction.extract_num, decimal=True).apply(
    lambda x: -999 if np.isnan(x).any() else x[1])
project['longitude'] = project.location_marker.apply(extraction.extract_num, decimal=True).apply(
    lambda x: -999 if np.isnan(x).any() else x[0])
project_poi = project.drop(project[(project.longitude.abs() > 180) | (project.latitude.abs() > 90)].index, axis=0)
proj_distance = distcalculator \
    .calculate_distance(poi_ref=project_poi, ref_id='project_dwid', distance_limit=distance_limit) \
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'project_dwid'})

# merge all tables
# de-duplicate
pred_pcl_id = pred.land_parcel_id
mrt_dist_master = mrt_dist_master[~mrt_dist_master.land_parcel_id.isin(pred_pcl_id)]
bus_dist_master = bus_dist_master[~bus_dist_master.land_parcel_id.isin(pred_pcl_id)]
sch_dist_master = sch_dist_master[~sch_dist_master.land_parcel_id.isin(pred_pcl_id)]
infra_dist_master = infra_dist_master[~infra_dist_master.land_parcel_id.isin(pred_pcl_id)]
parcel_dist_master = parcel_dist_master[~parcel_dist_master.land_parcel_id_a.isin(pred_pcl_id)]
proj_dist_master = proj_dist_master[~proj_dist_master.land_parcel_id.isin(pred_pcl_id)]
proj_dist_info_master = proj_dist_info_master[~proj_dist_info_master.land_parcel_id.isin(pred_pcl_id)]

# concat infrastructure data
merged_mrt_dist = pd.concat([mrt_dist_master, mrt_distance], ignore_index=True)
merged_bus_dist = pd.concat([bus_dist_master, bus_distance], ignore_index=True)
merged_sch_dist = pd.concat([sch_dist_master, sch_distance], ignore_index=True)
merged_infra_dist = pd.concat([infra_dist_master, dist_to_infra], ignore_index=True)

# for mrt dist, de-duplicate
mrt_name_dict = dict(zip(list(mrt_line_linkage.poi_name), list(mrt_line_linkage.mrt_station)))
merged_mrt_dist['mrt_station'] = merged_mrt_dist.mrt_station.map(mrt_name_dict)
merged_mrt_dist['key'] = merged_mrt_dist.land_parcel_id + merged_mrt_dist.mrt_station
merged_mrt_dist = merged_mrt_dist.sort_values(by=['key', 'distance_m'], ascending=False).drop_duplicates(subset=['key'], keep='last')
merged_mrt_dist.drop('key', axis=1, inplace=True)

# merge into num of lines of each mrt station
merged_mrt_dist = merged_mrt_dist.merge(mrt_line_linkage[['mrt_station', 'num_lines']], how='left', on='mrt_station')

# concat parcel-parcel data
merged_parcel_dist = pd.concat([parcel_dist_master, parcel_distance_with_time], ignore_index=True)

# concat parcel-project data
merged_proj_dist = pd.concat([proj_dist_master, proj_distance], ignore_index=True)

# # upload
# if mrt_dist_master.shape[0] + mrt_distance.shape[0] == merged_mrt_dist.shape[0]:
#     dbconn.copy_from_df(merged_mrt_dist, "data_science.sg_land_parcel_mrt_distance")
# else:
#     print("Dist table: land to mrt, upload failed")
#
# if bus_dist_master.shape[0] + bus_distance.shape[0] == merged_bus_dist.shape[0]:
#     dbconn.copy_from_df(merged_bus_dist, "data_science.sg_land_parcel_bus_stop_distance")
# else:
#     print("Dist table: land to bus stop, upload failed")
#
# if sch_dist_master.shape[0] + sch_distance.shape[0] == merged_sch_dist.shape[0]:
#     dbconn.copy_from_df(merged_sch_dist, "data_science.sg_land_parcel_school_distance")
# else:
#     print("Dist table: land to school, upload failed")
#
# if infra_dist_master.shape[0] + dist_to_infra.shape[0] == merged_infra_dist.shape[0]:
#     dbconn.copy_from_df(merged_infra_dist, "data_science.sg_land_parcel_distance_to_infrastructure")
# else:
#     print("Dist table: land to infrastructure, upload failed")
#
# if parcel_dist_master.shape[0] + parcel_distance_with_time.shape[0] == merged_parcel_dist.shape[0]:
#     dbconn.copy_from_df(merged_parcel_dist, "data_science.sg_land_parcel_pairwise_distance")
# else:
#     print("Dist table: land to land, upload failed")
#
# if proj_dist_master.shape[0] + proj_distance.shape[0] == merged_proj_dist.shape[0]:
#     dbconn.copy_from_df(merged_proj_dist, "data_science.sg_land_parcel_distance_to_project")
# else:
#     print("Dist table: land to project, upload failed")


# further wrangling
proj_dist_info = proj_distance.merge(pred[['land_parcel_id', 'land_use_type', 'project_type', 'year_launch']],
                                     how='left', on='land_parcel_id') \
    .merge(project[['project_dwid', 'project_type_code', 'completion_year']], how='left', on='project_dwid') \
    .rename(columns={'project_type': 'land_project_type',
                     'year_launch': 'land_launch_year',
                     'completion_year': 'proj_completion_year'})
proj_dist_info['project_type_group'] = proj_dist_info.project_type_code.apply(processor.land_use_regroup)
merged_proj_dist_info = pd.concat([proj_dist_info_master, proj_dist_info])

# # upload
# if proj_dist_info_master.shape[0] + proj_dist_info.shape[0] == merged_proj_dist_info.shape[0]:
#     dbconn.copy_from_df(merged_proj_dist_info, 'data_science.sg_land_parcel_filled_info_distance_to_project')
# else:
#     print("Dist table: land to project (filled info), upload failed")

print('Calculation completed: {:.3f}s'.format(time.time() - start))

check = 42

# feature engineering
cat_cols = ['region',  #
            'zone',  #
            'land_use_type',  # 'devt_class',
            'project_type',  # 'devt_type',
            'source'
            ]
num_cols = ['latitude',
            'longitude',
            'site_area_sqm',
            'gpr',
            'lease_term',
            'num_bidders',  # should be dynamic
            'joint_venture',  # should be dynamic
            'year_launch',
            'timediff_launch_to_close',  #
            'proj_num_of_units',  # should be dynamic
            'proj_max_floor',  # should be dynamic
            'num_nearby_parcels_3km_past_6month',  # use format
            'num_proj_nearby_2km_past_5years',  # use format
            'num_mrt_1km',  #
            'num_bus_stop_500m',  #
            'num_school_1km',  #
            'dist_to_nearest_parcel_launched_past_6month',  #
            'dist_to_nearest_proj_completed_past_5years',  #
            'dist_to_cbd',
            'dist_to_mrt',
            'dist_to_bus_stop',
            'dist_to_school',
            'comparable_price_psm_gfa'  #
            ]

### fill in zone info based on lat and lon

# timediff_launch_to_close
master_new['timediff_launch_to_close'] = pd.to_datetime(master_new.date_close, format='%d/%m/%Y') - \
                                         pd.to_datetime(master_new.date_launch, format='%d/%m/%Y')
master_new['timediff_launch_to_close'] = master_new.timediff_launch_to_close.apply(lambda x: x.days)

# number of nearby parcels
parcel_dist_lim = 3  # km
parcel_time_lim = 6  # months

merged_parcel_dist_info = merged_parcel_dist.merge(master_unique_land_parcels[['land_parcel_id', 'land_use_type']],
                                                      how='left',
                                                      left_on='land_parcel_id_a',
                                                      right_on='land_parcel_id').rename(columns={'land_use_type': 'land_use_type_a'})\
                                                .drop('land_parcel_id', axis=1)\
                                                .merge(master_unique_land_parcels[['land_parcel_id', 'land_use_type']],
                                                      how='left',
                                                      left_on='land_parcel_id_b',
                                                      right_on='land_parcel_id').rename(columns={'land_use_type': 'land_use_type_b'})\
                                                .drop('land_parcel_id', axis=1)

parcels_selected = merged_parcel_dist_info[(merged_parcel_dist_info.distance_m <= parcel_dist_lim * 1000) &
                                           (merged_parcel_dist_info.distance_m > 0) &
                                           (merged_parcel_dist_info.launch_time_diff_days < 30 * parcel_time_lim) &
                                           (merged_parcel_dist_info.launch_time_diff_days >= 0) &
                                           (merged_parcel_dist_info.land_use_type_a == merged_parcel_dist_info.land_use_type_b)]

nearby_pcls = parcels_selected[['land_parcel_id_a', 'land_parcel_id_b']]\
    .groupby('land_parcel_id_a').count() \
    .reset_index() \
    .rename(columns={'land_parcel_id_a': 'land_parcel_id',
                     'land_parcel_id_b': f'num_nearby_parcels_{parcel_dist_lim}km_past_{parcel_time_lim}month'})

# distance to nearest parcel launched within past months
parcels_nearest = merged_parcel_dist_info[(merged_parcel_dist_info.distance_m > 0) &
                                          (merged_parcel_dist_info.launch_time_diff_days < 30 * parcel_time_lim) &
                                          (merged_parcel_dist_info.launch_time_diff_days >= 0) &
                                          (merged_parcel_dist_info.land_use_type_a == merged_parcel_dist_info.land_use_type_b)]

parcels_nearest = parcels_nearest[['land_parcel_id_a', 'distance_m']]\
    .groupby('land_parcel_id_a').min('distance_m') \
    .reset_index() \
    .rename(columns={'land_parcel_id_a': 'land_parcel_id',
                     'distance_m': f'dist_to_nearest_parcel_launched_past_{parcel_time_lim}month'})


# number of nearby proj
proj_dist_lim = 2  # km
proj_time_lim = 5  # year

proj_selected = merged_proj_dist_info[(merged_proj_dist_info.distance_m > 0) &
                                      (merged_proj_dist_info.distance_m <= proj_dist_lim * 1000) &
                                      (merged_proj_dist_info.land_launch_year - merged_proj_dist_info.proj_completion_year <= proj_time_lim) &
                                      (0 < merged_proj_dist_info.land_launch_year - merged_proj_dist_info.proj_completion_year) &
                                      (merged_proj_dist_info.land_use_type == merged_proj_dist_info.project_type_group)]

nearby_proj = proj_selected[['land_parcel_id', 'project_dwid']]\
    .groupby('land_parcel_id').count() \
    .reset_index() \
    .rename(columns={'project_dwid': f'num_proj_nearby_{proj_dist_lim}km_past_{proj_time_lim}years'})

# distance to nearest project completed within past years
proj_nearest = merged_proj_dist_info[(merged_proj_dist_info.distance_m > 0) &
                                     (merged_proj_dist_info.land_launch_year - merged_proj_dist_info.proj_completion_year <= proj_time_lim) &
                                     (0 < merged_proj_dist_info.land_launch_year - merged_proj_dist_info.proj_completion_year) &
                                     (merged_proj_dist_info.land_use_type == merged_proj_dist_info.project_type_group)]

proj_nearest = proj_nearest[['land_parcel_id', 'distance_m']]\
    .groupby('land_parcel_id').min('distance_m') \
    .reset_index() \
    .rename(columns={'distance_m': f'dist_to_nearest_proj_completed_past_{proj_time_lim}years'})


# number of nearby mrt stations
mrt_dist_lim = 1  # km
nearby_mrt = merged_mrt_dist[merged_mrt_dist.distance_m <= mrt_dist_lim * 1000][['land_parcel_id', 'distance_m']]\
    .groupby('land_parcel_id')\
    .count() \
    .reset_index() \
    .rename(columns={'distance_m': f'num_mrt_{mrt_dist_lim}km'})

# number of nearby mrt lines
nearby_mrt_lines = merged_mrt_dist[merged_mrt_dist.distance_m <= mrt_dist_lim * 1000][['land_parcel_id', 'num_lines']]\
    .groupby('land_parcel_id')\
    .sum('num_lines')\
    .reset_index() \
    .rename(columns={'num_lines': f'num_mrt_lines_{mrt_dist_lim}km'})


# number of nearby bus stop
bus_stop_dist_lim = 500  # meters
nearby_bus = merged_bus_dist[merged_bus_dist.distance_m <= bus_stop_dist_lim][['land_parcel_id', 'distance_m']]\
    .groupby('land_parcel_id')\
    .count() \
    .reset_index() \
    .rename(columns={'distance_m': f'num_bus_stop_{bus_stop_dist_lim}m'})


# number of nearby school
school_dist_lim = 1  # km
nearby_sch = merged_sch_dist[merged_sch_dist.distance_m <= school_dist_lim * 1000][['land_parcel_id', 'distance_m']]\
    .groupby('land_parcel_id')\
    .count()\
    .reset_index()\
    .rename(columns={'distance_m': f'num_school_{school_dist_lim}km'})


# merge all to master table
feature_df = master_new.merge(nearby_pcls, how='left', on='land_parcel_id')\
    .merge(nearby_proj, how='left', on='land_parcel_id')\
    .merge(nearby_mrt, how='left', on='land_parcel_id')\
    .merge(nearby_mrt_lines, how='left', on='land_parcel_id')\
    .merge(nearby_bus, how='left', on='land_parcel_id')\
    .merge(nearby_sch, how='left', on='land_parcel_id')\
    .merge(parcels_nearest, how='left', on='land_parcel_id')\
    .merge(proj_nearest, how='left', on='land_parcel_id')\
    .merge(merged_infra_dist, how='left', on='land_parcel_id')

# fillna for certain columns
fill_na_cols = [f'num_nearby_parcels_{parcel_dist_lim}km_past_{parcel_time_lim}month',
                f'num_proj_nearby_{proj_dist_lim}km_past_{proj_time_lim}years',
                f'dist_to_nearest_proj_completed_past_{proj_time_lim}years',
                f'num_mrt_{mrt_dist_lim}km',
                f'num_mrt_lines_{mrt_dist_lim}km',
                f'num_bus_stop_{bus_stop_dist_lim}m',
                f'num_school_{school_dist_lim}km',
                ]
fill_na_df = pd.DataFrame(np.zeros((feature_df.shape[0], len(fill_na_cols))), columns=fill_na_cols)
feature_df = feature_df.fillna(fill_na_df)

fill_na_dist_cols = [f'dist_to_nearest_parcel_launched_past_{parcel_time_lim}month',
                     f'dist_to_nearest_proj_completed_past_{proj_time_lim}years',
                     'dist_to_mrt',
                     'dist_to_bus_stop',
                     'dist_to_school']
fill_na_dist_df = pd.DataFrame(np.full((feature_df.shape[0], len(fill_na_dist_cols)), 5001), columns=fill_na_dist_cols)
feature_df = feature_df.fillna(fill_na_dist_df)

print('Feature engineering completed: {:.3f}s'.format(time.time() - start))


check = 42

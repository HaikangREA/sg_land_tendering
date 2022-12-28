import pandas as pd
import numpy as np
import SQL_connect
from geopy.distance import geodesic
from glspred import preprocess, distcal, extraction

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
proj_dist_info_master = dbconn.read_data('''select * from data_science.sg_land_parcel_filled_info_distance_to_project''')


# preprocessing: align header and concat data
master['predicting'] = 0
pred_raw['predicting'] = 1
pred = pred_raw.copy()
num_pred_parcels = pred.shape[0]
processor = preprocess.Preprocess()
pred_processed = processor.process(pred, master)
master_new = pd.concat([master, pred_processed], ignore_index=True)

# calculate pairwise distances
# infrastructure
distance_limit = 5000
poi_df = poi_df.rename(columns={'poi_lat': 'latitude', 'poi_long': 'longitude'})
poi_mrt = poi_df[poi_df.poi_subtype == 'mrt station'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_bus = poi_df[poi_df.poi_subtype == 'bus stop'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_sch = poi_df[poi_df.poi_type == 'school'].drop_duplicates(subset='poi_name').reset_index(drop=True)

# transform for land parcels
master_new['coordinates'] = list(zip(list(master_new.latitude), list(master_new.longitude)))
master_new['year_launch'] = master_new.date_launch.apply(lambda date: date.split('/')[-1])
master_new['month_launch'] = master_new.date_launch.apply(lambda date: date.split('/')[1])
master_new['day_launch'] = master_new.date_launch.apply(lambda date: date.split('/')[0])
master_new['date_launch_index'] = master_new.year_launch.astype(str) + \
                                  master_new.month_launch.astype(str).apply(lambda x: x.zfill(2)) + \
                                  master_new.day_launch.astype(str).apply(lambda x: x.zfill(2))

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
bus_distance = distcalculator.calculate_distance(poi_ref=poi_bus, ref_id='poi_name', distance_limit=distance_limit)\
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'bus_stop'})
sch_distance = distcalculator.calculate_distance(poi_ref=poi_sch, ref_id='poi_name', distance_limit=distance_limit) \
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'school'})

# find the nearest poi to parcels
nearby_mrt = distcalculator.find_nearby(mrt_distance, {'distance_m': 'dist_to_mrt'})
nearby_bus = distcalculator.find_nearby(bus_distance, {'distance_m': 'dist_to_bus_stop'})
nearby_sch = distcalculator.find_nearby(sch_distance, {'distance_m': 'dist_to_school'})

# calculate distance to cbd
sg_cbd_coord = (1.2884113726733633, 103.85252198698596)
dist_to_cbd = pd.DataFrame(
    {'dist_to_cbd': pred_parcels_poi.coordinates.apply(lambda x: round(geodesic(x, sg_cbd_coord).m), 2)})

# merge into distance-to-infrastructure table
dist_to_infra = pd.concat([nearby_mrt,
                           nearby_bus,
                           nearby_sch,
                           dist_to_cbd], axis=1).rename_axis('land_parcel_id').reset_index()

# calculate parcel-parcel distances and project-parcel distances
# parcel-parcel distances
parcel_distance = distcalculator.calculate_distance(poi_ref=land_parcel_poi, ref_id='land_parcel_id',
                                                    distance_limit=distance_limit)
# add timings
parcel_distance_with_time = parcel_distance.merge(pred[['land_parcel_id', 'date_launch_index']],
                                                  how='left',
                                                  left_on='poi_a',
                                                  right_on='land_parcel_id')\
                                            .rename(columns={'date_launch_index': 'date_a'})\
                                            .drop('land_parcel_id', axis=1)\
                                            .merge(master_new[['land_parcel_id', 'date_launch_index']],
                                                   how='left',
                                                   left_on='poi_b',
                                                   right_on='land_parcel_id')\
                                            .rename(columns={'date_launch_index': 'date_b'})\
                                            .drop('land_parcel_id', axis=1)\
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
project['latitude'] = project.location_marker.apply(extraction.extract_num, decimal=True).apply(lambda x: -999 if np.isnan(x).any() else x[1])
project['longitude'] = project.location_marker.apply(extraction.extract_num, decimal=True).apply(lambda x: -999 if np.isnan(x).any() else x[0])
project_poi = project.drop(project[(project.longitude.abs() > 180) | (project.latitude.abs() > 90)].index, axis=0)
proj_distance = distcalculator\
    .calculate_distance(poi_ref=project_poi, ref_id='project_dwid', distance_limit=distance_limit)\
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'project_dwid'})

# merge all tables and upload
# concat infrastructure data
merged_mrt_dist = pd.concat([mrt_dist_master, mrt_distance], ignore_index=True)
merged_bus_dist = pd.concat([bus_dist_master, bus_distance], ignore_index=True)
merged_sch_dist = pd.concat([sch_dist_master, sch_distance], ignore_index=True)
merged_infra_dist = pd.concat([infra_dist_master, dist_to_infra], ignore_index=True)

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
proj_dist_info = proj_distance.merge(pred[['land_parcel_id', 'land_use_type', 'project_type', 'year_launch']], how='left', on='land_parcel_id')\
                                .merge(project[['project_dwid', 'project_type_code', 'completion_year']], how='left', on='project_dwid')\
                                .rename(columns={'project_type': 'land_project_type',
                                                 'year_launch': 'land_launch_year',
                                                 'completion_year': 'proj_completion_year'})
proj_dist_info['project_type_group'] = proj_dist_info.project_type_code.apply(processor.land_use_regroup)
merged_proj_dist_info = pd.concat([proj_dist_info_master, proj_dist_info])

# copy from df




check = 42

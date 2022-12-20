# to calculate pairwise distance between poi (for land parcels, can also calculate the time difference in launch)
# input: poi table a, poi table b (both with cols [id, latitude, longitude])
# output: df[id_a, id_b, distance]


import pandas as pd
import numpy as np
import SQL_connect
from geopy.distance import geodesic
from tqdm import tqdm
import hashlib
from datetime import date, datetime

dbconn = SQL_connect.DBConnectionRS()


def get_uuid(id_text: str):
    try:
        return hashlib.sha256(id_text.encode('utf-8')).hexdigest()
    except:
        return


def calculate_distance(poi_a, key_a, poi_b, key_b, distance_limit=5000):
    loop_a, loop_b = poi_a[key_a], poi_b[key_b]
    poi_a.index = loop_a
    poi_b.index = loop_b
    output = pd.DataFrame(columns=['poi_a',
                                   'poi_b',
                                   'distance'])
    for id_a in tqdm(loop_a, desc='Main'):
        coord_a = (poi_a.loc[id_a].latitude, poi_a.loc[id_a].longitude)
        for id_b in loop_b:
            coord_b = (poi_b.loc[id_b].latitude, poi_b.loc[id_b].longitude)
            try:
                distance = round(geodesic(coord_a, coord_b).m, 2)
            except ValueError:
                distance = -1
            to_append = [id_a, id_b, distance]
            if 0 <= distance <= distance_limit:
                output.loc[len(output)] = to_append

    return output


poi_df = dbconn.read_data("""select poi_name , poi_type , poi_subtype , poi_lat , poi_long , location_poi_dwid  , town
                             from masterdata_sg.poi
                             ;""")
gls = dbconn.read_data("""select * from data_science.sg_new_full_land_bidding_filled_features;""")
pred = dbconn.read_data("""select * from data_science.sg_gls_land_parcel_for_prediction""")

poi_df = poi_df.rename(columns={'poi_lat': 'latitude', 'poi_long': 'longitude'})
poi_mrt = poi_df[poi_df.poi_subtype == 'mrt station'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_bus = poi_df[poi_df.poi_subtype == 'bus stop'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_sch = poi_df[poi_df.poi_type == 'school'].drop_duplicates(subset='poi_name').reset_index(drop=True)
cols = ['land_parcel_id',
        'land_parcel_std',
        'latitude',
        'longitude',
        'year_launch',
        'month_launch',
        'day_launch']
poi_land_parcel = gls[cols].drop_duplicates(subset='land_parcel_std').reset_index(drop=True)
pred_parcels_poi = pred[cols]

# execute function for infrastructure
mrt_distance = calculate_distance(pred_parcels_poi, 'land_parcel_id', poi_mrt, 'poi_name', distance_limit=5000)\
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'mrt_station'})
bus_distance = calculate_distance(pred_parcels_poi, 'land_parcel_id', poi_bus, 'poi_name', distance_limit=5000)\
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'bus_stop'})
sch_distance = calculate_distance(pred_parcels_poi, 'land_parcel_id', poi_sch, 'poi_name', distance_limit=5000)\
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'school'})
check = 42

# read in and merge into infrastructure tables
mrt_dist_master = dbconn.read_data('''select * from data_science.sg_land_parcel_mrt_distance''')
bus_dist_master = dbconn.read_data('''select * from data_science.sg_land_parcel_bus_stop_distance''')
sch_dist_master = dbconn.read_data('''select * from data_science.sg_land_parcel_school_distance''')
length_mrt = mrt_dist_master.shape[0]
length_bus = bus_dist_master.shape[0]
length_sch = sch_dist_master.shape[0]
# merge in
mrt_dist_master = pd.concat([mrt_dist_master, mrt_distance[mrt_dist_master.columns]])
bus_dist_master = pd.concat([bus_dist_master, bus_distance[bus_dist_master.columns]])
sch_dist_master = pd.concat([sch_dist_master, sch_distance[sch_dist_master.columns]])
# upload
if mrt_dist_master.shape[0] == length_mrt + mrt_distance.shape[0]:
    dbconn.copy_from_df(mrt_dist_master, "data_science.sg_land_parcel_mrt_distance")
else:
    print('Error in uploading for MRT station distance')

if bus_dist_master.shape[0] == length_bus + bus_distance.shape[0]:
    dbconn.copy_from_df(bus_dist_master, "data_science.sg_land_parcel_bus_stop_distance")
else:
    print('Error in uploading for bus stop distance')

if sch_dist_master.shape[0] == length_sch + sch_distance.shape[0]:
    dbconn.copy_from_df(sch_dist_master, "data_science.sg_land_parcel_school_distance")
else:
    print('Error in uploading for school distance')

# create land parcel distance to infrastructure summary table
land_to_infra = dbconn.read_data('''with
                                    mrt as(
                                    select land_parcel_id , min(distance) as dist_to_mrt 
                                    from data_science.sg_land_parcel_mrt_distance
                                    group by 1)
                                    ,
                                    bus as(
                                    select land_parcel_id , min(distance) as dist_to_bus_stop
                                    from data_science.sg_land_parcel_bus_stop_distance
                                    group by 1)
                                    ,
                                    sch as(
                                    select land_parcel_id , min(distance) as dist_to_school
                                    from data_science.sg_land_parcel_school_distance
                                    group by 1)
                                    select *
                                    from mrt
                                        left join bus using (land_parcel_id)
                                        left join sch using (land_parcel_id)
                                    ;''')
if len(land_to_infra) > 0:
    dbconn.copy_from_df(land_to_infra, 'data_science.sg_land_parcel_distance_to_infrastructure')


check = 42





# below for land parcel
parcel_distance = calculate_distance(pred_parcels_poi, 'land_parcel_id', poi_land_parcel, 'land_parcel_id')

# add in time dimension:
# create date index
poi_land_parcel['date_index'] = poi_land_parcel.year_launch.astype(str) + poi_land_parcel.month_launch.astype(
    str).apply(lambda x: x.zfill(2)) + poi_land_parcel.day_launch.astype(str).apply(lambda x: x.zfill(2))
pred_parcels_poi['date_index'] = pred_parcels_poi.year_launch.astype(str) + pred_parcels_poi.month_launch.astype(
    str).apply(lambda x: x.zfill(2)) + pred_parcels_poi.day_launch.astype(str).apply(lambda x: x.zfill(2))

# join time dimension to result df
pairwise_dist = parcel_distance \
    .merge(pred_parcels_poi.reset_index(drop=True)[['land_parcel_id', 'date_index']],
           how='left',
           left_on='poi_a',
           right_on='land_parcel_id') \
    .drop('land_parcel_id', axis=1) \
    .rename(columns={'date_index': 'a_date'}) \
    .merge(poi_land_parcel.reset_index(drop=True)[['land_parcel_id', 'date_index']],
           how='left',
           left_on='poi_b',
           right_on='land_parcel_id') \
    .drop('land_parcel_id', axis=1) \
    .rename(columns={'date_index': 'b_date'}) \

# calculate time difference
pairwise_dist['launch_time_diff_days'] = (pairwise_dist.a_date.apply(lambda x: datetime.strptime(x, '%Y%m%d')) - \
                                          pairwise_dist.b_date.apply(
                                              lambda x: datetime.strptime(x, '%Y%m%d'))) / np.timedelta64(1, 'D')

pairwise_dist['launch_time_diff_days'] = pairwise_dist.launch_time_diff_days.astype(int)

# transform for uploading
pairwise_dist = pairwise_dist.rename(columns={'poi_a': 'land_parcel_id_a',
                                              'poi_b': 'land_parcel_id_b',
                                              'distance': 'distance_m'})
pairwise_dist[['a_date', 'b_date']] = pairwise_dist[['a_date', 'b_date']].astype(int)


# read in pairwise distance table
pw_dist_master = dbconn.read_data('''select * from data_science.sg_gls_pairwise_nearby_land_parcels''')
length = pw_dist_master.shape[0]
pw_dist_master = pd.concat([pw_dist_master, pairwise_dist[pw_dist_master.columns]])

check = 42
if pw_dist_master.shape[0] == length + pairwise_dist.shape[0]:
    dbconn.copy_from_df(pw_dist_master, "data_science.sg_gls_pairwise_nearby_land_parcels")
else:
    print('Error in uploading parcel pairwise distances')


# # upload tables
# dbconn.copy_from_df(
#     mrt_distance,
#     "data_science.sg_land_parcel_mrt_distance",
# )
# dbconn.copy_from_df(
#     bus_distance,
#     "data_science.sg_land_parcel_bus_stop_distance",
# )
# dbconn.copy_from_df(
#     sch_distance,
#     "data_science.sg_gls_land_parcel_school_distance",
# )

# recalculate distance to cbd (1.2884113726733633, 103.85252198698596)
# ctr_coord = (1.2884113726733633, 103.85252198698596)
# gls['dist_to_cbd'] = gls.coordinates.apply(lambda x: geodesic(x, ctr_coord).km if pd.notna(list(x)).all() else -1)
check = 42

# to calculate pairwise distance between poi (for land parcels, can also calculate the time difference in launch)
# input: poi table a, poi table b (both with cols [id, latitude, longitude])
# output: df[id_a, id_b, distance]


import pandas as pd
import numpy as np
import SQL_connect
from geopy.distance import geodesic
from tqdm  import tqdm
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


def extract_num(string: str, type: str = 'all', decimal: bool = False, ignore_sep: str = None, keep: str = None):
    # 'type' means all numbers or just num between whitespaces by specifying type='between_spaces'
    # 'ignore_sep' can be 'any' to ignore all sep, or specify a sep like ',', then func won't treat ',' as a separator
    # 'keep' allows the func to keep all matched numbers or selected ones
    import re
    import itertools

    # if the input is already int or float, return itself: input=1234 -> output=1234
    if isinstance(string, int) or isinstance(string, float):
        num = string
        return num

    else:
        string = str(string)
        # # remove all spaces from string
        # string = ''.join(string.split(' '))
        try:
            # if the string can directly be converted to number, do so (e.g. input='1234' -> output=1234.0)
            num = float(string)
            return num

        except:
            pattern = r"\d+"  # find all numbers, any digits (ignore decimal number: input='$12.3' -> output=['12','3']
            if decimal:
                pattern = r"\d*\.?\d+"  # also match decimal numbers: input='$12.3' -> output='12.3'
            if type == 'between_spaces':
                pattern = r"\b" + pattern + r"\b"
                # match numbers in between white spaces
                # input='is $10.5 per box' -> output=None; input='is 10.5 dollars per box' -> output='10.5'
            num_list = re.findall(pattern, string)

            if ignore_sep:
                if ignore_sep == 'any':  # ignore any separator between numbers
                    # input='123a456,789.654' -> output='123456789654'
                    if len(num_list) >= 1:
                        num = "".join(num_list)
                        return float(num)
                    else:
                        return np.nan
                else:
                    # ignore specified separator
                    # input='$1,234,567.05' -> output ignore ',' & decimal is T='1234567.05'
                    # output without ignoring & decimal is T=['1','234','567.05']
                    string = string.replace(ignore_sep, "")
                    num_list = re.findall(pattern, string)
            num_list = [float(num) for num in num_list]  # convert all matched str item to float, stored in list

            if keep:  # to specify certain numbers to keep by index, e.g. num_list=[5, 6, 7], keep=1 -> output=[5]
                strip = [i.split(",") for i in keep.split("-")]
                # for now only support ",", for "-" will amend this later
                keep_idx = list(set([int(i) for i in list(itertools.chain.from_iterable(strip))]))
                if len(num_list) > len(keep_idx):  # if not keeping all items, raise a msg to notify
                    print(f"{len(num_list)} numbers detected")
                num_list = [num_list[i - 1] for i in keep_idx if 0 <= i - 1 < len(num_list)]

                if len(num_list) > 0:
                    return num_list[0] if len(num_list) == 1 else num_list
                else:
                    return np.nan

            if len(num_list) == 1:
                return num_list[0]  # if the result num_list has only 1 value, output the value as float
            elif len(num_list) > 1:
                return num_list  # otherwise output the whole num_list
            else:
                return np.nan


# read in data
poi_df = dbconn.read_data("""select poi_name , poi_type , poi_subtype , poi_lat , poi_long , location_poi_dwid  , town
                             from masterdata_sg.poi
                             ;""")
gls = dbconn.read_data("""select * from data_science.sg_new_full_land_bidding_filled_features;""")
pred = dbconn.read_data("""select * from data_science.sg_land_bidding_filled_features_with_comparable_price""")
project = dbconn.read_data("""  select project_dwid, project_name, project_type_code, completion_year, location_marker
                                from masterdata_sg.project;""")

# transform for infra
poi_df = poi_df.rename(columns={'poi_lat': 'latitude', 'poi_long': 'longitude'})
poi_mrt = poi_df[poi_df.poi_subtype == 'mrt station'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_bus = poi_df[poi_df.poi_subtype == 'bus stop'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_sch = poi_df[poi_df.poi_type == 'school'].drop_duplicates(subset='poi_name').reset_index(drop=True)

# transform for land parcels
cols = ['land_parcel_id',
        'land_parcel_std',
        'latitude',
        'longitude',
        'year_launch',
        'month_launch',
        'day_launch']
poi_land_parcel = gls[cols].drop_duplicates(subset='land_parcel_id').reset_index(drop=True)
pred_parcels_poi = pred[cols]

# transform for project
project['latitude'] = project.location_marker.apply(extract_num, decimal=True).apply(lambda x: -999 if np.isnan(x).any() else x[1])
project['longitude'] = project.location_marker.apply(extract_num, decimal=True).apply(lambda x: -999 if np.isnan(x).any() else x[0])
project_poi = project.drop(project[(project.longitude.abs() > 180) | (project.latitude.abs() > 90)].index, axis=0)

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

# create distance to cbd
cbd_coord = (1.2884113726733633, 103.85252198698596)
land_parcel_merged = pd.concat([poi_land_parcel.reset_index(drop=True),
                                pred_parcels_poi.reset_index(drop=True)],
                               ignore_index=True)[['land_parcel_id', 'latitude', 'longitude']]
land_parcel_merged['coord'] = list(zip(list(land_parcel_merged.latitude), list(land_parcel_merged.longitude)))
land_parcel_merged['dist_to_cbd'] = land_parcel_merged.coord\
    .apply(lambda x: geodesic(x, cbd_coord).m if pd.notna(list(x)).all() else -1)

# merge into infras table
land_to_infra = land_to_infra.merge(land_parcel_merged[['land_parcel_id', 'dist_to_cbd']],
                                    how='left',
                                    on='land_parcel_id')

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

check = 42

# land parcel - project distances
poi_land_parcel = poi_land_parcel[['land_parcel_id', 'latitude', 'longitude', 'year_launch']]
proj_distances = calculate_distance(poi_land_parcel, 'land_parcel_id', project_poi, 'project_dwid')
dbconn.copy_from_df(proj_distances, "data_science.sg_land_parcel_distance_to_project")

proj_dist_info = dbconn.read_data('''   with tb1 as(
                                        with tb as(
                                        select a.* , b.devt_class as land_use_big, b.devt_type as land_use , b.year_launch as land_launch_year
                                        from data_science.sg_land_parcel_distance_to_project as a
                                            left join data_science.sg_land_bidding_filled_features_with_comparable_price as b
                                            using (land_parcel_id)
                                        )
                                        select tb.*, c.project_type_code , c.completion_year as proj_completion_year
                                        from tb
                                            left join masterdata_sg.project as c
                                            using (project_dwid)
                                        )
                                        select *, 
                                        case 
                                            when project_type_code in (
                                            'ec', 
                                            'condo-w-house', 
                                            'detach-house', 
                                            'apt-w-house', 
                                            'hdb', 
                                            'condo', 
                                            'landed-housing-group', 
                                            'semid-house', 
                                            'cluster-house', 
                                            'apt', 
                                            'terrace-house')
                                            then 'residential'
                                            when project_type_code in ('commercial') then 'commercial'
                                            when project_type_code ilike '%mixed%' then 'mixed'
                                            else 'others'
                                        end as proj_devt_class
                                        from tb1''')

dbconn.copy_from_df(proj_dist_info, 'data_science.sg_land_parcel_filled_info_distance_to_project')





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

check = 42


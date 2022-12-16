# to calculate pairwise distance between poi
# input: poi table a, poi table b (both with cols [id, latitude, longitude])
# output: df[id_a, id_b, distance]


import pandas as pd
import numpy as np
import SQL_connect
from geopy.distance import geodesic
from tqdm import tqdm
import hashlib
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
            if 0 <= distance <= distance_limit:
                output.loc[len(output)] = [id_a, id_b, distance]

    return output


poi_df = dbconn.read_data("""select poi_name , poi_type , poi_subtype , poi_lat , poi_long , location_poi_dwid  , town 
                             from masterdata_sg.poi
                             ;""")
# gls = dbconn.read_data("""select * from data_science.sg_new_full_land_bidding_filled_features;""")
gls = dbconn.read_data("""select * from data_science.sg_gls_land_parcel_for_prediction""")
poi_df = poi_df.rename(columns={'poi_lat': 'latitude', 'poi_long': 'longitude'})
poi_mrt = poi_df[poi_df.poi_subtype == 'mrt station'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_bus = poi_df[poi_df.poi_subtype == 'bus stop'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_sch = poi_df[poi_df.poi_type == 'school'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_land_parcel = gls[['land_parcel_id',
                       'land_parcel_std',
                       'latitude',
                       'longitude']].drop_duplicates(subset='land_parcel_std').reset_index(drop=True)


# execute function
mrt_distance = calculate_distance(poi_land_parcel, 'land_parcel_std', poi_mrt, 'poi_name', distance_limit=5000)
bus_distance = calculate_distance(poi_land_parcel, 'land_parcel_std', poi_bus, 'poi_name', distance_limit=5000)
sch_distance = calculate_distance(poi_land_parcel, 'land_parcel_std', poi_sch, 'poi_name', distance_limit=5000)
check = 42

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


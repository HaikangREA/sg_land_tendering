import pandas as pd
import numpy as np
import SQL_connect
from geopy.distance import geodesic
from glspred import preprocess, distcal


dbconn = SQL_connect.DBConnectionRS()

# read in predicting data and master table
pred_raw = pd.read_csv(r'G:\REA\Working files\land-bidding\pipeline\Parcel for land bidding prediction.csv')
master = pd.read_csv(r'G:\REA\Working files\land-bidding\pipeline\gls_origin.csv')
master['predicting'] = 0
pred_raw['predicting'] = 1
pred = pred_raw.copy()
num_pred_parcels = pred.shape[0]

# preprocessing: align header and concat data
processor = preprocess.Preprocess()
pred_processed = processor.process(pred, master)
master_new = pd.concat([master, pred_processed], ignore_index=True)


# calculate pairwise distances
# infrastructure
# read in poi table
distance_limit = 5000
poi_df = dbconn.read_data("""select poi_name, poi_type, poi_subtype, poi_lat, poi_long, location_poi_dwid, town
                             from masterdata_sg.poi;""").rename(columns={'poi_lat': 'latitude', 'poi_long': 'longitude'})
poi_mrt = poi_df[poi_df.poi_subtype == 'mrt station'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_bus = poi_df[poi_df.poi_subtype == 'bus stop'].drop_duplicates(subset='poi_name').reset_index(drop=True)
poi_sch = poi_df[poi_df.poi_type == 'school'].drop_duplicates(subset='poi_name').reset_index(drop=True)

# transform for land parcels
master_new['coordinates'] = list(zip(list(master_new.latitude), list(master_new.longitude)))
master_new['year_launch'] = master_new.date_launch.apply(lambda date: date.split('/')[-1])
master_new['month_launch'] = master_new.date_launch.apply(lambda date: date.split('/')[1])
master_new['day_launch'] = master_new.date_launch.apply(lambda date: date.split('/')[0])

cols = ['land_parcel_id',
        'land_parcel_name',
        'coordinates',
        'year_launch',
        'month_launch',
        'day_launch']
land_parcel_poi = master_new[cols].drop_duplicates(subset='land_parcel_id').reset_index(drop=True)
pred = master_new[master_new.predicting == 1]
pred_parcels_poi = pred[cols]

distcal = distcal.PairwiseDistCalculator(poi_master=pred_parcels_poi, master_id='land_parcel_id')
# execute function for infrastructure
mrt_distance = distcal.calculate_distance(poi_ref=poi_mrt, ref_id='poi_name', distance_limit=distance_limit)\
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'mrt_station'})
# bus_distance = distcal.calculate_distance(poi_ref=poi_bus, ref_id='poi_name', distance_limit=distance_limit)\
#     .rename(columns={'poi_a': 'land_parcel_id',
#                      'poi_b': 'bus_stop'})
sch_distance = distcal.calculate_distance(poi_ref=poi_sch, ref_id='poi_name', distance_limit=distance_limit)\
    .rename(columns={'poi_a': 'land_parcel_id',
                     'poi_b': 'school'})

# find the nearest poi to parcels
nearby_mrt = distcal.find_nearby(mrt_distance, 'dist_to_mrt')
# nearby_bus = distcal.find_nearby(bus_distance, 'dist_to_bus_stop')
nearby_sch = distcal.find_nearby(sch_distance, 'dist_to_school')

# calculate distance to cbd
sg_cbd_coord = (1.2884113726733633, 103.85252198698596)
dist_to_cbd = pd.DataFrame({'dist_to_cbd': pred_parcels_poi.coordinates.apply(lambda x: round(geodesic(x, sg_cbd_coord).m), 2)})

# merge into distance-to-infrastructure table
dist_to_infra = pd.concat([nearby_mrt,
                           # nearby_bus,
                           nearby_sch,
                           dist_to_cbd], axis=1).rename_axis('land_parcel_id').reset_index()


# calculate parcel-parcel distances and project-parcel distances



# merge all tables and upload

check = 42
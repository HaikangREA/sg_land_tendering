# This script is to find nearby land parcels of each certain parcel, by calculating the distance using lat and lon
# input: gls table with cols [parcel id, latitude, longitude, launch year/month/day]
# output: land parcels nearby in df, pair-wise distance & time diff table


import pandas as pd
import numpy as np
import SQL_connect
from geopy.distance import geodesic
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, TypeVar
import time
from tqdm import tqdm


class LandParcel:
    LandParcel = TypeVar("LandParcel")

    def __init__(self,
                 name='unknown',
                 gls_id='unknown',
                 land_parcel_id='unknown',
                 lat: float = 999,
                 lon: float = 999,
                 year: int = 0,
                 month: int = 0,
                 day: int = 0,
                 distance=-1,
                 launch_time_diff=np.nan,
                 nearby_by_distance=[],
                 nearby_before_launch=[],
                 nearby_at_launch=[]
                 ):
        self.name = name
        self.gls_id = gls_id
        self.land_parcel_id = land_parcel_id
        self.lat = lat
        self.lon = lon
        self.coord = (self.lat, self.lon)
        self.year = year
        self.month = month
        self.day = day
        self.launch_time = int(str(year) + str(int(month)).zfill(2) + str(int(day)).zfill(2))
        self.launch_time_diff = launch_time_diff
        self.distance_m = distance
        self.nearby = nearby_by_distance
        self.nearby_before_launch = nearby_before_launch
        self.nearby_at_launch = nearby_at_launch

    def distance_from(self: LandParcel, compare: LandParcel) -> float:
        try:
            return round(geodesic(self.coord, compare.coord).m)
        except ValueError:
            return -1

    def days_from(self, compare: LandParcel):  # return value in days
        try:
            time_diff = date(self.year, self.month, self.day) - date(compare.year, compare.month, compare.day)
            return time_diff.days
        except:
            return np.nan

    def date_back(self, period: int, measure='days') -> int:
        date_obj = date(self.year, self.month, self.day)
        date_back = date_obj + relativedelta(days=-1 * period)
        if measure == 'months':
            date_back = date_obj + relativedelta(months=-1 * period)
        elif measure == 'years':
            date_back = date_obj + relativedelta(years=-1 * period)

        return eval(''.join(str(date_back).split('-')))


def find_nearby(gls_raw, loop_col):
    gls = gls_raw.drop_duplicates(subset=loop_col)
    gls_list = gls[loop_col]
    gls.index = gls_list
    parcelA_w_nearby = []

    for id in tqdm(gls_list, desc='Main'):
        parcel_info = gls.loc[id]
        parcelA = LandParcel(name=parcel_info.land_parcel_std,
                             gls_id=id,
                             land_parcel_id=parcel_info.land_parcel_id,
                             lat=parcel_info.latitude,
                             lon=parcel_info.longitude,
                             year=parcel_info.year_launch,
                             month=parcel_info.month_launch,
                             day=parcel_info.day_launch)
        # land parcels nearby geographically
        nearby = []

        for id_x in [i for i in gls_list if i != id]:
            parcel_info_x = gls.loc[id_x]
            parcelB = LandParcel(name=parcel_info_x.land_parcel_std,
                                 gls_id=id_x,
                                 land_parcel_id=parcel_info_x.land_parcel_id,
                                 lat=parcel_info_x.latitude,
                                 lon=parcel_info_x.longitude,
                                 year=parcel_info_x.year_launch,
                                 month=parcel_info_x.month_launch,
                                 day=parcel_info_x.day_launch)
            parcelB.distance_m = parcelA.distance_from(parcelB)
            parcelB.launch_time_diff = parcelA.days_from(parcelB)
            if 0 <= parcelB.distance_m <= distance_limit:
                nearby.append(parcelB)

        # land parcels nearby before launch of parcel A
        nearby_before_launch = [parcel for parcel in nearby if parcel.launch_time <= parcelA.launch_time]

        # land parcels nearby by dimensions of time and space (within 6 months)
        time_index = parcelA.date_back(time_limit, 'months')
        nearby_at_launch = [parcel for parcel in nearby_before_launch if parcel.launch_time >= time_index]

        # substitute attributes of parcel A
        parcelA.nearby = nearby
        parcelA.nearby_before_launch = nearby_before_launch
        parcelA.nearby_at_launch = nearby_at_launch

        parcelA_w_nearby.append(parcelA)

    return parcelA_w_nearby


start = time.time()
dbconn = SQL_connect.DBConnectionRS()
gls_raw = dbconn.read_data('''select * from data_science.sg_new_full_land_bidding_filled_features;''')
distance_limit = 5000
time_limit = 6  # months

# initialize output dfs
output = pd.DataFrame(columns=['sg_gls_id',
                               'land_parcel_id',
                               'launch_date_index',
                               'num_nearby',
                               'num_nearby_bf_launch',
                               'num_nearby_at_launch'])

pw_output = pd.DataFrame(columns=['land_parcel_id_a',
                                  'land_parcel_id_b',
                                  'distance_m',
                                  'launch_time_diff_days',
                                  'A_date',
                                  'B_date'])

# execute function
parcelA_w_nearby = find_nearby(gls_raw, 'sg_gls_id')

# append output df
for parcelA in tqdm(parcelA_w_nearby, desc='Output'):
    output.loc[len(output)] = [parcelA.gls_id,
                               parcelA.land_parcel_id,
                               parcelA.launch_time,
                               len(parcelA.nearby),
                               len(parcelA.nearby_before_launch),
                               len(parcelA.nearby_at_launch),
                               ]
    for parcelB in parcelA.nearby:
        pw_output.loc[len(pw_output)] = [parcelA.land_parcel_id,
                                         parcelB.land_parcel_id,
                                         parcelB.distance_m,
                                         parcelB.launch_time_diff,
                                         parcelA.launch_time,
                                         parcelB.launch_time
                                         ]

check = 42
# dbconn.copy_from_df(
#     output,
#     "data_science.sg_gls_nearby_land_parcels",
# )

dbconn.copy_from_df(
    pw_output,
    "data_science.sg_gls_pairwise_nearby_land_parcels",
)

print('{:.2f}s'.format(time.time()-start))

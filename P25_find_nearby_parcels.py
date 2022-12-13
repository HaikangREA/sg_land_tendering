# This script is to find nearby land parcels of each certain parcel, by calculating the distance using lat and lon

import pandas as pd
import numpy as np
import SQL_connect
from geopy.distance import geodesic
from typing import List, Tuple, TypeVar


class LandParcel:
    LandParcel = TypeVar("LandParcel")

    def __init__(self,
                 name='unknown',
                 gls_id='',
                 land_parcel_id='',
                 lat: float = None,
                 lon: float = None,
                 year: int = 0,
                 month: int = 0,
                 day: int = 0,
                 distance=-1,
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
        self.distance_m = distance
        self.nearby = nearby_by_distance
        self.nearby_before_launch = nearby_before_launch
        self.nearby_at_launch = nearby_at_launch

    def distance_from(self: LandParcel, destination: LandParcel) -> float:
        try:
            return round(geodesic(self.coord, destination.coord).m)
        except ValueError:
            return -1

    def date_back(self, period: int, measure='days') -> int:
        from datetime import date
        from dateutil.relativedelta import relativedelta
        date_obj = date(self.year, self.month, self.day)
        date_back = date_obj + relativedelta(days=-1 * period)
        if measure == 'months':
            date_back = date_obj + relativedelta(months=-1 * period)
        elif measure == 'years':
            date_back = date_obj + relativedelta(years=-1 * period)

        return eval(''.join(str(date_back).split('-')))


dbconn = SQL_connect.DBConnectionRS()
gls_raw = dbconn.read_data('''select * from data_science.sg_new_full_land_bidding_filled_features;''')
distance_limit = 4000
time_limit = 6  # months
output = pd.DataFrame(columns=['sg_gls_id',
                               'land_parcel_id',
                               'launch_date_index',
                               'num_nearby',
                               'num_nearby_bf_launch',
                               'num_nearby_at_launch'])
# parcel1 = LandParcel(lat=1.27, lon=103.84)
# parcel2 = LandParcel(lat=1.28, lon=103.85)
# print(parcel1.distance_from(parcel2))
gls = gls_raw.drop_duplicates(subset='sg_gls_id')
gls_list = gls.sg_gls_id
gls.index = gls.sg_gls_id

for id in gls_list:
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
                             land_parcel_id=parcel_info_x.land_parcel_std,
                             lat=parcel_info_x.latitude,
                             lon=parcel_info_x.longitude,
                             year=parcel_info_x.year_launch,
                             month=parcel_info_x.month_launch,
                             day=parcel_info_x.day_launch)
        parcelB.distance_m = parcelA.distance_from(parcelB)
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

    # append output df
    output.loc[len(output)] = [parcelA.gls_id,
                               parcelA.land_parcel_id,
                               parcelA.launch_time,
                               len(parcelA.nearby),
                               len(parcelA.nearby_before_launch),
                               len(parcelA.nearby_at_launch)]

check = 42
dbconn.copy_from_df(
    output,
    "data_science.sg_gls_nearby_land_parcels",
)



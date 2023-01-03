import pandas as pd
import numpy as np
import SQL_connect
from glspred.distcal import PairwiseDistCalculator


class Preprocess:

    def __init__(self, dbconn=None, pred: pd.DataFrame = None, master: pd.DataFrame = None):
        self.dbconn = dbconn
        if pred is not None:
            self.pred_origin = pred.copy()  # store original pred data
            self.pred = pred
        if master is not None:
            self.master = master
        self.res_keywords = ['residential',
                             'condo',
                             'hdb',
                             'apartment',
                             'apt',
                             'executive condo',
                             'ec',
                             'flat',
                             'landed',
                             'terrace',
                             'bungalow',
                             'condo-w-house',
                             'detach-house',
                             'apt-w-house',
                             'landed-housing-group',
                             'semid-house',
                             'cluster-house',
                             'terrace-house'
                             ]
        self.com_keywords = ['commercial',
                             'recreation']
        self.res_kw_condo_hdb = ['CO', 'AP', 'LP', 'FT', 'BH', 'TH']

    @staticmethod
    def get_uuid(text_id: str, mode='hashlib'):
        # create 64-digit uuid
        import hashlib
        if mode == 'hashlib':
            return hashlib.sha256(text_id.encode('utf-8')).hexdigest()
        else:
            txt_cleaned = text_id.lower().replace(' ', '')
            return ''.join([s for s in txt_cleaned if s.isalnum()]).ljust(64, '0')

    # to create uuid for land parcel and gls sale record
    def encode(self, pred: pd.DataFrame = None) -> pd.DataFrame:
        # if param passed in, overwrite self
        if pred is not None:
            self.pred = pred

        pred = self.pred
        input_cols = pred.columns  # store original input columns
        # define dimensions used to create uuid
        land_parcel_dimensions = ['land_parcel_name',
                                  'latitude',
                                  'longitude']
        gls_dimensions = ['land_parcel_id',
                          'award_month_index',
                          'site_area_sqm']

        # if key dimensions not exist, create with values of NA
        for col in land_parcel_dimensions + gls_dimensions:
            if col not in pred.columns:
                pred[col] = np.nan

        # create land parcel uuid
        # in case of lacking any dimension, create uuid as <name><00000...> (total 64-digit), same for gls uuid
        # pred['land_parcel_id'] = pred.land_parcel_name.apply(lambda x: x.lower().replace(' ', ''))\
        #     .apply(lambda x: ''.join([s for s in x if s.isalnum()]).ljust(64, '0'))
        pred['land_parcel_id'] = pred.land_parcel_name.apply(self.get_uuid, mode='default')
        land_parcel_non_na_index = pred[(pred.land_parcel_name.notna())
                                        & (pred.latitude.notna())
                                        & (pred.longitude.notna())].index
        land_parcel_id_text = pred.land_parcel_name + pred.latitude.astype(str) + pred.longitude.astype(str)
        pred.loc[land_parcel_non_na_index, 'land_parcel_id'] = land_parcel_id_text.loc[land_parcel_non_na_index]\
                                                                                .apply(self.get_uuid, mode='hashlib')

        # create gls uuid
        pred['sg_gls_id'] = pred.land_parcel_name.apply(self.get_uuid, mode='default')
        gls_non_na_index = pred[(~pred.land_parcel_id.str.contains('0'*5))
                                & (pred.award_month_index.notna())
                                & (pred.site_area_sqm.notna())].index
        gls_id_text = pred.land_parcel_id + pred.award_month_index.astype(str)
        pred.loc[gls_non_na_index, 'sg_gls_id'] = gls_id_text.loc[gls_non_na_index].apply(self.get_uuid, mode='hashlib')

        # add id cols to output cols
        output_cols = ['sg_gls_id', 'land_parcel_id'] + list(input_cols)
        # update self.pred
        self.pred = pred[output_cols]

        return self.pred

    # to balance metrics of gfa, gpr and site area (just in case)
    # only fill in values for NA; will not overwrite non-NA values
    def balance_metrics(self, pred: pd.DataFrame = None) -> pd.DataFrame:
        # if param passed in, overwrite self
        if pred is not None:
            self.pred = pred

        pred = self.pred
        # get index of NA values
        gfa_na_index = pred[pred.gfa_sqm.isna()].index
        gpr_na_index = pred[pred.gpr.isna()].index
        area_na_index = pred[pred.site_area_sqm.isna()].index

        # calculate values and fill into table
        pred.loc[gfa_na_index, 'gfa_sqm'] = pred.loc[gfa_na_index, 'site_area_sqm'] * pred.loc[gfa_na_index, 'gpr']
        pred.loc[gpr_na_index, 'gpr'] = pred.loc[gpr_na_index, 'gfa_sqm'] / pred.loc[gpr_na_index, 'site_area_sqm']
        pred.loc[area_na_index, 'site_area_sqm'] = pred.loc[area_na_index, 'gfa_sqm'] * pred.loc[area_na_index, 'gpr']

        # update
        self.pred = pred

        return self.pred

    # to align the columns between master and pred tables, by adjusting pred columns
    def align_columns(self, pred: pd.DataFrame = None, master: pd.DataFrame = None, fill_value=np.nan) -> pd.DataFrame:
        # if param passed in, overwrite self
        if pred is not None:
            self.pred = pred
        if master is not None:
            self.master = master

        pred_cols = self.pred.columns
        # create NA cols for columns in master but not in pred
        self.pred[[col for col in self.master.columns if col not in pred_cols]] = fill_value
        # update
        self.pred = self.pred[self.master.columns]

        return self.pred

    def land_use_regroup(self, land_use_type_raw: str):
        if land_use_type_raw:
            type_code_lower = land_use_type_raw.lower()
            if (any(kw in type_code_lower for kw in self.res_keywords) and any(kw in type_code_lower for kw in self.com_keywords)) \
                    or ('mix' in type_code_lower) or ('white' in type_code_lower):
                return 'mixed'
            elif any(kw in type_code_lower for kw in self.res_keywords):
                return 'residential'
            elif any(kw in type_code_lower for kw in self.com_keywords):
                return 'commercial'
            else:
                return 'others'

        else:
            return np.nan

    # to merge func into one
    def process(self, pred: pd.DataFrame = None, master: pd.DataFrame = None, fill_value=np.nan) -> pd.DataFrame:
        if pred is not None:
            self.pred = pred
            self.pred_origin = pred.copy()
        if master is not None:
            self.master = master

        self.pred = self.pred_origin.copy()
        pred_encoded = self.encode()
        pred_balanced = self.balance_metrics()
        pred_aligned = self.align_columns()
        pred_aligned.loc[:, 'land_use_type'] = pred_aligned.land_use_type.apply(self.land_use_regroup)
        self.pred = pred_aligned

        return self.pred

    def find_region_zone(self, poi_a: pd.DataFrame, ref_a, dist_lim=3000, cutoff=False):
        # cutoff means to break the loop when one location whose distance is within limit has been found
        distcal = PairwiseDistCalculator()
        location_df = self.dbconn.read_data('''select address_dwid, city_area as "zone", region_admin_district as region, latitude, longitude
                                            from masterdata_sg.address''')
        location_df.rename(columns={'address_dwid': 'poi_b'}, inplace=True)
        location_df['coordinates'] = list(zip(location_df.latitude.tolist(), location_df.longitude.tolist()))
        dist_df = distcal.calculate_distance(poi_a, location_df, ref_a, 'poi_b', dist_lim, cutoff)
        location_df.reset_index(drop=True, inplace=True)
        poi_a.reset_index(drop=True, inplace=True)
        region_zone = dist_df.merge(location_df[['poi_b', 'region', 'zone']], how='left', on='poi_b').rename(columns={'poi_a': ref_a})
        return region_zone[[ref_a, 'region', 'zone']]


if __name__ == "__main__":
    dbconn = SQL_connect.DBConnectionRS()
    gls_dup = pd.read_csv(r'G:\REA\Working files\land-bidding\pipeline\gls_dup.csv')
    prep = Preprocess(dbconn=dbconn)
    # region_zone_info = prep.find_region_zone(pred_raw, 'land_parcel_name', 500, cutoff=True)
    gls_dup_encoded = prep.encode(gls_dup)
    breakpoint()
    print(0)


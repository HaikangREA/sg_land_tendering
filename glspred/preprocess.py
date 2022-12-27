import pandas as pd
import numpy as np


class Preprocess:

    def __init__(self, pred: pd.DataFrame = None, master: pd.DataFrame = None):
        if pred is not None:
            self.pred_origin = pred.copy()  # store original pred data
            self.pred = pred
        if master is not None:
            self.master = master

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
        gls_non_na_index = pred[(pred.land_parcel_id.str.contains('0'*5))
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

        return pred_aligned

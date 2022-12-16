import pandas as pd
import numpy as np
import SQL_connect

dbconn = SQL_connect.DBConnectionRS()
gls = dbconn.read_data('''select * from data_science.sg_new_full_land_bidding_filled_features;''')
comp = dbconn.read_data('''select * from data_science.updated_sg_new_comparable_land_bidding''')

# gls = gls.drop(['comparable_price_psm_gfa', 'comparable_price_error'], axis=1)
gls = gls.drop_duplicates(subset=['land_parcel_id'], keep='last')
gls_new = pd.merge(gls, comp[['sg_gls_id' ,'comparable_price_psm_gfa']], how='left', on='sg_gls_id')
gls_new['comparable_price_error'] = abs(gls_new.comparable_price_psm_gfa / gls_new.successful_price_psm_gfa - 1)

check = 42
dbconn.copy_from_df(gls_new, "data_science.sg_land_bidding_filled_features_with_comparable_price")
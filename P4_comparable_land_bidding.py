import pandas as pd
import numpy as np

# adjust for price index
price_index = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\hi_2000_new.csv')
gls = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\ready for uploading\gls_details_spread.csv')
gls_with_index = pd.merge(gls, price_index, how='left', on='year_launch')
print(gls_with_index.hi_tender_price.isna().sum())
gls_with_index['tender_price_real'] = gls_with_index.successful_tender_price / gls_with_index.hi_tender_price
gls_with_index['tender_price_real'] = gls_with_index['tender_price_real'].apply(lambda x: '%.2f' %x)
gls_with_index['price_psm_real'] = gls_with_index.successful_price_psm_gfa / gls_with_index.hi_price_psm_gfa
gls_with_index['price_psm_real'] = gls_with_index['price_psm_real'].apply(lambda x: '%.2f' %x)
gls_price = gls_with_index[['sg_gls_id', 'year_launch', 'successful_tender_price', 'hi_tender_price', 'tender_price_real', 'successful_price_psm_gfa', 'price_psm_real']]
# gls_price.to_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_with_index.csv', index=False)
gls_with_index.to_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_with_index.csv', index=False)
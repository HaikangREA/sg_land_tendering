import pandas as pd
import numpy as np
import hashlib


def recency(gls, col_id, col_time, tgt_id, compare_id):
    parcel_tgt = gls[gls[col_id] == tgt_id]
    parcel_com = gls[gls[col_id] == compare_id]
    tgt_time = pd.to_datetime(parcel_tgt[col_time].values)
    com_time = pd.to_datetime(parcel_com[col_time].values)
    month_delta = ((tgt_time.year - com_time.year)*12 + tgt_time.month - com_time.month).tolist()[0]
    return month_delta

def region(gls, col_id, col_region, tgt_id, compare_id):
    parcel_tgt = gls[gls[col_id] == tgt_id]
    parcel_com = gls[gls[col_id] == compare_id]
    tgt_region = parcel_tgt[col_region].values
    com_region = parcel_com[col_region].values
    return tgt_region == com_region

def find_common(df, col1, col2):
    common_ele = []
    for i in range(len(df)):
        val1 = df[col1][i]
        val2 = df[col2][i]
        common_ele.append([item for item in val1 if item in val2])
    df['common_elements'] = common_ele
    return df


# adjust for price index
# price_index = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\hi_2000_new.csv')
# gls = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\ready for uploading\gls_details_spread.csv')
# gls_with_index = pd.merge(gls, price_index, how='left', on='year_launch')
# print(gls_with_index.hi_tender_price.isna().sum())
# gls_with_index['tender_price_real'] = gls_with_index.successful_tender_price / gls_with_index.hi_tender_price
# gls_with_index['tender_price_real'] = gls_with_index['tender_price_real'].apply(lambda x: '%.2f' %x)
# gls_with_index['price_psm_real'] = gls_with_index.successful_price_psm_gfa / gls_with_index.hi_price_psm_gfa
# gls_with_index['price_psm_real'] = gls_with_index['price_psm_real'].apply(lambda x: '%.2f' %x)
# gls_price = gls_with_index[['sg_gls_id', 'year_launch', 'successful_tender_price', 'hi_tender_price', 'tender_price_real', 'successful_price_psm_gfa', 'price_psm_real']]
# # gls_price.to_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_with_index.csv', index=False)
# gls_with_index.to_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_with_index.csv', index=False)
gls_with_index = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_with_index.csv')
gls_with_index.insert(loc=0, column="land_parcel_id", value=gls_with_index.land_parcel_std.apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()))
gls_with_index.sort_values(by=['year_launch', 'month_launch', 'day_launch'], inplace=True)
gls_with_index.drop_duplicates(subset=['land_parcel_id'], keep='last', inplace=True)
land_parcel_id = list(gls_with_index.land_parcel_id)[:120]

time_limit = 12
# land_parcel_id = ['e1cc8490c7c83df452efb74500c5fd2d6defdd7683882eb291cd671bfaa3a759']
recency_comparable_dict = {}
for id in land_parcel_id:
    id_list_all = list(gls_with_index.land_parcel_id)[:120]
    id_list_all.remove(id)
    comparable_list_recency = []
    comparable_list_region = []
    for id_compare in id_list_all:
        time_diff = recency(gls_with_index, 'land_parcel_id', 'date_launch', id, id_compare)
        if 0 <= time_diff < 12:
            # print(time_diff)
            comparable_list_recency.append(id_compare)
    recency_comparable_dict[id] = comparable_list_recency

# check = gls_with_index[(gls_with_index['land_parcel_id'] == '64c1e3fc025b6355b732f7e74d1e77b00710f103871bcfc46763f28852814535') | (gls_with_index['land_parcel_id'] == '764a2356084f7510c42b37f403a43be8e7f2f16ade3d641999e59daaf8c016ca')]
comparable_df = pd.DataFrame({'land_parcel_id': recency_comparable_dict.keys(),
                              'comparable_recency': recency_comparable_dict.values()})

region_comparable_dict = {}
for id in land_parcel_id:
    id_list_all = list(gls_with_index.land_parcel_id)[:120]
    id_list_all.remove(id)
    comparable_list_recency = []
    comparable_list_region = []
    for id_compare in id_list_all:
        same_region = region(gls_with_index, 'land_parcel_id', 'region', id, id_compare)
        if same_region:
            # print(same_region)
            comparable_list_region.append(id_compare)
    region_comparable_dict[id] = comparable_list_region

region_com_df = pd.DataFrame({'land_parcel_id': region_comparable_dict.keys(), 'comparable_region': region_comparable_dict.values()})
comparable_df = pd.merge(comparable_df, region_com_df, how='left', on='land_parcel_id')

comparable_df = find_common(comparable_df, 'comparable_recency', 'comparable_region')
comparable_df['num'] = comparable_df.common_elements.apply(lambda x: len(x))

# try site area
# try devt type
# try zone

#%%

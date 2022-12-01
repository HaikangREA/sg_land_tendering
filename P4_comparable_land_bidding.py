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


def site_area(gls, col_id, col_area, tgt_id, compare_id):
    parcel_tgt = gls[gls[col_id] == tgt_id]
    parcel_com = gls[gls[col_id] == compare_id]
    tgt_area = parcel_tgt[col_area].values
    com_area = parcel_com[col_area].values

    # return month_delta


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
land_parcel_id = list(gls_with_index.land_parcel_id)

time_limit = 24
# land_parcel_id = ['e1cc8490c7c83df452efb74500c5fd2d6defdd7683882eb291cd671bfaa3a759']

# recency within last 24 months
recency_comparable_dict = {}
for id in land_parcel_id:
    id_list_all = list(gls_with_index.land_parcel_id)
    id_list_all.remove(id)
    comparable_list_recency = []
    comparable_list_region = []
    for id_compare in id_list_all:
        time_diff = recency(gls_with_index, 'land_parcel_id', 'date_launch', id, id_compare)
        if 0 <= time_diff <= time_limit:
            # print(time_diff)
            comparable_list_recency.append(id_compare)
    recency_comparable_dict[id] = comparable_list_recency

# check = gls_with_index[(gls_with_index['land_parcel_id'] == '64c1e3fc025b6355b732f7e74d1e77b00710f103871bcfc46763f28852814535') | (gls_with_index['land_parcel_id'] == '764a2356084f7510c42b37f403a43be8e7f2f16ade3d641999e59daaf8c016ca')]
comparable_df = pd.DataFrame({'land_parcel_id': recency_comparable_dict.keys(),
                              'comparable_recency': recency_comparable_dict.values()})

# same region
region_comparable_dict = {}
for i in range(len(comparable_df)):
    land_parcel = comparable_df.iloc[i, 0]
    tgt_region = gls_with_index[gls_with_index.land_parcel_id == land_parcel]['region'].values[0]
    comparable_list = comparable_df.iloc[i, 1]
    comp_gls = gls_with_index[gls_with_index.land_parcel_id.isin(comparable_list)]
    region_comp = []
    for j in range(len(comp_gls)):
        gls_record = comp_gls.iloc[j, :]
        if gls_record.region == tgt_region:
            region_comp.append(gls_record.land_parcel_id)
    region_comparable_dict[land_parcel] = region_comp
region_comparable_df = pd.DataFrame({'land_parcel_id': region_comparable_dict.keys(), 'comparable_region': region_comparable_dict.values()})
comparable_df = pd.merge(comparable_df, region_comparable_df, how='left', on='land_parcel_id')

# same devt class
devt_comparable_dict = {}
for i in range(len(comparable_df)):
    land_parcel = comparable_df.iloc[i, 0]
    tgt_devt = gls_with_index[gls_with_index.land_parcel_id == land_parcel]['devt_class'].values[0]
    comparable_list = comparable_df.iloc[i, 2]
    comp_gls = gls_with_index[gls_with_index.land_parcel_id.isin(comparable_list)]
    devt_comp = []
    for j in range(len(comp_gls)):
        gls_record = comp_gls.iloc[j, :]
        if gls_record.devt_class == tgt_devt:
            devt_comp.append(gls_record.land_parcel_id)
    devt_comparable_dict[land_parcel] = devt_comp
devt_comparable_df = pd.DataFrame({'land_parcel_id': devt_comparable_dict.keys(), 'comparable_devt': devt_comparable_dict.values()})
comparable_df = pd.merge(comparable_df, devt_comparable_df, how='left', on='land_parcel_id')

# area range within 20%
area_comparable_dict = {}
for i in range(len(comparable_df)):
    land_parcel = comparable_df.iloc[i, 0]
    tgt_area = gls_with_index[gls_with_index.land_parcel_id == land_parcel]['site_area_sqm'].values[0]
    comparable_list = comparable_df.iloc[i, 3]
    comp_gls = gls_with_index[gls_with_index.land_parcel_id.isin(comparable_list)]
    area_comp = []
    for j in range(len(comp_gls)):
        gls_record = comp_gls.iloc[j, :]
        if abs((gls_record.site_area_sqm / tgt_area) - 1) < 0.2:
            area_comp.append(gls_record.land_parcel_id)
    area_comparable_dict[land_parcel] = area_comp
area_comparable_df = pd.DataFrame({'land_parcel_id': area_comparable_dict.keys(), 'comparable_area': area_comparable_dict.values()})
comparable_df = pd.merge(comparable_df, area_comparable_df, how='left', on='land_parcel_id')

comparable_df_merge = pd.merge(comparable_df, gls_with_index[['land_parcel_id', 'sg_gls_id', 'successful_tender_price', 'successful_price_psm_gfa']], how='left', on='land_parcel_id')
comparable_df_merge['comparable_parcels'] = np.nan
comparable_df_merge['method'] = 'NO COMPARABLE'
for i in range(len(comparable_df_merge)):
    if len(comparable_df_merge.comparable_area[i]) == 0:
        if len(comparable_df_merge.comparable_devt[i]) == 0:
            if len(comparable_df_merge.comparable_region[i]) == 0:
                pass
            else:
                comparable_df_merge.comparable_parcels[i] = comparable_df_merge.comparable_region[i]
                comparable_df_merge.method[i] = 'past 24m+same region'
        else:
            comparable_df_merge.comparable_parcels[i] = comparable_df_merge.comparable_devt[i]
            comparable_df_merge.method[i] = 'past 24m+same region+same dev type'
    else:
        comparable_df_merge.comparable_parcels[i] = comparable_df_merge.comparable_area[i]
        comparable_df_merge.method[i] = 'past 24m+same region+same dev type+area range 20%'

# calculate comparable price
def find_comparable_price(comp_id_list, ref_df, id_col, price_col):
    try:
        ref_records = ref_df[ref_df[id_col].isin(comp_id_list)]
        return ref_records[price_col].mean()
    except TypeError:
        return comp_id_list

comparable_df_merge['comparable_tender_price'] = comparable_df_merge.comparable_parcels.apply(find_comparable_price, ref_df=gls_with_index, id_col='land_parcel_id', price_col='successful_tender_price')
comparable_df_merge['comparable_price_psm_gfa'] = comparable_df_merge.comparable_parcels.apply(find_comparable_price, ref_df=gls_with_index, id_col='land_parcel_id', price_col='successful_price_psm_gfa')
comparable_df_merge['num_comparable_parcels'] = comparable_df_merge.comparable_parcels.apply(lambda x: len(x) if isinstance(x, list) else 0)
comparable_df_merge['total_price_error'] = abs(comparable_df_merge.comparable_tender_price / comparable_df_merge.successful_tender_price - 1)
comparable_df_merge['price_psm_gfa_error'] = abs(comparable_df_merge.comparable_price_psm_gfa / comparable_df_merge.successful_price_psm_gfa - 1)

# comparable_df_merge[['land_parcel_id', 'method', 'successful_tender_price',
#                      'comparable_tender_price', 'successful_price_psm_gfa',
#                      'comparable_price_psm_gfa', 'total_price_error', 'price_psm_gfa_error',
#                      'num_comparable_parcels', 'comparable_parcels']] \
#     .to_csv(
#     r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\comparable_land_bidding.csv',
#     index = False)
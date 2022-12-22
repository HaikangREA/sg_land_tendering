import pandas as pd
import numpy as np
import hashlib
import SQL_connect
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from geopy.distance import geodesic

dbconn = SQL_connect.DBConnectionRS()


def recency(gls, col_id, col_time, tgt_id, compare_id):
    parcel_tgt = gls[gls[col_id] == tgt_id]
    parcel_com = gls[gls[col_id] == compare_id]
    tgt_time = pd.to_datetime(parcel_tgt[col_time].values)
    com_time = pd.to_datetime(parcel_com[col_time].values)
    month_delta = ((tgt_time.year - com_time.year) * 12 + tgt_time.month - com_time.month).tolist()[0]
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


def get_month_index_from(month, gap):
    d = datetime.datetime.strptime(str(int(month)), "%Y%m") + relativedelta(months=gap)
    year = d.year
    month = d.month
    return str(year) + str(month).zfill(2)


# calculate comparable price
def find_comparable_price(comparable_df, dat, index_table, price_col):
    try:
        comparable_df = comparable_df.merge(
            index_table[['year_launch', 'hi_price_psm_gfa']], how="left", on="year_launch",
        )
        comparable_df['base_hi'] = \
        index_table[index_table.year_launch == dat.year_launch.values[0]].hi_price_psm_gfa.values[0]
        comparable_df['hi_price_psf'] = comparable_df[
                                            price_col] / comparable_df.hi_price_psm_gfa * comparable_df.base_hi
        return comparable_df[price_col].mean()
    except TypeError:
        return None


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

gls_with_index = dbconn.read_data("""   select * from data_science.sg_land_bidding_filled_features_with_comparable_price """)
# gls_with_index = dbconn.read_data("""   select *
#                                         from data_science.sg_new_full_land_bidding_filled_features
#                                         left join data_science.sg_land_bidding_psm_price_hedonic_index_2022
#                                         using (year_launch)
#                                         left join data_science.sg_land_bidding_total_price_hedonic_index_2022
#                                         using (year_launch);
#                                         """)
# pred = dbconn.read_data('''select * from data_science.sg_gls_land_parcel_for_prediction''')
land_bid_index = dbconn.read_data(""" select * from data_science.sg_land_bidding_psm_price_hedonic_index_2022 ;""")

# pred = gls_with_index[gls_with_index.sg_gls_id.astype(str).str[-10:] == '0'*10]
#
# create coordinates
gls_with_index['coordinates'] = list(zip(list(gls_with_index.latitude), list(gls_with_index.longitude)))
#
# # create na values for col in gls but pred doesnt have
# for col in [col for col in gls_with_index.columns if col not in pred.columns]:
#     pred[col] = np.nan
#
# # append gls with pred
# pred = pred[gls_with_index.columns]
# gls_with_index = pd.concat([gls_with_index, pred])

# # change dev type with resi & comm to rc
# rc_idx = gls_with_index[gls_with_index.devt_type.str.contains(r'(?=.*[Rr]esidential)(?=.*[Cc]ommercial)')].index
# gls_with_index.loc[rc_idx, 'devt_class'] = 'rc'
# gls_with_index.loc[0, 'devt_class'] = 'residential'

gls_with_index.sort_values(by=['year_launch', 'month_launch', 'day_launch'], inplace=True)
gls_with_index.drop_duplicates(subset=['land_parcel_id'], keep='last', inplace=True)
sg_gls_id = list(gls_with_index.sg_gls_id)
gls_with_index.index = gls_with_index.sg_gls_id

time_limit = 24
distance_limit_km = 5
area_limit = 0.2
# land_parcel_id = ['e1cc8490c7c83df452efb74500c5fd2d6defdd7683882eb291cd671bfaa3a759']
# gls_with_index.drop(['hi_price_psm_gfa'], axis=1, inplace=True)

# main code
comparable_final = []
for id in tqdm(sg_gls_id[:-1]):
    id_list_all = list(gls_with_index.sg_gls_id)
    id_list_all.remove(id)
    dat = gls_with_index[gls_with_index['sg_gls_id'] == id]

    # recency within last 24 months
    comparable_df = gls_with_index[
        (gls_with_index.launch_month_index.astype(str) > get_month_index_from(dat.launch_month_index, -24))
        & ((dat.launch_month_index.values - gls_with_index.launch_month_index) > 0)
        ]
    if comparable_df.shape[0] <= 0:
        comparable_final.append([id, gls_with_index.loc[id].land_parcel_id, None, "No comparable", 0])
        continue

    # same dev class
    comparable_df = comparable_df[comparable_df['devt_class'] == dat.devt_class.values[0]]
    if comparable_df.shape[0] <= 0:
        comparable_final.append([id, gls_with_index.loc[id].land_parcel_id,  None, "No comparable", 0])
        continue

    # Same region
    comparable_df = comparable_df[comparable_df['region'] == dat.region.values[0]]
    if comparable_df.shape[0] <= 0:
        comparable_final.append([id, gls_with_index.loc[id].land_parcel_id,  None, "No comparable", 0])
        continue

    # Same zone
    comparable_df = comparable_df[comparable_df['zone'] == dat.zone.values[0]]
    if comparable_df.shape[0] <= 0:
        comparable_final.append([id, gls_with_index.loc[id].land_parcel_id, None, "No comparable", 0])
        continue

    # within certain distance
    coord_dat = dat.reset_index(drop=True).loc[0, 'coordinates']
    dist_limit_bool = comparable_df.coordinates.apply(lambda x: geodesic(x, coord_dat).km < distance_limit_km if not np.isnan(x+coord_dat).any() else False)
    comparable_df = comparable_df[dist_limit_bool]
    if comparable_df.shape[0] <= 0:
        comparable_final.append([id, gls_with_index.loc[id].land_parcel_id,  None, "No comparable", 0])
        continue

    # Area <= 20%
    comparable_df_area = comparable_df[abs((comparable_df.site_area_sqm / dat.site_area_sqm.values[0]) - 1) < area_limit]
    if comparable_df_area.shape[0] > 0:
        est = find_comparable_price(comparable_df_area, dat, land_bid_index, price_col='price_psm_gfa_1st')
        comparable_final.append(
            [id, gls_with_index.loc[id].land_parcel_id, est, f"past {time_limit}m, same dev, same region, same zone, wihtin {distance_limit_km}km, area<{area_limit*100}%", comparable_df_area.shape[0]])
    else:
        est = find_comparable_price(comparable_df, dat, land_bid_index, price_col='price_psm_gfa_1st')
        comparable_final.append([id, gls_with_index.loc[id].land_parcel_id, est, f"past {time_limit}m, same dev, same region, same zone, wihtin {distance_limit_km}km", comparable_df.shape[0]])

final_df = pd.DataFrame(comparable_final,
                        columns=['sg_gls_id', 'land_parcel_id', 'comparable_price_psm_gfa', 'method', 'num_comparable_parcels'])

# final_df = final_df.merge(gls_with_index[['land_parcel_id', 'devt_class', 'devt_type', 'successful_price_psm_gfa']],
#                           how='left',
#                           on='land_parcel_id')
# dbconn.copy_from_df(final_df, "data_science.updated_sg_new_comparable_land_bidding")
check = 42

# # same region
# region_comparable_dict = {}
# for i in range(len(comparable_df)):
#     land_parcel = comparable_df.iloc[i, 0]
#     tgt_region = gls_with_index[gls_with_index.land_parcel_id == land_parcel]['region'].values[0]
#     comparable_list = comparable_df.iloc[i, 1]
#     comp_gls = gls_with_index[gls_with_index.land_parcel_id.isin(comparable_list)]
#     region_comp = []
#     for j in range(len(comp_gls)):
#         gls_record = comp_gls.iloc[j, :]
#         if gls_record.region == tgt_region:
#             region_comp.append(gls_record.land_parcel_id)
#     region_comparable_dict[land_parcel] = region_comp
# region_comparable_df = pd.DataFrame({'land_parcel_id': region_comparable_dict.keys(), 'comparable_region': region_comparable_dict.values()})
# comparable_df = pd.merge(comparable_df, region_comparable_df, how='left', on='land_parcel_id')
#
# # same devt class
# devt_comparable_dict = {}
# for i in range(len(comparable_df)):
#     land_parcel = comparable_df.iloc[i, 0]
#     tgt_devt = gls_with_index[gls_with_index.land_parcel_id == land_parcel]['devt_class'].values[0]
#     comparable_list = comparable_df.iloc[i, 2]
#     comp_gls = gls_with_index[gls_with_index.land_parcel_id.isin(comparable_list)]
#     devt_comp = []
#     for j in range(len(comp_gls)):
#         gls_record = comp_gls.iloc[j, :]
#         if gls_record.devt_class == tgt_devt:
#             devt_comp.append(gls_record.land_parcel_id)
#     devt_comparable_dict[land_parcel] = devt_comp
# devt_comparable_df = pd.DataFrame({'land_parcel_id': devt_comparable_dict.keys(), 'comparable_devt': devt_comparable_dict.values()})
# comparable_df = pd.merge(comparable_df, devt_comparable_df, how='left', on='land_parcel_id')
#
# # area range within 20%
# area_comparable_dict = {}
# for i in range(len(comparable_df)):
#     land_parcel = comparable_df.iloc[i, 0]
#     tgt_area = gls_with_index[gls_with_index.land_parcel_id == land_parcel]['site_area_sqm'].values[0]
#     comparable_list = comparable_df.iloc[i, 3]
#     comp_gls = gls_with_index[gls_with_index.land_parcel_id.isin(comparable_list)]
#     area_comp = []
#     for j in range(len(comp_gls)):
#         gls_record = comp_gls.iloc[j, :]
#         if abs((gls_record.site_area_sqm / tgt_area) - 1) < 0.2:
#             area_comp.append(gls_record.land_parcel_id)
#     area_comparable_dict[land_parcel] = area_comp
# area_comparable_df = pd.DataFrame({'land_parcel_id': area_comparable_dict.keys(), 'comparable_area': area_comparable_dict.values()})
# comparable_df = pd.merge(comparable_df, area_comparable_df, how='left', on='land_parcel_id')
#
# comparable_df_merge = pd.merge(comparable_df, gls_with_index[['land_parcel_id', 'sg_gls_id', 'successful_tender_price', 'successful_price_psm_gfa']], how='left', on='land_parcel_id')
# comparable_df_merge['comparable_parcels'] = np.nan
# comparable_df_merge['method'] = 'NO COMPARABLE'
# for i in range(len(comparable_df_merge)):
#     if len(comparable_df_merge.comparable_area[i]) == 0:
#         if len(comparable_df_merge.comparable_devt[i]) == 0:
#             if len(comparable_df_merge.comparable_region[i]) == 0:
#                 pass
#             else:
#                 comparable_df_merge.comparable_parcels[i] = comparable_df_merge.comparable_region[i]
#                 comparable_df_merge.method[i] = 'past 24m+same region'
#         else:
#             comparable_df_merge.comparable_parcels[i] = comparable_df_merge.comparable_devt[i]
#             comparable_df_merge.method[i] = 'past 24m+same region+same dev type'
#     else:
#         comparable_df_merge.comparable_parcels[i] = comparable_df_merge.comparable_area[i]
#         comparable_df_merge.method[i] = 'past 24m+same region+same dev type+area range 20%'
#
# comparable_df_merge['comparable_tender_price'] = comparable_df_merge.comparable_parcels.apply(find_comparable_price, index_table=land_bid_index, ref_df=gls_with_index, id_col='land_parcel_id', price_col='successful_tender_price')
# comparable_df_merge['comparable_price_psm_gfa'] = comparable_df_merge.comparable_parcels.apply(find_comparable_price, index_table=land_bid_index, ref_df=gls_with_index, id_col='land_parcel_id', price_col='successful_price_psm_gfa')
# comparable_df_merge['num_comparable_parcels'] = comparable_df_merge.comparable_parcels.apply(lambda x: len(x) if isinstance(x, list) else 0)
# comparable_df_merge['total_price_error'] = abs(comparable_df_merge.comparable_tender_price / comparable_df_merge.successful_tender_price - 1)
# comparable_df_merge['price_psm_gfa_error'] = abs(comparable_df_merge.comparable_price_psm_gfa / comparable_df_merge.successful_price_psm_gfa - 1)

# comparable_df_merge[['land_parcel_id', 'method', 'successful_tender_price',
#                      'comparable_tender_price', 'successful_price_psm_gfa',
#                      'comparable_price_psm_gfa', 'total_price_error', 'price_psm_gfa_error',
#                      'num_comparable_parcels', 'comparable_parcels']] \
#     .to_csv(
#     r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\comparable_land_bidding.csv',
#     index = False)

import pandas as pd
import numpy as np
import SQL_connect
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from geopy.distance import geodesic
from glspred import extraction, utils

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


def parse_coord(coord_text: str):
    import re
    coord = re.findall(r'\((.*?)\)', coord_text)
    try:
        coord_num = extraction.extract_num(coord[0], decimal=True)
        return tuple(coord_num)
    except (IndexError, TypeError):
        return (np.nan, np.nan)


if __name__ == '__main__':
    gls_master = dbconn.read_data("""select * from data_science_test.sg_gls_bidding_master_filled_features""")
    gls_pred = dbconn.read_data("""select * from data_science_test.sg_gls_bidding_upcoming_filled_features""")
    land_bid_index = dbconn.read_data("""select * from data_science.sg_land_bidding_psm_price_hedonic_index_2022""")

    gls_all = pd.concat([gls_master, gls_pred])
    gls_all.sort_values(by=['year_launch', 'month_launch', 'day_launch'], inplace=True)
    gls_all.drop_duplicates(subset=['land_parcel_id'], keep='last', inplace=True)
    sg_gls_id = list(gls_all.sg_gls_id)
    gls_all.index = gls_all.sg_gls_id

    time_limit = 24
    distance_limit_km = 5
    area_limit = 0.2

    # main code
    comparable_final = []
    for id in tqdm(sg_gls_id, desc="Searching for comparable"):
        id_list_all = list(gls_all.sg_gls_id)
        id_list_all.remove(id)
        dat = gls_all[gls_all['sg_gls_id'] == id]

        # recency within last 24 months
        comparable_df = gls_all[
            (gls_all.launch_month_index.astype(str) > get_month_index_from(dat.launch_month_index, -time_limit))
            & ((dat.launch_month_index.values - gls_all.launch_month_index) > 0)
            ]
        if comparable_df.shape[0] <= 0:
            comparable_final.append([id, gls_all.loc[id].land_parcel_id, None, "No comparable", 0])
            continue

        # same land use
        comparable_df = comparable_df[comparable_df['land_use_type'] == dat.land_use_type.values[0]]
        if comparable_df.shape[0] <= 0:
            comparable_final.append([id, gls_all.loc[id].land_parcel_id,  None, "No comparable", 0])
            continue

        # Same region
        comparable_df = comparable_df[comparable_df['region'] == dat.region.values[0]]
        if comparable_df.shape[0] <= 0:
            comparable_final.append([id, gls_all.loc[id].land_parcel_id,  None, "No comparable", 0])
            continue

        # within certain distance
        coord_dat = parse_coord(dat.reset_index(drop=True).loc[0, 'coordinates'])
        dist_limit_bool = comparable_df.coordinates.apply(parse_coord)\
            .apply(lambda x: geodesic(x, coord_dat).km < distance_limit_km if not np.isnan(x+coord_dat).any() else False)
        comparable_df = comparable_df[dist_limit_bool]
        if comparable_df.shape[0] <= 0:
            comparable_final.append([id, gls_all.loc[id].land_parcel_id,  None, "No comparable", 0])
            continue

        # Same zone
        comparable_df_zone = comparable_df[comparable_df['zone'] == dat.zone.values[0]]
        if comparable_df_zone.shape[0] > 0:
            comparable_df = comparable_df_zone
            same_zone = ', same zone'
        else:
            # comparable_final.append([id, gls_all.loc[id].land_parcel_id, None, "No comparable", 0])
            same_zone = ''

        # Area <= 20%
        comparable_df_area = comparable_df[abs((comparable_df.site_area_sqm / dat.site_area_sqm.values[0]) - 1) < area_limit]
        if comparable_df_area.shape[0] > 0:
            est = find_comparable_price(comparable_df_area, dat, land_bid_index, price_col='price_psm_gfa_1st')
            comparable_final.append(
                [id, gls_all.loc[id].land_parcel_id, est, f"past {time_limit}m, same dev, same region, wihtin {distance_limit_km}km{same_zone}, area<{area_limit*100}%", comparable_df_area.shape[0]])
        else:
            est = find_comparable_price(comparable_df, dat, land_bid_index, price_col='price_psm_gfa_1st')
            comparable_final.append([id, gls_all.loc[id].land_parcel_id, est, f"past {time_limit}m, same dev, same region, wihtin {distance_limit_km}km{same_zone}", comparable_df.shape[0]])

    # breakpoint()
    final_df = pd.DataFrame(comparable_final,
                            columns=['sg_gls_id', 'land_parcel_id', 'comparable_price_psm_gfa', 'method', 'num_comparable_parcels'])

    gls_all_with_comparable = gls_all.reset_index(drop=True).merge(final_df[['sg_gls_id', 'comparable_price_psm_gfa']],
                                                                   how='left',
                                                                   on='sg_gls_id')
    gls_all_with_comparable = gls_all_with_comparable.merge(land_bid_index[['year_launch', 'hi_price_psm_gfa']],
                                                            how='left',
                                                            on='year_launch')

    # print out comparable prices
    gls_pred_with_comparable = gls_all_with_comparable[gls_all_with_comparable.predicting == 1][['sg_gls_id',
                                                                                                 'land_parcel_id',
                                                                                                 'land_parcel_name',
                                                                                                 'gfa_sqm',
                                                                                                 'comparable_price_psm_gfa']]
    gls_pred_with_comparable['comparable_total_tender_price'] = gls_pred_with_comparable.gfa_sqm * gls_pred_with_comparable.comparable_price_psm_gfa
    print('-' * 8, "Predicted tender price", '-' * 8)
    for i in gls_pred_with_comparable.index:
        if pd.notna(gls_pred_with_comparable.loc[i, 'comparable_price_psm_gfa']):
            if pd.notna(gls_pred_with_comparable.loc[i, 'gfa_sqm']):
                print("{}: ${:,.2f} ({:,.2f} psm of GFA)".format(gls_pred_with_comparable.loc[i, 'land_parcel_name'],
                                                                 gls_pred_with_comparable.loc[i, 'comparable_total_tender_price'],
                                                                 gls_pred_with_comparable.loc[i, 'comparable_price_psm_gfa']), sep='\n')
            else:
                print("{}: ${:,.2f} psm of GFA".format(gls_pred_with_comparable.loc[i, 'land_parcel_name'],
                                                       gls_pred_with_comparable.loc[i, 'comparable_price_psm_gfa']), sep='\n')
        else:
            print("{}: No comparable price".format(gls_pred_with_comparable.loc[i, 'land_parcel_name']))
    gls_pred_upload = gls_pred_with_comparable[['sg_gls_id',
                                                'land_parcel_id',
                                                'land_parcel_name',
                                                'comparable_price_psm_gfa',
                                                'comparable_total_tender_price']]

    breakpoint()
    utils.upload(dbconn, [gls_all_with_comparable, gls_pred_upload], ['data_science_test.sg_gls_bidding_all_filled_features_comparable_prices',
                                                                      'data_science_test.sg_gls_bidding_upcoming_predicted_prices'])


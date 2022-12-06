import pandas as pd
import numpy as np
import SQL_connect
import re

dbconn = SQL_connect.DBConnectionRS()

address_tb = dbconn.read_data('''select * from masterdata_sg.address limit 50;''')
project_tb = dbconn.read_data('''select * from masterdata_sg.project;''')
gls = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_with_index.csv')
# gls_no_proj = gls[gls.proj_name_raw.isna()][['sg_gls_id', 'land_parcel_std', 'street', 'zone', 'region',]]
gls_res = gls[(gls.devt_class == 'residential') | (gls.devt_class == 'rc')]
gls_res_prj_id = pd.merge(gls, project_tb[['project_name', 'project_dwid', 'address_dwid']], how='left', left_on='proj_name_res', right_on='project_name')


def compare_address(add1, add2):
    sep1 = [x.lower() for x in add1.split(' ')]
    sep2 = [x.lower() for x in add2.split(' ')]
    common_txt = set(sep1).intersection(set(sep2))
    r1 = len(common_txt) / len(set(sep1))
    r2 = len(common_txt) / len(set(sep2))
    return (r1 + r2)/2


for i in list(gls_res.proj_name_res.unique()):
    if i not in list(project_tb.project_name.unique()):
        print(i)

cols = list(gls_res_prj_id.columns)

save_gls = gls_res_prj_id[[cols[1]] + [cols[0]] + [cols[-2]] + [cols[-1]] + cols[2:-2]]
save_gls.to_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_1206.csv', index=False)






pass

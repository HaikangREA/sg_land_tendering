import SQL_connect
import pandas as pd
import numpy as np
from varname import nameof
from typing import List


def upload(dbconn, df: List[pd.DataFrame], tbl_name: List[str]):
    if isinstance(df, pd.DataFrame):
        df = [df]
    if isinstance(tbl_name, str):
        tbl_name = [tbl_name]
    if all(isinstance(item, list) for item in [df, tbl_name]):
        df_list = df
        tbl_list = tbl_name

        if len(df_list) != len(tbl_list):
            print("Length of dataframes and table names are not equal")
            return None

        for i in range(len(df_list)):
            print(f"Shape of dataframe{i+1}: {df_list[i].shape}", f"Name of table{i+1}: {tbl_list[i]}", sep='\n')
        check_upload = ''
        while check_upload not in ['Y', 'y'] or check_upload not in ['N', 'n']:
            check_upload = input("Confirm uploading (Y/N)\n")
            if check_upload in ['Y', 'y']:
                for i in range(len(df_list)):
                    dbconn.copy_from_df(df_list[i], tbl_list[i])
                    # print(df_list[i].shape, tbl_list[i])
                print("Uploading ended")
                break
            elif check_upload in ['N', 'n']:
                break
            else:
                print(f'"{check_upload}" is not a valid command')
    else:
        print("Type error")


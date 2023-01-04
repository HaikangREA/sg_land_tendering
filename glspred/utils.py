import SQL_connect
import pandas as pd
import numpy as np
from varname import nameof
from typing import List


def upload(dbconn, df: List[pd.DataFrame], tbl_name: List[str], auto_check_conditions: List[bool] = None):
    if isinstance(df, pd.DataFrame):
        df = [df]
    if isinstance(tbl_name, str):
        tbl_name = [tbl_name]
    if isinstance(auto_check_conditions, bool):
        auto_check_conditions = [auto_check_conditions]
    if all(isinstance(item, list) for item in [df, tbl_name]):
        df_list = df
        tbl_list = tbl_name
        acc_list = auto_check_conditions
        print("-"*48)

        if len(df_list) != len(tbl_list):
            print(f"Unequal number of dataframes and table names: {len(df_list)} and {len(tbl_list)}")
            return None

        for i in range(len(df_list)):
            print(f"Shape of dataframe{i + 1}: {df_list[i].shape}", f"Name of table{i + 1}: {tbl_list[i]}\n", sep='\n')
        check_upload = ''

        if acc_list is not None:
            check_upload = 'Y'
            for i in range(max(len(df_list), len(acc_list))):
                try:
                    statement = acc_list[i]
                    if statement is True:
                        dbconn.copy_from_df(df_list[i], tbl_list[i])
                        print(f"Dataframe{i + 1} {df_list[i].shape} uploaded as {tbl_list[i]}")
                except IndexError:
                    print(f"Unequal number of dataframes and conditions: {len(df_list)} and {len(acc_list)}")
                    if len(df_list) > len(acc_list):
                        if len(df_list) - len(acc_list) > 1:
                            addition = f" to dataframe{len(df_list)}"
                        else:
                            addition = ''
                        print(f"Unable to check for dataframe{len(acc_list) + 1}{addition}")
                    break
            print("Auto-check and uploading finished")

        while check_upload not in ['Y', 'y'] and check_upload not in ['N', 'n']:
            check_upload = input("Confirm uploading (Y/N)\n")
            if check_upload in ['Y', 'y']:
                for i in range(len(df_list)):
                    dbconn.copy_from_df(df_list[i], tbl_list[i])
                    print(f"Dataframe{i + 1} {df_list[i].shape} uploaded as {tbl_list[i]}")
                print("Uploading finished")
                break
            elif check_upload in ['N', 'n']:
                break
            else:
                print(f'"{check_upload}" is not a valid command')
    else:
        print("Type error")


if __name__ == '__main__':
    # breakpoint()
    a = pd.DataFrame(np.zeros((4, 3)))
    b = pd.DataFrame(np.zeros((3, 3)))
    lena = len(a)
    lenb = len(b)
    c = pd.concat([a, b])
    d = pd.concat([a, b, c])
    acc = lena + lenb == len(c)
    names = ['a', 'b']
    upload(0, [a, b], names)
    breakpoint()

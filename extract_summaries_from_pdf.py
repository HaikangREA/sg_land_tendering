import pandas as pd
import numpy as np
import camelot
import os
from zipfile import ZipFile
import re
import itertools


# func to extract tables from text-based pdf
def extract_pdf_tables(file: str, password: str = None, pages: str = '1', export=False, output_type: str = 'csv',
                       output: str = "extraction_output", combine: bool = False):
    import camelot
    import os
    from zipfile import ZipFile
    # camelot-py 0.10.1 documentation: https://buildmedia.readthedocs.org/media/pdf/camelot-py/latest/camelot-py.pdf
    # read tables in pdf
    tables = camelot.read_pdf(file, password=password, pages=pages)

    # specify output type in a dict
    output_type = output_type.lower()
    ext_dict = {"csv": "csv",
                "excel": "xlsx",
                "html": "html",
                "json": "json",
                "markdown": "md",
                "sqlite": "db"
                }
    # create output file name
    output_name = ".".join([output, ext_dict[output_type]])
    if export:
        tables.export(output_name, f=output_type, compress=True)  # will generate a zip containing all results

        # decompress zip to generate a folder containing all results
        zip_file = ".".join([output, "zip"])
        with ZipFile(zip_file, "r") as zip:
            zip.extractall(os.path.join(os.path.abspath('.'), output))

    if combine:
        # create a list to store all tables extracted and combine them (output will be the combined table)
        tables_list = [tb.df for tb in tables]
        table_combined = pd.concat(tables_list, ignore_index=True)
        # table_combined.to_csv(".".join([output + '_all', 'csv']), index=False, header=True)
        return table_combined

    # output is a table list object, unless combine=T
    return tables


# func to auto-detect if there's any split row in dataframe, by evaluating the ratio of blank ("") values in each row
# a new column named 'potential_split_row' will be created, the higher the value in this col, the more likely the row is split
def detect_split_row(df, inplace=False):
    df_copy = df.copy(deep=True)
    blank_count = [list(df.iloc[i, :]).count('') for i in range(df.shape[0])]
    split_row = [num/df.shape[1] for num in blank_count]
    df_copy["potential_split_row"] = split_row
    if inplace:
        df["potential_split_row"] = split_row
        return df
    return df_copy


# func to deal with problem of rows split apart
# assume rows are split from above (thus the row below is part of the row above)
# we suggest checking the tables extracted first before running this function
def amend_split_row(df, index_pair=None, auto_detect=False, limit=0.7, drop=False):
    if index_pair:
        if isinstance(index_pair, list) and len(index_pair) == 2:
            amended = [f'{a_}{b_}' for a_, b_ in zip(list(df.iloc[index_pair[0], :]), list(df.iloc[index_pair[1], :]))]
            df.iloc[index_pair[0], :] = amended
            if drop:
                df.drop(index_pair[1], axis=0, inplace=True)
            return df
        else:
            print("Index pair should be a list with length=2")

    elif auto_detect:
        detected = detect_split_row(df)
        split_index = detected[detected.potential_split_row >= limit].index

        if len(split_index) > 0:
            for idx in split_index:
                index_pair = [idx-1, idx]
                if index_pair[0] >= 0:

                    try:
                        amended = [f'{a_}{b_}' for a_, b_ in zip(list(df.iloc[index_pair[0], :]), list(df.iloc[index_pair[1], :]))]
                        df.iloc[index_pair[0], :] = amended
                        if drop:
                            df.drop(index_pair[1], axis=0, inplace=True)
                    except IndexError:
                        print("Index out of range. Some rows may have been split into more than 2 rows")
                        pass

                else:
                    print("Index out of range. The programme may have failed to read all tables")
        else:
            print("No detected split rows")
        return df

    else:
        print("Insufficient parameters")

# func to check length of num_list
def check_len(pdseries, length):
    idx_list = list(pdseries[pdseries.str.len() >= length].index)
    print(f"Length>={length}: Index{idx_list}")
    return idx_list

# func to extract num from str
def extract_num(string: str, type: str = 'all', decimal: bool = False, ignore_sep: str = None, keep: str = None):
    # 'type' means all numbers or just num between whitespaces by specifying type='between_spaces'
    # 'ignore_sep' can be 'any' to ignore all sep, or specify a sep like ',', then func won't treat ',' as a separator
    # 'keep' allows the func to keep all matched numbers or selected ones
    import re
    import itertools

    # if the input is already int or float, return itself: input=1234 -> output=1234
    if isinstance(string, int) or isinstance(string, float):
        num = string
        return num

    else:
        string = str(string)
        # # remove all spaces from string
        # string = ''.join(string.split(' '))
        try:
            # if the string can directly be converted to number, do so (e.g. input='1234' -> output=1234.0)
            num = float(string)
            return num

        except:
            pattern = r"\d+"  # find all numbers, any digits (ignore decimal number: input='$12.3' -> output=['12','3']
            if decimal:
                pattern = r"\d*\.?\d+"  # also match decimal numbers: input='$12.3' -> output='12.3'
            if type == 'between_spaces':
                pattern = r"\b" + pattern + r"\b"
                # match numbers in between white spaces
                # input='is $10.5 per box' -> output=None; input='is 10.5 dollars per box' -> output='10.5'
            num_list = re.findall(pattern, string)

            if ignore_sep:
                if ignore_sep == 'any':  # ignore any separator between numbers
                    # input='123a456,789.654' -> output='123456789654'
                    if len(num_list) >= 1:
                        num = "".join(num_list)
                        return float(num)
                    else:
                        return np.nan
                else:
                    # ignore specified separator
                    # input='$1,234,567.05' -> output ignore ',' & decimal is T='1234567.05'
                    # output without ignoring & decimal is T=['1','234','567.05']
                    string = string.replace(ignore_sep, "")
                    num_list = re.findall(pattern, string)
            num_list = [float(num) for num in num_list]  # convert all matched str item to float, stored in list

            if keep:  # to specify certain numbers to keep by index, e.g. num_list=[5, 6, 7], keep=1 -> output=[5]
                strip = [i.split(",") for i in keep.split("-")]
                # for now only support ",", for "-" will amend this later
                keep_idx = list(set([int(i) for i in list(itertools.chain.from_iterable(strip))]))
                if len(num_list) > len(keep_idx):  # if not keeping all items, raise a msg to notify
                    print(f"{len(num_list)} numbers detected")
                num_list = [num_list[i - 1] for i in keep_idx if 0 <= i - 1 < len(num_list)]

                if len(num_list) > 0:
                    return num_list[0] if len(num_list) == 1 else num_list
                else:
                    return np.nan

            if len(num_list) == 1:
                return num_list[0]  # if the result num_list has only 1 value, output the value as float
            elif len(num_list) > 1:
                return num_list  # otherwise output the whole num_list
            else:
                return np.nan # otherwise output NA


# func to decide which is gfa or gpr
# used for missing gpr or gfa (mainly in mixed devt data)
def fill_gfa_or_gpr(num):
    if isinstance(num, float) or isinstance(num, int):
        if num > 200:
            num = [num, None]
        else:
            num = [None, num]
    return num


# func to produce renaming dict
def rename_cols(old_cols=None, new_cols=None):
    return dict(zip(old_cols, new_cols))


# func to remove redundant spaces and leading & trailing spaces from text
def reduce_space(text: str) -> str:
    if pd.isna(text):
        return text
    else:
        pattern = r' +'
        return re.sub(pattern, ' ', text).strip()


# func to check the length of names (whether there's unusually long or short text as outliers)
def check_text(df, df_cols, length=True, starting_letter_case=None):
    # 'capital' is used to detect the unconformity in the case of first letter of the text
    # if all should start with uppercase, then ones with lowercase might be problem
    # 'capital' = {'upper' , 'lower'}, 'upper' means to detect text starting with uppercase letter
    if isinstance(df_cols, list) is False:
        df_cols = [df_cols]

    final_result = {}
    result = {}
    if length:
        print("=" * 15, "Length Checking", "=" * 15)
        for col in df_cols:
            # len_df = df[col].transform(len)
            len_df = df[col].apply(lambda x: len(x) if pd.notna(x) else x)
            length_low = round(len_df.min())
            length_low_index = list(len_df[len_df == length_low].index)
            length_high = round(len_df.max())
            length_high_index = list(len_df[len_df == length_high].index)
            mean = round(len_df.mean())
            print(
                f"'{col}' has average text length {mean}, ranging {length_low}{length_low_index} to {length_high}{length_high_index}")

            sub_result = {"length_low": length_low,
                          "length_high": length_high,
                          "length_low_index": length_low_index,
                          "length_high_index": length_high_index,
                          "length_mean": len_df.mean(),
                          "length_std": len_df.std()}
            result[col] = sub_result
    final_result["length"] = result

    result_case = {}
    if starting_letter_case:
        print("=" * 15, "Case Checking", "=" * 15)
        for col in df_cols:
            # ascii_1st = df[col].transform(lambda x: x.str[0]).apply(ord)
            ascii_1st = df[col].apply(lambda x: ord(str(x)[0]) if pd.notna(x) else x)
            if starting_letter_case == 'upper':
                found_text = ascii_1st[(65 <= ascii_1st) & (ascii_1st <= 90)]
            elif starting_letter_case == 'lower':
                found_text = ascii_1st[(97 <= ascii_1st) & (ascii_1st <= 122)]
            else:
                found_text = None
                print("Invalid input for 'starting_letter_case'")

            if len(found_text) > 0:
                print(
                    f"'{col}' has {len(found_text)} text starting with {starting_letter_case}-case letter{list(found_text.index)}")
                if len(found_text) == 1:
                    sub_result = {"error_case_index": list(found_text.index)[0]}
                else:
                    sub_result = {"error_case_index": list(found_text.index)}
                result_case[col] = sub_result

        if result_case:
            final_result["case"] = result_case
        else:
            print("No error detected")

    return final_result

if __name__ == "__main__":
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    # read in pdf file and extract tables
    # pdf name should be "[source]_[devt type]_xxx.pdf", e.g. 'hdb_mixed_land_sales.pdf' for land sold by HDB for mixed devt
    # accepted devt type kwargs: condo, mixed, ec
    # source: hdb, jtc
    pdf_file = "hdb_ec_land_sales.pdf"
    summary_pages = '1-6'
    name_parser = pdf_file.split('_')
    data_source = name_parser[0].lower()
    devt_type = name_parser[1].lower()

    summary = extract_pdf_tables(pdf_file, pages=summary_pages, export=True, output_type='csv', output='_'.join([data_source, devt_type]), combine=True)
    # drop rows and cols with all NA
    summary = summary.dropna(axis=0, how='all').dropna(axis=1, how='all').replace({'\n': ''}, regex=True)
    summary = summary.replace("", np.nan)

    # raise first row as header
    summary.columns = summary.iloc[0, :]
    summary.drop(0, axis=0, inplace=True)
    summary = summary.reset_index(drop=True)

    # deal with splitting row issue
    if detect_split_row(summary).potential_split_row.max() >= 0.8:
        summary = amend_split_row(summary, auto_detect=True, limit=0.8, drop=True).reset_index(drop=True)

    # standardize col names
    std_header = [
        "date_launch",
        "date_close",
        "date_award",
        "land_parcel",
        "street",
        "site_area_sqm",
        "devt_type",
        "lease_term",
        "gpr",
        "gfa_sqm",
        "num_bidders",
        "tenderer_rank",
        "tenderer_name",
        "tender_price",
        "proj_name"
    ]


    if devt_type == "mixed":
        ## create columns missing in mixed devt data
        summary.insert(loc=6, column="devt_type", value=["Mixed"] * summary.shape[0])
        summary.insert(loc=12, column="tenderer_rank", value=1)
        ## split gfa and gpr (for mixed devt)
        gfa_gpr = summary["Permissible GFA / (GPR)"].apply(extract_num, decimal=True, ignore_sep=",").apply(fill_gfa_or_gpr)
        gfa = [numlist[0] for numlist in gfa_gpr]
        gpr = [numlist[1] for numlist in gfa_gpr]
        summary["gfa_sqm"] = gfa
        summary["gpr"] = gpr
        summary.drop("Permissible GFA / (GPR)", axis=1, inplace=True)
        ## rename header
        header = list(summary.columns)
        reorder_header = header[:8] + [header[-1]] + [header[-2]] + header[-5:-3] + header[8:10] + [header[-3]]

    elif devt_type == "condo":
        ## create columns missing in hdb condo data
        summary["tenderer_rank"] = 1
        header = list(summary.columns)
        reorder_header = header[:10] + [header[12]] + [header[-1]] + header[10:12] + [header[-2]]

    elif devt_type == "ec":
        summary["tenderer_rank"] = 1
        header = list(summary.columns)
        reorder_header = header[:10] + [header[-3]] + [header[-1]] + header[-5:-3] + [header[-2]]
        #print(reorder_header)

    try:
        #summary = summary.rename(columns=rename_dict)[std_header]
        summary = summary.rename(columns=rename_cols(reorder_header, std_header))[std_header]
    except NameError:
        pass

    tp = summary.tender_price.apply(extract_num, decimal=True, ignore_sep=",")
    idx_2 = check_len(tp, 2)
    summary.num_bidders.iloc[idx_2] = tp.iloc[idx_2].apply(lambda x: x[-1])
    summary.tender_price.iloc[idx_2] = tp.iloc[idx_2].apply(lambda x: x[0])

    # extract number from strings
    num_strings = ["site_area_sqm", "lease_term", "gpr", "gfa_sqm", "num_bidders", "tender_price"]
    # summary.site_area_sqm = summary.site_area_sqm.apply(extract_num, decimal=True, ignore_sep=",")
    # summary.tender_price = summary.tender_price.apply(extract_num, decimal=True, ignore_sep=",")
    for col in num_strings:
        summary[col] = summary[col].apply(extract_num, decimal=True, ignore_sep=",")


    # clean text values
    summary = summary.replace("", np.nan)
    clean_text = summary.transform({"land_parcel": reduce_space,
                                    "street": reduce_space,
                                    "tenderer_name": reduce_space,
                                    "proj_name": reduce_space})
    summary[clean_text.columns] = clean_text

    # check quality of text values
    check_text(summary, ["land_parcel", "street", "tenderer_name", "proj_name"], starting_letter_case='lower')

    # if no big issue, export cleaned dataframe
    summary["source"] = data_source
    summary.to_csv(f"{data_source}_{devt_type}_summary.csv", header=True, index=False)



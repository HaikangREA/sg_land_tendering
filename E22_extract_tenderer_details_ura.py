# to extract ura tender details from pdf

import pandas as pd
import numpy as np
import os
import re
import pdfplumber
import camelot
from datetime import datetime
from zipfile import ZipFile


def extract_pdf_tables(file: str, password: str = None, pages: str = '1', export=False, output_type: str = 'csv',
                       output: str = "extraction_output", combine: bool = False):

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


def extract_detail(pdf_name: str):
    pdf = pdfplumber.open(pdf_name)
    pattern_parcel = '[Ll][Aa][Nn][Dd] ?[Pp][Aa][Rr][Cc][Ee][Ll] ?[a-zA-Z]? ?[Aa][Tt] (.*?) ?\n+'
    pattern_dol = '[Da][Aa][Tt][Ee] ?[Oo][Ff] ?[Ll][Aa][Uu][Nn][Cc][Hh] ?: ?(.*?) ?\n+'
    pattern_gfa = '[Gg][Rr]?[Oo]?[Ss]?[Ss]? ?[Ff][Ll]?[Oo]?[Oo]?[Rr]? ?[Aa][Rr]?[Ee]?[Aa]? ?: ?(.*?) ?[Ss]?[Qq]?[Mm]'

    try:
        raw_text = pdf.pages[0].extract_text()
    except:
        raw_text = ''

    try:
        # clean text
        raw_text = re.sub(' +', ' ', raw_text).strip()
        land_parcel = re.findall(pattern_parcel, raw_text)
        if len(land_parcel) > 1:
            print(f'Multiple land parcel name parsed: {pdf_name}')
        # formatting name
        land_parcel = land_parcel[0]
        land_parcel = re.sub(' +', ' ', land_parcel).strip()
        if '/' in land_parcel:
            land_parcel = re.sub(' ?/ ?', ' / ', land_parcel)
    except:
        land_parcel = ''

    try:
        dol = re.findall(pattern_dol, raw_text)
        if len(dol) > 1:
            print(f'Multiple date of launch parsed: {pdf_name}')
        dol = dol[0]

        # change date format
        dol = re.sub(' +', ' ', dol).strip()
        dol_num = ''
        try:
            dol_num = datetime.strptime(dol, '%d %B %Y').strftime('%Y-%m-%d')
        except:
            try:
                dol_num = datetime.strptime(dol, '%d/%m/%Y').strftime('%Y-%m-%d')
            except:
                pass
    except:
        dol_num = ''
        pass

    try:
        gfa = re.findall(pattern_gfa, raw_text)
        if len(gfa) > 1:
            print(f'Multiple gfa parsed: {pdf_name}')
        gfa = extract_num(gfa[0], decimal=True, ignore_sep=',')
    except:
        gfa = 0

    return land_parcel.title(), dol_num, gfa


if __name__ == '__main__':
    os.chdir(r'G:\REA\Working files\land-bidding\Table extraction\tenderer_details_ura\temp')

    # pdf_file = "hdb_ec_land_sales.pdf"
    # summary_pages = 6
    # name_parser = pdf_file.split('_')
    # data_source = name_parser[0].lower()
    # devt_type = name_parser[1].lower()

    tender_results = os.listdir(os.getcwd())
    pdf_page_dict = {}
    for pdf_file in tender_results:
        pdf = pdfplumber.open(pdf_file)
        pages = len(pdf.pages)
        pdf_page_dict[pdf_file] = pages
    pdf_page_df = pd.DataFrame({'file': pdf_page_dict.keys(), 'pages': pdf_page_dict.values()})
    # print(pdf_page_df.pages.describe())

    # deal with files with only 1 page
    pdf_1p = list(pdf_page_df[pdf_page_df.pages == 1].file)
    parse_info = {}

    for pdf_file in pdf_1p:
        parse_info[pdf_file] = extract_detail(pdf_file)

    pdf_list = list(parse_info.keys())
    land_parcel_list = [tuple_[0] for tuple_ in parse_info.values()]
    dol_list = [tuple_[1] for tuple_ in parse_info.values()]
    gfa_list = [tuple_[2] for tuple_ in parse_info.values()]
    info_df = pd.DataFrame({'pdf': pdf_list, 'land_parcel': land_parcel_list, 'date_of_launch': dol_list, 'site_gfa': gfa_list})
    info_df_problems = info_df[(info_df.site_gfa == 0) & (info_df.date_of_launch == '')].reset_index(drop=True)
    info_df_ok = info_df[(info_df.site_gfa != 0) & (info_df.date_of_launch != '')].reset_index(drop=True)
    extracted_df = []
    for i in range(info_df_ok.shape[0]):
        try:
            table = extract_pdf_tables(info_df_ok.pdf[i], pages='1')[0].df.iloc[1:, :]
            if table is not None:
                table['land_parcel'] = info_df_ok.land_parcel[i]
                table['date_of_launch'] = info_df_ok.date_of_launch[i]
                table['gfa_sqm'] = info_df_ok.site_gfa[i]
                table['source_file'] = info_df_ok.pdf[i]
                extracted_df.append(table)
        except:
            pass

    ext_df_combined = pd.concat(extracted_df)
    ext_df_combined = ext_df_combined.rename(columns={0: 'tenderer_rank', 1: 'tenderer_name', 2: 'tender_price', 3: 'price_psm_gfa'})

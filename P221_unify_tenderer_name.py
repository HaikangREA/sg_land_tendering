# This script is to clean and transform successful tenderer names
# input: gls table with successful tenderer info; name dictionary
# output: gls table with successful tenderer name cleaned, split and unified
# Example:
# input value -> 0 successful_tenderer_name = "CDL, MCL Land and CapitaLand Group"
# output value (in multiple rows) ->
# 0 tenderer_1 | City Developments Limited (CDL) |...
# 1 tenderer_2 | MCL Land |...
# 2 tenderer_3 | CapitaLand |...
# Also, "Acresvale Investment Pte. Ltd." and "Sherwood Development Pte. Ltd." etc are all subsidiaries of Keppel Land, they will be unified as "Keppel Land"


import pandas as pd
import numpy as np
import re
from typing import List
import SQL_connect
pd.reset_option('display.float_format')
dbconn = SQL_connect.DBConnectionRS()


def nameFormat(companyName: str)-> str:

    import re
    pte_suffix = ['[Pp]rivate', '[Pp][Tt][Ee]']
    ltd_suffix = ['[Ll]imited', '[Ll]imit', '[Ll][Tt][Dd]']

    try:
        # remove line breaks and slashes
        companyName = companyName.strip()
        companyName = re.sub(r' +', r' ', companyName)
        # companyName = re.sub(r'\\+', r'\\', companyName)
        companyName = re.sub(r'\\+n?', '', companyName)
        companyName = re.sub(r'\n', r'', companyName)

        # replace suffix with identical format
        for suffix in pte_suffix:
            pattern = f'\(?{suffix}\.?,?\)?'
            companyName = re.sub(pattern, 'Pte.', companyName)

        for suffix in ltd_suffix:
            pattern = f'\(?{suffix}\.?\)?'
            companyName = re.sub(pattern, 'Ltd.', companyName)

        companyName = re.sub('\(?[Pp][Ll][.]?\)?[\W]?$', 'Pte. Ltd.', companyName)
        companyName = re.sub('\(?[Pp][Ll][ ,./]\)?', 'Pte. Ltd.,', companyName)
        companyName = re.sub('\(?[Ii][Nn][Cc][.]?\)?[\W]?$', 'Inc.', companyName)
        companyName = re.sub('\(?[Ii]ncorporate[d]?\)?', 'Inc.', companyName)
        companyName = re.sub('\(?[Ii][Nn][Cc][.]? +\)?', 'Inc. ', companyName)
        companyName = re.sub('\(?[Jj][oint]*?[ -/&]?[Vv]e?n?t?u?r?e?[.]?\)?', 'J.V.', companyName)
        companyName = re.sub('\(?[Cc][Oo][Rr][Pp][oration]*\)?', 'Corp.', companyName)

        # identify separators and split multiple company names
        sep_id = ['[Ll][Tt][Dd]',
                  '[Ii][Nn][Cc]',
                  '[Cc][Oo][Rr][Pp]',
                  '[Gg][Mm][Bb][Hh]',
                  '[Jj].?[Vv].?',
                  '[Ll][Ll][Cc]',
                  '[Pp][Ll][Cc]',
                  '[Ll][Ll][Pp]',
                  '[Gg][Rr][Oo][Uu][Pp]']
        repl = ['Ltd', 'Inc', 'Corp', 'Gmbh', 'J.V', 'LLC', 'Plc', 'LLP', 'Group']
        repl_dict = dict(zip(sep_id, repl))
        for suffix in repl_dict.keys():
            sep_pattern_and = f'{suffix}[.]?[ ,;]?[\W]*?[Aa][Nn][Dd]? +'
            # sep_pattern_comma = 'ltd[.]?[ ]*[,;&][\W]?[,;]?[ ]?'
            sep_pattern_comma_ampersand = f'{suffix}[.]?[ ]*[,;&/][\W]?[ ]?'
            suffix_repl = repl_dict[suffix]
            companyName = re.sub(sep_pattern_and, f'{suffix_repl}. | ', companyName)
            companyName = re.sub('\.+', '.', re.sub(sep_pattern_comma_ampersand, f'{suffix_repl}. | ', companyName))

    except AttributeError:
        pass

    return companyName


def stripName(companyName: List[str])->List[str] :
    strip_pattern = [f' +{string}[.]?$' for string in ['[Ll][Tt][Dd]',
                                                       '[Pp][Tt][Ee]',
                                                       '[Pp][Tt][Ee][.]? +[Ll][Tt][Dd]',
                                                       '[Ii][Nn][Cc]',
                                                       '[Cc][Oo][Rr][Pp]',
                                                       '[Gg][Mm][Bb][Hh]',
                                                       '[Jj].?[Vv]',
                                                       '[Ll][Ll][Cc]',
                                                       '[Pp][Ll][Cc]',
                                                       '[Ll][Ll][Pp]',
                                                       '[Cc][Oo][.]?[,]? +[Pp][Tt][Ee][.]?[,]? +[Ll][Tt][Dd]',
                                                       '[Cc][Oo][.]?[,]? +[Ll][Tt][Dd]',
                                                       '[Cc][Oo][Rr][Pp][.]?[,]? +[Pp][Tt][Ee][.]?[,]? +[Ll][Tt][Dd]']]

    pattern = '|'.join(strip_pattern)
    companyName = [companyName] if isinstance(companyName, str) else companyName
    stripped_name = []
    for name in companyName:
        try:
            stripped_name.append(re.sub(pattern, '', name))
        except AttributeError and TypeError:
            stripped_name.append(name)
    return stripped_name

# texts = ['abc Pte. Ltd.', 'ccc inc', 'cbc gMbh', 'ddd J.V', 'abc Co. Pte. Ltd', np.nan, 7]
# print(stripName(texts))


def name_by_keyword(name: str, dictionary: dict):
    for text in dictionary.keys():
        if text in name.lower():
            return dictionary[text]
    return 0

# test
# company = 'REA pte., LTD, SOreal LLC and Cushman & Wakefield, Incorporated/ P&G PlC & JLL and Lasalle, consultant, IP inc.AND CDL intl pte., ltd.;,Capitaland and hdb joint-venture and cc group,and Corp. Ltd.'
# print(nameFormat(company))


def combine_same_tenderer(gls):
    dedup_df = None
    sale_id_list = gls.sg_gls_id.unique()

    for item in sale_id_list:
        df = gls[gls.sg_gls_id == item]
        df = df.drop_duplicates(subset=['unified_tenderer_name'], keep='first')
        df['num_successful_tenderers'] = len(df)
        dedup_df = pd.concat([dedup_df, df])

    return dedup_df


gls = dbconn.read_data("""select * from data_science.sg_new_full_land_bidding_filled_features;""")
name_dict = dbconn.read_data("""select * from data_science.sg_gls_bidder_name_dictionary;""")

gls = gls.dropna(subset=['successful_tenderer_name'], axis=0)
# gls.tenderer_name = gls.tenderer_name.apply(lambda x: re.sub('\(? ?[Aa][Ss] ?[Tt][Rr][Uu][Ss][Tt][Ee][Ee].*?[Tt][Rr][Uu][Ss][Tt] ?\)?', '', x))
gls["separated_names"] = gls.successful_tenderer_name.apply(nameFormat)
# gls['tenderer_rank'] = gls.tenderer_rank.apply(lambda x: int(re.findall('\d+', x)[0]))
gls["list_of_tenderers"] = gls.separated_names.apply(lambda x: x.split(' | '))
gls["num_tenderers_same_rank"] = gls.list_of_tenderers.apply(lambda x: len(x))
gls.num_tenderers_same_rank.describe()

# create additional cols for multiple tenderers
max_num_td = gls.num_tenderers_same_rank.max()
gls['tenderer_names_filled'] = gls.list_of_tenderers.apply(lambda x: x + [np.nan] * (max_num_td - len(x)))
td_num_list = []
for col in range(max_num_td):
    gls[f"tenderer_{col + 1}"] = gls.tenderer_names_filled.apply(lambda x: x[col])
    td_num_list.append(f"tenderer_{col + 1}")

# read in dictionary
# name_dict = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\ready for uploading\tenderer_name_dict_keywords.csv')
# name_dict = name_dict[['key_word', 'unified_name']]
name_dict_dic = dict(zip(list(name_dict.key_word.apply(lambda x: x.lower())), list(name_dict.unified_name)))

# unpivot table
td_df = gls[['sg_gls_id', 'separated_names', 'num_tenderers_same_rank'] + td_num_list]
td_df = td_df.melt(id_vars=['sg_gls_id', 'separated_names', 'num_tenderers_same_rank'], var_name='tenderer_id', value_name='tenderer_name')\
    .dropna(axis = 0, subset = ['tenderer_name'])
td_df['tenderer_name'] = td_df.tenderer_name.apply(lambda x: re.sub('\(.*?\)', '', x))\
    .apply(lambda x: re.sub(' +', ' ', x))\
    .apply(lambda x: x.strip())
td_df = td_df[['sg_gls_id', 'num_tenderers_same_rank', 'tenderer_id', 'tenderer_name']]
td_df['unified_name'] = td_df.tenderer_name.apply(name_by_keyword, dictionary=name_dict_dic)
# print out num of records with no unified name
td_zero = td_df[td_df.unified_name==0]
print(td_zero.shape[0])

# transform and export
td_df.rename(columns={'num_tenderers_same_rank': 'num_successful_tenderers',
                      'tenderer_id': 'successful_tenderer_id',
                      'tenderer_name': 'sep_tenderer_name',
                      'unified_name': 'unified_tenderer_name'}, inplace=True)

gls_merge_td_name = pd.merge(gls, td_df, how='right', on='sg_gls_id')\
    .drop(['num_tenderers_same_rank',
           'tenderer_names_filled'] + td_num_list, axis=1)

gls_dedup = combine_same_tenderer(gls_merge_td_name)
pass
# gls_dedup.to_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_unified_tenderer.csv', index=False)
dbconn.copy_from_df(
    gls_dedup[['sg_gls_id', 'num_successful_tenderers', 'successful_tenderer_id', 'unified_tenderer_name', 'sep_tenderer_name']],
    "data_science.sg_gls_unified_tenderer_name",
)
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def nameFormat(companyName: str)-> str:

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
        sep_id = ['[Ll][Tt][Dd]', '[Ii][Nn][Cc]', '[Cc][Oo][Rr][Pp]', '[Gg][Mm][Bb][Hh]', '[Jj].?[Vv].?', '[Ll][Ll][Cc]', '[Pp][Ll][Cc]', '[Ll][Ll][Pp]', '[Gg][Rr][Oo][Uu][Pp]']
        repl = ['Ltd', 'Inc', 'Corp', 'Gmbh', 'J.V', 'LLC', 'Plc', 'LLP', 'Group']
        repl_dict = dict(zip(sep_id, repl))
        for suffix in repl_dict.keys():
            sep_pattern_and = f'{suffix}[.]?[ ,;]?[\W]*?[Aa][Nn][Dd]? +'
            # sep_pattern_comma = 'ltd[.]?[ ]*[,;&][\W]?[,;]?[ ]?'
            sep_pattern_comma_ampersand = f'{suffix}[.]?[ ]*[,;&/][\W]?[ ]?'
            suffix_repl = repl_dict[suffix]
            companyName = re.sub(sep_pattern_and, f'{suffix_repl}. | ', companyName)
            companyName = re.sub(sep_pattern_comma_ampersand, f'{suffix_repl}. | ', companyName)

    except AttributeError:
        pass

    return companyName


gls = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\ready for uploading\gls_details_filled_full.csv')
gls.tenderer_name = gls.tenderer_name.apply(nameFormat).apply(lambda x: re.sub(' ?, ?', ' ', x))
gls.successful_tenderer_name = gls.successful_tenderer_name.apply(nameFormat).apply(lambda x: re.sub(' ?, ?', ' ', x))
gls_top1 = gls[gls.tenderer_rank <= 1]
gls_top2 = gls[gls.tenderer_rank == 2][['sg_gls_id', 'tenderer_name', 'tender_price', 'price_psm_gfa']]
gls_top2 = gls_top2.rename(columns={'tenderer_name': 'tenderer_name_2nd', 'tender_price': 'tender_price_2nd', 'price_psm_gfa': 'price_psm_gfa_2nd'})
gls_top1 = gls_top1.rename(columns={'tenderer_name': 'tenderer_name_1st', 'tender_price': 'tender_price_1st', 'price_psm_gfa': 'price_psm_gfa_1st'})

gls_spread = pd.merge(gls_top1, gls_top2, how='left', on='sg_gls_id')

gls_spread['price_premium_total'] = gls_spread.tender_price_1st - gls_spread.tender_price_2nd
gls_spread['price_premium_psm'] = gls_spread.price_psm_gfa_1st - gls_spread.price_psm_gfa_2nd
gls_spread['premium_pct'] = gls_spread.price_premium_total / gls_spread.tender_price_2nd

header = list(gls_spread.columns)
header.remove('source_file')
header.extend(['source_file'])
gls_spread = gls_spread[header]

# data check
check = gls_spread[['successful_tenderer_name', 'tenderer_name_1st', 'tenderer_name_2nd', 'num_bidders']]
check['successful_tenderer_name'] = check.successful_tenderer_name.apply(lambda x: x.lower().split(' ')[:2])
check['tenderer_name_1st'] = check.tenderer_name_1st.apply(lambda x: x.lower().split(' ')[:2])
name_err = check.loc[~(check.successful_tenderer_name == check.tenderer_name_1st)]

# make sure no comma in values
checkForComma = [col for col in header if gls_spread[gls_spread[col].astype(str).str.contains(',')].shape[0]]

# print(gls_spread.premium_pct.describe())
gls_spread.to_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\ready for uploading\gls_details_spread.csv', index=False)

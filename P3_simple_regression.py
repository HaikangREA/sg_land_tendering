
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, accuracy_score
import matplotlib.pylab as plt
import statsmodels.api as sm


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


# gls = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\ready for uploading\gls_details_filled_full.csv')
# gls.tenderer_name = gls.tenderer_name.apply(nameFormat).apply(lambda x: re.sub(' ?, ?', ' ', x))
# gls.successful_tenderer_name = gls.successful_tenderer_name.apply(nameFormat).apply(lambda x: re.sub(' ?, ?', ' ', x))
# gls_top1 = gls[gls.tenderer_rank <= 1]
# gls_top2 = gls[gls.tenderer_rank == 2][['sg_gls_id', 'tenderer_name', 'tender_price', 'price_psm_gfa']]
# gls_top2 = gls_top2.rename(columns={'tenderer_name': 'tenderer_name_2nd', 'tender_price': 'tender_price_2nd', 'price_psm_gfa': 'price_psm_gfa_2nd'})
# gls_top1 = gls_top1.rename(columns={'tenderer_name': 'tenderer_name_1st', 'tender_price': 'tender_price_1st', 'price_psm_gfa': 'price_psm_gfa_1st'})
#
# gls_spread = pd.merge(gls_top1, gls_top2, how='left', on='sg_gls_id')
#
# gls_spread['price_premium_total'] = gls_spread.tender_price_1st - gls_spread.tender_price_2nd
# gls_spread['price_premium_psm'] = gls_spread.price_psm_gfa_1st - gls_spread.price_psm_gfa_2nd
# gls_spread['premium_pct'] = gls_spread.price_premium_total / gls_spread.tender_price_2nd
#
# header = list(gls_spread.columns)
# header.remove('source_file')
# header.extend(['source_file'])
# gls_spread = gls_spread[header]
#
# # data check
# check = gls_spread[['successful_tenderer_name', 'tenderer_name_1st', 'tenderer_name_2nd', 'num_bidders']]
# check['successful_tenderer_name'] = check.successful_tenderer_name.apply(lambda x: x.lower().split(' ')[:2])
# check['tenderer_name_1st'] = check.tenderer_name_1st.apply(lambda x: x.lower().split(' ')[:2])
# name_err = check.loc[~(check.successful_tenderer_name == check.tenderer_name_1st)]
#
# # make sure no comma in values
# checkForComma = [col for col in header if gls_spread[gls_spread[col].astype(str).str.contains(',')].shape[0]]
#
# # print(gls_spread.premium_pct.describe())
# gls_spread.to_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\ready for uploading\gls_details_spread.csv', index=False)

gls = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_with_index.csv')
gls.dropna(subset=['gpr'], inplace=True)
gls = gls.loc[~(gls['zone']=='UNKNOWN')]
gls['quarter_launch'] = gls.year_launch.astype(str) + ' Q' + gls.month_launch.apply(lambda x: x//4 + 1).astype(str)
gdp = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\macroecondata.csv')
gls['lg_site_area'] = np.log(gls.site_area_sqm)
gls['lg_price_psm_real'] = np.log(gls.price_psm_real)
gls = gls.sort_values(by=['year_launch', 'month_launch', 'day_launch']).reset_index(drop=True)


features = [
    # 'zone',
    'region',
    'site_area_sqm',
    'devt_class',
    'gpr',
    'num_bidders',
    'source',
    'timediff_launch_to_close',
    # 'year_launch'
]

target = ['price_psm_real']

gls_feat = gls[features]
gls_feat_dummy = pd.get_dummies(gls_feat, drop_first=True)
gls_target = gls[target]

x_train, x_test, y_train, y_test = train_test_split(gls_feat_dummy, gls_target, random_state=42, test_size=0.2)

reg = LinearRegression()
reg.fit(x_train, y_train)
y_hat = reg.predict(x_train)
r2_score(y_hat, y_train)

x = sm.add_constant(x_train)
reg2 = sm.OLS(y_train, x).fit()
print(reg2.summary())
y_hat_test = reg.predict(x_test)
df_test = pd.DataFrame(y_hat_test, columns=['predict'])
y_test = y_test.reset_index(drop=True)
df_test['target'] = y_test
df_test['residual'] = df_test['target'] - df_test['predict']
df_test['diff%'] = np.absolute(df_test['residual'] / df_test['target']*100)
df_test = df_test.sort_values(by='diff%')
print(r2_score(y_test, y_hat_test))
print(mean_absolute_percentage_error(y_test, y_hat_test))

# plt.scatter(y_train, y_hat)
# plt.show()
print(gls_feat.devt_class.nunique())




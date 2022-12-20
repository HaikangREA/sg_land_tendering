import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging
import SQL_connect

logger = logging.getLogger(__name__)
dbconn = SQL_connect.DBConnectionRS()


def hedonic(transactions, y_col, m_or_qtr, index_len):
    HEDONIC_PENALTY = 1.5  # multiplier to evaluation metrics (MSE), to favour more for repeated-sales method

    cat_val = [
        m_or_qtr,
        "devt_class",
        "source",
        "zone",
        "region"
    ]
    num_val = [
        "site_area_sqm",
        "gpr",
        "lease_term",
        "num_bidders",
        'timediff_launch_to_close',
        'proj_num_of_units',
        'proj_max_floor',
        'dist_to_cbd',
        'dist_to_mrt',
        'dist_to_bus_stop',
        'dist_to_school',
        'comparable_price_psm_gfa'
    ]
    if "age" in transactions.columns:
        transactions["new_building"] = transactions["age"].apply(
            lambda a: True if a < 0 else False
        )
    col = transactions.columns[transactions.nunique() > 1]
    cat_val = list(np.intersect1d(col, cat_val))
    num_val = list(np.intersect1d(col, num_val))

    if num_val:
        scaler = MinMaxScaler()
        transactions[num_val] = scaler.fit_transform(transactions[num_val])
    transactions = transactions.dropna(how='any', subset=[y_col])
    y = np.log(transactions[y_col])
    hedonic_index = None
    mse = 1
    try:
        x = pd.get_dummies(
            data=transactions[num_val + cat_val], columns=cat_val, drop_first=False
        )
        x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    except ValueError:
        logger.warning("ValueError when converting x to dummies")
        return hedonic_index, mse

    try:
        model = LinearRegression()
        fit = model.fit(x, y)
        r2 = model.score(x, y)
        r2_adj = 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)
        print(f"R2 score {r2} \nAdjusted R2 score {r2_adj}")
    except LinAlgError:
        logger.warning("LinAlgError encountered, skip")
        return hedonic_index, mse
    except ValueError:
        logger.warning("ValueError encountered, skip")
        return hedonic_index, mse
    hedonic_index = pd.DataFrame(
        [[fit.coef_[i], s] for i, s in enumerate(x.columns) if m_or_qtr in s],
        columns=["hi", m_or_qtr],
    )

    if len(hedonic_index) < index_len:
        mse = 1
        hedonic_index = None
    else:
        hedonic_index.sort_values(by=m_or_qtr, inplace=True)
        mse = mean_squared_error(y, fit.predict(x)) * HEDONIC_PENALTY

        hedonic_index[m_or_qtr] = hedonic_index[m_or_qtr].apply(
            lambda x: x.split("_")[-1]
        )
        base_coef = hedonic_index.hi.iloc[0]
        hedonic_index["hi"] = hedonic_index["hi"] - base_coef
        hedonic_index["hi"] = hedonic_index["hi"].apply(lambda x: np.exp(x))
        logger.info(f"categorical cols: {cat_val}")
        logger.info(f"numerical cols: {num_val}")

    return hedonic_index, mse


def rebase(series, base=0):
    base_value = series[base]
    return series.apply(lambda x: x/base_value)


gls = dbconn.read_data('''select * from data_science.sg_new_full_land_bidding_filled_features''')
y_col = 'successful_price_psm_gfa'
hi = hedonic(gls, y_col, 'year_launch', 5)[0]
hi_rebase = pd.DataFrame({"hi_price_psm_gfa": rebase(hi.hi, base=len(hi)-1), "year_launch": hi.year_launch})
hi_rebase.insert(loc=0, column="mean_price_psm_gfa", value=gls[[y_col, "year_launch"]].groupby(by="year_launch").mean().values)
hi_rebase['year_launch'] = hi_rebase.year_launch.astype(int)
# hi_rebase["mean_price_psm_gfa"] = rebase(hi_rebase.mean_price_psm_gfa, 10)

check = 42
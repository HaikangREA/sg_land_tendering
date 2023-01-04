import SQL_connect
import pandas as pd
import numpy as np
from glspred import utils
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error as mape

dbconn = SQL_connect.DBConnectionRS()

# read in data
gls = dbconn.read_data('''select * from data_science_test.sg_gls_bidding_all_filled_features_comparable_prices''')
pred_prices = dbconn.read_data('''select * from data_science_test.sg_gls_bidding_upcoming_predicted_prices''')

# split predicting data
pred = gls[gls.predicting == 1]

# training data
gls = gls.drop(pred.index, axis=0)

# select target
gls['price_psm_real'] = gls.successful_price_psm_gfa / gls.hi_price_psm_gfa
target = 'price_psm_real'

# select features
gls['num_winners'] = gls.tenderer_name_1st.apply(lambda x: len(x.split('|')))
gls['joint_venture'] = gls.num_winners.apply(lambda x: 1 if x > 1 else 0)

cat_cols = ['region',  #
            'zone',  #
            'land_use_type',  # 'devt_class',
            'project_type',  # 'devt_type',
            'source'
            ]
num_cols = ['latitude',
            'longitude',
            'site_area_sqm',
            'gpr',
            'lease_term',
            'num_bidders',  # should be dynamic
            'joint_venture',  # should be dynamic
            'year_launch',
            'timediff_launch_to_close',  #
            'proj_num_of_units',  # should be dynamic
            'proj_max_floor',  # should be dynamic
            'num_nearby_parcels_3km_past_6month',  # use format
            'num_proj_nearby_2km_past_5years',  # use format
            'num_mrt_stations_1km',  #
            'num_mrt_lines_1km',  #
            'num_bus_stop_500m',  #
            'num_school_1km',  #
            'dist_to_nearest_parcel_launched_past_6month',  #
            'dist_to_nearest_proj_completed_past_5years',  #
            'dist_to_cbd',
            'dist_to_mrt',
            'dist_to_bus_stop',
            'dist_to_school',
            # 'comparable_price_psm_gfa'  #
            ]

feature_cols = cat_cols + num_cols
cols = feature_cols + [target]

# pre-process
gls = gls.sort_values(by=['year_launch', 'month_launch', 'date_launch'])
gls = gls.dropna(subset=[target]).reset_index(drop=True)
missing_values = gls.isna().sum()

# split data by randomly selecting most recent records as testing data
time_threshold = 2011
gls_dummy = pd.get_dummies(gls[cols]).dropna().reset_index(drop=True)
gls_train = gls_dummy[gls_dummy.year_launch < time_threshold]
gls_test = gls_dummy[gls_dummy.year_launch >= time_threshold]
x = gls_train.drop(target, axis=1)
y = gls_train[target]
x_test_to_select = gls_test.drop(target, axis=1)
y_test_to_select = gls_test[target]
# dmat = DMatrix(data=x, label=y)

x_train_to_append, x_test, y_train_to_append, y_test = train_test_split(x_test_to_select,
                                                                        y_test_to_select,
                                                                        test_size=0.5,
                                                                        random_state=42)
x_train = pd.concat([x, x_train_to_append])
y_train = pd.concat([y, y_train_to_append])
test_size = round(len(x_test) / len(gls), 2)
# breakpoint()

# param tuning
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
ridge = Ridge()
grid = dict()
grid['alpha'] = np.linspace(0, 0.2, 21)
gscv = GridSearchCV(ridge, grid, scoring='neg_mean_absolute_percentage_error', cv=cv, n_jobs=-1)
results = gscv.fit(x_train, y_train)
print('MAPE: %.5f' % np.absolute(results.best_score_))
print('Config: %s' % results.best_params_)
breakpoint()

# train and validate
alpha_tuned = results.best_params_['alpha']
model = Ridge(alpha=alpha_tuned).fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_percentage_error', cv=cv, n_jobs=-1)
scores = np.absolute(scores)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train_mape = mape(y_train, y_train_pred)
test_mape = mape(y_test, y_test_pred)
print(f"Test size: {test_size}", f"Train MAPE: {train_mape}", f"Test MAPE: {test_mape}", f"CV MAPE: \n{scores}",
      sep='\n')
breakpoint()

# predict
pred_x = pd.get_dummies(pred[feature_cols])
for col in x_train.columns:
    if col not in pred_x.columns:
        print(col)
        pred_x[col] = 0
pred_x = pred_x[model.feature_names_in_]

# manual filling
pred_x.num_bidders.fillna(3, inplace=True)
pred_x['proj_max_floor'] = [18, 40, 20, 22]

pred_y = model.predict(pred_x)

# print out result
pred_result = pd.concat([pred[['land_parcel_id', 'land_parcel_name', 'gfa_sqm']].reset_index(drop=True),
                         pd.DataFrame(pred_y, columns=['ridge_psm_price'])], axis=1)
pred_result['ridge_total_price'] = pred_result.gfa_sqm * pred_result.ridge_psm_price
print('-' * 24, "Predicted tender price", '-' * 24)
for i in pred_result.index:
    print("{}: ${:,.2f} ({:,.2f} psm of GFA)".format(pred_result.loc[i, 'land_parcel_name'],
                                                     pred_result.loc[i, 'ridge_total_price'],
                                                     pred_result.loc[i, 'ridge_psm_price']), sep='\n')
breakpoint()
# append to predicted prices and upload
pred_prices_upload = pred_prices.merge(pred_result[['land_parcel_id',
                                                    'ridge_psm_price',
                                                    'ridge_total_price']], how='left', on='land_parcel_id')
utils.upload(dbconn,
             df=pred_prices_upload,
             tbl_name='data_science_test.sg_gls_bidding_upcoming_predicted_prices',
             auto_check_conditions=pred_prices_upload.shape[0] == pred.shape[0])
breakpoint()


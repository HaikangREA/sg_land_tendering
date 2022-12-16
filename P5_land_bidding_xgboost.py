import pandas as pd
import numpy as np
import SQL_connect
from xgboost import XGBRegressor, DMatrix, cv, train, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape, r2_score
import matplotlib.pylab as plt
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

dbconn = SQL_connect.DBConnectionRS()

# read in data
gls = dbconn.read_data('''select * from data_science.sg_land_bidding_filled_features_with_comparable_price;''')
# gls = dbconn.read_data('''  with
#                             sch as(
#                                 select land_parcel_id, count(school) as num_school_1km
#                                 from data_science.sg_land_parcel_school_distance
#                                 where distance < 1000
#                                 group by 1)
#                                 ,
#                             bus as(
#                                 select land_parcel_id, count(bus_stop) as num_bus_stop_500m
#                                 from data_science.sg_land_parcel_bus_stop_distance
#                                 where distance < 500
#                                 group by 1)
#                                 ,
#                             mrt as(
#                                 select land_parcel_id, count(mrt_station) as num_mrt_1km
#                                 from data_science.sg_land_parcel_mrt_distance
#                                 where distance < 1000
#                                 group by 1)
#                             select *
#                             from data_science.sg_new_full_land_bidding_filled_features
#                                 left join mrt
#                                 using (land_parcel_id)
#                                 left join bus
#                                 using (land_parcel_id)
#                                 left join sch
#                                 using (land_parcel_id)
#                             ;
#                             ''')
pred = dbconn.read_data('''select * from data_science.sg_gls_land_parcel_for_prediction;''')

# select target
target = 'price_psm_real'
# select features
cat_cols = ['region',
            'zone',
            'devt_class',
            'source'
            ]
num_cols = ['site_area_sqm',
            'lease_term',
            'gpr',
            'num_bidders',
            'year_launch',
            'timediff_launch_to_close',
            'proj_num_of_units',
            'proj_max_floor',
            # 'num_of_nearby_completed_proj_200m',
            'num_mrt_1km',
            'num_bus_stop_500m',
            'num_school_1km',
            'comparable_price_psm_gfa'
            ]
cols = cat_cols + num_cols

# pre-process
gls = gls.sort_values(by=['year_launch', 'month_launch', 'date_launch'])
gls = gls.dropna(subset=[target]).reset_index(drop=True)
gls = gls.fillna(pd.DataFrame(np.zeros((gls.shape[0], 3)),
                              columns=['num_mrt_1km',
                                       'num_bus_stop_500m',
                                       'num_school_1km']
                              )
                 )
x = pd.get_dummies(gls[cols])
y = gls[target]
dmat = DMatrix(data=x, label=y)

# initial model with default params
test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=test_size)
train_data, test_data = DMatrix(x_train, label=y_train), \
                        DMatrix(x_test, label=y_test)

# initialize xgboost and record result
xgb = XGBRegressor(objective='reg:squarederror').fit(x_train, y_train)
pred_train, pred_test = xgb.predict(x_train), \
                        xgb.predict(x_test)
mape_train, mape_test = mape(y_train, pred_train), \
                        mape(y_test, pred_test)
res_df = pd.DataFrame({'mape_train': [mape_train],
                       'mape_test': [mape_test]}
                      )
key_params = ['max_depth',
              'learning_rate',
              'gamma',
              'reg_lambda',
              'min_child_weight'
              ]
initial_params = xgb.get_params()
for param in key_params:
    res_df[param] = initial_params[param]

# random search cv
# define search space
param_space = {'max_depth': [6, 7, 8],
               'learning_rate': [0.02, 0.025, 0.03],
               'gamma': [0.10, 0.15, 0.25],
               'reg_lambda': [1.0, 1.05, 1.2],
               'min_child_weight': [3, 4, 5]
               }
scoring = ['neg_mean_absolute_percentage_error']
kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
random_search = RandomizedSearchCV(estimator=xgb,
                                   param_distributions=param_space,
                                   n_iter=50,
                                   scoring=scoring,
                                   refit=scoring[0],
                                   n_jobs=-1,
                                   cv=kfold,
                                   verbose=1)

# fit data
random_search_res = random_search.fit(x_train, y_train)
# Print the best score and the corresponding hyperparameters
print(f'MAPE: {-random_search_res.best_score_:.4f}')
print(f'Hyperparameters: {random_search_res.best_params_}')

# apply tuned params
param_tuned = random_search_res.best_params_
xgb_tuned = train(params=param_tuned, dtrain=train_data, num_boost_round=100)
pred_train, pred_test = xgb_tuned.predict(train_data), \
                        xgb_tuned.predict(test_data)
mape_train, mape_test = mape(y_train, pred_train), \
                        mape(y_test, pred_test)
res_tuned = [mape_train, mape_test] + [param_tuned[param] for param in key_params]
res_df.loc[len(res_df)] = res_tuned

# # apply tuned model
# train_data = DMatrix(x, label=y)
# param_tuned = {'max_depth': 7,
#                'learning_rate': 0.02,
#                'gamma': 0.25,
#                'reg_lambda': 0.85,
#                'min_child_weight': 4
#                }
# xgb_tuned = train(params=param_tuned, dtrain=train_data, num_boost_round=100)
# pred_train = xgb_tuned.predict(train_data)
# prediction = xgb_tuned.predict(pred_x)
# mape = mape(y, pred_train)
print("Test size: %f" %test_size, "MAPE train: %f" %mape_train, "MAPE test: %f" %mape_test, "MAPE test-train: %f" %(mape_test-mape_train), sep='\n')

# # cross validation
# eval_res = cv(dtrain=train_data,
#               params=param_tuned,
#               nfold=3,
#               num_boost_round=100,
#               early_stopping_rounds=10,
#               metrics='mape',
#               as_pandas=True,
#               seed=42)

check = 42

# real life prediction
dynamic_var = ['num_bidders', 'proj_max_floor']
# pred.loc[0, 'proj_max_floor'] = 50
# pred.loc[1, 'proj_max_floor'] = 24
pred.loc[0, 'zone'] = 'downtown core'
feat = x.columns

pred_x = pd.get_dummies(pred[cols])
for col in [item for item in feat if item not in pred_x.columns]:
    pred_x[col] = 0
train_data = DMatrix(x, label=y)
param_tuned = {'max_depth': 7,
               'learning_rate': 0.02,
               'gamma': 0.25,
               'reg_lambda': 0.85,
               'min_child_weight': 4
               }
xgb_tuned = train(params=param_tuned, dtrain=train_data, num_boost_round=100)
pred_train = xgb_tuned.predict(train_data)
pred_x = pred_x[xgb_tuned.feature_names]
pred_x_dmat = DMatrix(pred_x)
prediction = xgb_tuned.predict(pred_x_dmat)
mape = mape(y, pred_train)

for i in range(len(pred_x)):
    print("Predicted tender price for {}: {:.2f}".format(pred.loc[i, 'land_parcel_std'],
                                                         pred_x.loc[i, 'site_area_sqm'] * pred_x.loc[i, 'gpr'] *
                                                         prediction[i]))

# for marina: 880,000,000 (MTD) / 1,180,000,000 (Yuelin)
# for lentor: 520,000,000

fig, ax = plt.subplots(figsize=(10, 10))
plot_importance(xgb_tuned, max_num_features=20, height=0.5, ax=ax, importance_type='gain')
plt.show()

check = 42

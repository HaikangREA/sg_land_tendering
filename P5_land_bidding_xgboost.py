import pandas as pd
import numpy as np
from xgboost import XGBRegressor, DMatrix, cv, train, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape, r2_score
import matplotlib.pylab as plt
import graphviz
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

# read in data
gls = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\feature eng\gls_1207_new_feat.csv')
gls = gls.sort_values(by=['year_launch', 'month_launch', 'date_launch'])

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
            'timediff_launch_to_close',
            'avg_dist_cbd',
            'avg_dist_mrt',
            'avg_num_bus',
            'avg_num_good_sch',
            'proj_num_of_units',
            'proj_max_floor',
            'num_of_nearby_completed_proj_200m',
            'num_of_schools',
            'year_launch'
            ]
cols = cat_cols + num_cols

# pre-process
x = pd.get_dummies(gls[cols])
y = gls.price_psm_real
dmat = DMatrix(data=x, label=y)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.2)
train_data, test_data = DMatrix(x_train, label=y_train), \
                        DMatrix(x_test, label=y_test)

# initialize xgboost and record result
xgb = XGBRegressor(objective = 'reg:squarederror').fit(x_train, y_train)
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
               'learning_rate': [0.01, 0.02, 0.03],
               'gamma': [0.5, 0.75, 1.0],
               'reg_lambda': [0.5, 1.0, 1.5],
               'min_child_weight': [4, 5, 6]
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
params_tuned = random_search_res.best_params_
xgb_tuned = train(params=params_tuned, dtrain=train_data, num_boost_round=100)
pred_train, pred_test = xgb_tuned.predict(train_data), \
                        xgb_tuned.predict(test_data)
mape_train, mape_test = mape(y_train, pred_train), \
                        mape(y_test, pred_test)
res_tuned = [mape_train, mape_test] + [params_tuned[param] for param in key_params]
res_df.loc[len(res_df)] = res_tuned





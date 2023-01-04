import pandas as pd
import numpy as np
import SQL_connect
from glspred import utils
from xgboost import XGBRegressor, DMatrix, cv, train, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape, r2_score
import matplotlib.pylab as plt
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV


def tune_param_rs(x_train, y_train, x_test, y_test, test_size, params):
    scoring = ['neg_mean_absolute_percentage_error']
    kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    random_search = RandomizedSearchCV(estimator=xgb,
                                       param_distributions=params,
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
    xgb_tuned = train(params=param_tuned, dtrain=DMatrix(x_train, label=y_train), num_boost_round=1000)
    pred_train, pred_test = xgb_tuned.predict(train_data), \
                            xgb_tuned.predict(test_data)
    mape_train, mape_test = mape(y_train, pred_train), \
                            mape(y_test, pred_test)
    res_tuned = [mape_train, mape_test] + [param_tuned[param] for param in key_params]
    res_df.loc[len(res_df)] = res_tuned

    return random_search_res, mape_train, mape_test, param_tuned


if __name__ == '__main__':
    dbconn = SQL_connect.DBConnectionRS()

    # read in data
    gls = dbconn.read_data('''select * from data_science_test.sg_gls_bidding_all_filled_features_comparable_prices''')
    pred_prices = dbconn.read_data('''select * from data_science_test.sg_gls_bidding_upcoming_predicted_prices''')
    # # read in infra table
    # infra = dbconn.read_data(''' select * from data_science.sg_land_parcel_distance_to_infrastructure''')
    #
    # # read in nearby parcels data
    # dist_parcels = dbconn.read_data(''' select land_parcel_id_a as land_parcel_id, min(distance_m) as dist_to_nearest_parcel_launched_past_6m
    #                                     from data_science.sg_gls_pairwise_nearby_land_parcels
    #                                     where launch_time_diff_days >= 0
    #                                     and launch_time_diff_days <= 180
    #                                     and land_parcel_id_a != land_parcel_id_b
    #                                     group by 1
    #                                     ;''')
    #
    # nearby_parcels = dbconn.read_data('''   select land_parcel_id_a as land_parcel_id, count(*) as num_nearby_parcels_3km_past_6m
    #                                         from data_science.sg_gls_pairwise_nearby_land_parcels
    #                                         where launch_time_diff_days >= 0
    #                                         and launch_time_diff_days <= 180
    #                                         and land_parcel_id_a != land_parcel_id_b
    #                                         and distance_m <= 3000
    #                                         group by 1
    #                                         ;''')
    #
    # dist_proj = dbconn.read_data('''select land_parcel_id , min(distance) as dist_to_nearest_proj
    #                                 from data_science.sg_land_parcel_filled_info_distance_to_project
    #                                 where land_use_big = proj_devt_class
    #                                 and land_launch_year >= proj_completion_year
    #                                 and distance >= 10
    #                                 group by 1
    #                                 ;''')
    #
    # nearby_proj = dbconn.read_data('''  select land_parcel_id , count(project_dwid) as num_proj_nearby_2km_5yr
    #                                     from data_science.sg_land_parcel_filled_info_distance_to_project
    #                                     where land_use_big = proj_devt_class
    #                                     and distance > 10
    #                                     and distance < 2000
    #                                     and land_launch_year - proj_completion_year > 0
    #                                     and land_launch_year - proj_completion_year < 5
    #                                     group by 1;''')

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
                'comparable_psm_price'  #
                ]

    feature_cols = cat_cols + num_cols
    cols = feature_cols + [target]

    # pre-process
    gls = gls.sort_values(by=['year_launch', 'month_launch', 'date_launch'])
    gls = gls.dropna(subset=[target]).reset_index(drop=True)

    # split data by randomly selecting most recent records as testing data
    time_threshold = 2012
    gls_dummy = pd.get_dummies(gls[cols])
    gls_train = gls_dummy[gls_dummy.year_launch < time_threshold]
    gls_test = gls_dummy[gls_dummy.year_launch >= time_threshold]
    x = gls_train.drop(target, axis=1)
    y = gls_train[target]
    x_test_to_select = gls_test.drop(target, axis=1)
    y_test_to_select = gls_test[target]
    # dmat = DMatrix(data=x, label=y)

    x_train_to_append, x_test, y_train_to_append, y_test = train_test_split(x_test_to_select, y_test_to_select,
                                                                            test_size=0.5, random_state=42)
    x_train = pd.concat([x, x_train_to_append])
    y_train = pd.concat([y, y_train_to_append])
    test_size = round(len(x_test) / len(gls), 2)
    # breakpoint()

    # initial model with default params
    # # train-test split
    # test_size = 0.2
    # x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=test_size)
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
    param_space = {'max_depth': np.arange(3, 10),
                   'learning_rate': np.arange(0.01, 0.3, 0.01),
                   'gamma': np.arange(0, 1, 0.05),
                   'reg_lambda': np.arange(0.1, 5, 0.1),
                   'min_child_weight': np.arange(3, 10),
                   'subsample': np.arange(0.5, 1, 0.1),
                   'colsample_bytree': np.arange(0.5, 1, 0.01),
                   }

    # random_search_output = tune_param_rs(x_train, y_train, x_test, y_test, test_size, params=param_space)
    # random_search_res = random_search_output[0]
    # mape_train, mape_test = random_search_output[1], random_search_output[2]
    # breakpoint()

    #
    # print("Test size: %f" % test_size,
    #       "MAPE train: %f" % mape_train,
    #       "MAPE test: %f" % mape_test,
    #       "MAPE test-train: %f" % (mape_test - mape_train),
    #       sep='\n')

    # param_tuned = random_search_output[3]
    param_tuned = {'subsample': 0.55,
                   'reg_lambda': 1.7,
                   'min_child_weight': 3,
                   'max_depth': 5,
                   'learning_rate': 0.01,
                   'gamma': 0.5,
                   'colsample_bytree': 0.55
                   }

    # test for over-fitting
    xgb_test = train(params=param_tuned, dtrain=train_data, num_boost_round=1000)
    pred_train, pred_test = xgb_test.predict(DMatrix(x_train)), \
                            xgb_test.predict(DMatrix(x_test))
    mape_train, mape_test = mape(y_train, pred_train), \
                            mape(y_test, pred_test)

    xgb_cv = cv(params=param_tuned,
                dtrain=train_data,
                nfold=3,
                num_boost_round=1000,
                early_stopping_rounds=100,
                metrics='mape',
                as_pandas=True, seed=42)

    print(xgb_cv.tail())
    print("Test size: %f" % test_size,
          "MAPE train: %f" % mape_train,
          "MAPE test: %f" % mape_test,
          "MAPE test-train: %f" % (mape_test - mape_train),
          sep='\n')
    breakpoint()

    # prediction
    feat = x.columns

    for col in [item for item in feat if item not in pred.columns]:
        pred[col] = np.nan
    pred_x = pd.get_dummies(pred[feature_cols])

    # dmat_pred = DMatrix(x, label=y)
    dmat_pred = train_data
    xgb_tuned = train(params=param_tuned, dtrain=dmat_pred, num_boost_round=1000)
    pred_train = xgb_tuned.predict(dmat_pred)
    pred_test = xgb_tuned.predict(test_data)

    for col in [item for item in xgb_tuned.feature_names if item not in pred_x.columns]:
        pred_x[col] = np.nan

    pred_x = pred_x[xgb_tuned.feature_names]
    pred_x_dmat = DMatrix(pred_x)
    hi = gls[gls.year_launch == 2022].hi_price_psm_gfa.mean()
    prediction = xgb_tuned.predict(pred_x_dmat)
    mape = mape(y_test, pred_test)

    # print out predicted result
    pred_result = pd.concat([pred[['land_parcel_id', 'land_parcel_name', 'gfa_sqm']].reset_index(drop=True),
                             pd.DataFrame(prediction, columns=['xgb_psm_price'])], axis=1)
    pred_result['xgb_total_price'] = pred_result.gfa_sqm * pred_result.xgb_psm_price

    # pred_index_list = pred_x.index
    print('-' * 24, "Predicted tender price", '-' * 24)
    for i in pred_result.index:
        print("{}: ${:,.2f} ({:,.2f} psm of GFA)".format(pred_result.loc[i, 'land_parcel_name'],
                                                         pred_result.loc[i, 'xgb_total_price'],
                                                         pred_result.loc[i, 'xgb_psm_price']), sep='\n')

    # for marina: 880,000,000 (MTD) / 1,180,000,000 (Yuelin)
    # for lentor: 520,000,000

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_importance(xgb_tuned, max_num_features=20, height=0.5, ax=ax, importance_type='gain')
    plt.show()

    breakpoint()
    # append to predicted prices and upload
    pred_prices_upload = pred_prices.merge(pred_result[['land_parcel_id',
                                                        'xgb_psm_price',
                                                        'xgb_total_price']], how='left', on='land_parcel_id')
    utils.upload(dbconn,
                 df=pred_prices_upload,
                 tbl_name='data_science_test.sg_gls_bidding_upcoming_predicted_prices',
                 auto_check_conditions=pred_prices_upload.shape[0] == pred.shape[0])
    breakpoint()

# Import Packages
####################################################################
import numpy as np, pandas as pd
from random import shuffle
from tqdm import tqdm
from __future__ import division
import pyodbc
from sqlalchemy import create_engine, MetaData, Table, select
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from scipy import sparse
from scipy.sparse import vstack
from plotnine import *
from ggplot import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import time

# Database Connection
###################################################################
def my_create_engine(mydsn = '<...>', mydatabase = '<...>', **kwargs):
    conn = pyodbc.connect(
    r'DSN=<...>;'
    r'UID=<...>;'
    r'PWD=<...>;'
    )
    cursor = conn.cursor()
    connection_string = 'mssql+pyodbc://@%s' % mydsn
    cargs = {'database': mydatabase}
    cargs.update(**kwargs)
    e = create_engine(connection_string, connect_args=cargs)
    return e, cursor, conn

def tbl_write(tbl_name, engine, pandas_df):
    pandas_df.to_sql(tbl_name, engine, if_exists = 'append', chunksize = None, index = False)

eng, cursor, conn = my_create_engine()

# Import .csv Files
###################################################################
application_test = pd.read_csv('.../application_test.csv')
application_train = pd.read_csv('.../application_train.csv')
bureau = pd.read_csv('.../bureau.csv')
bureau_balance = pd.read_csv('.../bureau_balance.csv')
credit_card_balance = pd.read_csv('.../credit_card_balance.csv')
installments_payments = pd.read_csv('.../installments_payments.csv')
pos_cash_balance = pd.read_csv('.../pos_cash_balance.csv')
previous_application = pd.read_csv('.../previous_application.csv')

# Write Tables to SQL Server
###################################################################
tbl_write('application_test', eng, application_test)
tbl_write('application_train', eng, application_train)
tbl_write('bureau', eng, bureau)
tbl_write('bureau_balance', eng, bureau_balance)
tbl_write('credit_card_balance', eng, credit_card_balance)
tbl_write('installments_payments', eng, installments_payments)
tbl_write('pos_cash_balance', eng, pos_cash_balance)
tbl_write('previous_application', eng, previous_application)

# Import Manipulated Data from SQL Server DB
###################################################################
trn = pd.read_sql("select * from DB_GENERAL.DBO.HOME_CREDIT_AGG WHERE TRAIN_TEST = 'train'", eng.connect())
tst = pd.read_sql("select * from DB_GENERAL.DBO.HOME_CREDIT_AGG WHERE TRAIN_TEST = 'test'", eng.connect())

trn_meta = trn[['SK_ID_CURR', 'TRAIN_TEST']]
trn = trn.drop(['SK_ID_CURR', 'TRAIN_TEST'],axis = 1)

tst_meta = tst[['SK_ID_CURR', 'TRAIN_TEST']]
tst = tst.drop(['SK_ID_CURR', 'TRAIN_TEST'],axis = 1)

trn_x = sparse.coo_matrix(trn.drop(['TARGET'], axis = 1))
tst_x = sparse.coo_matrix(tst.drop(['TARGET'], axis = 1))
trn_y = trn['TARGET']
tst_y = tst['TARGET']

# Validation Split
######################################################################################
trn_y, val_y, trn_x, val_x = train_test_split(trn_y, trn_x, test_size = .2, random_state = 2018)

# Parameter Tuning
######################################################################################
# Parameters Attempted in Grid Search Cross Validation
start_tm = time.time()
params = {
 'max_depth': [4,6,8,10,12],
 #'min_child_weight': [4,5,7,10,13],
 'colsample_bytree': [0.2,0.4,0.6],
 #'colsample_bylevel': [0.2,0.4,0.6,0.8],
 'learning_rate': [.01]
}

gsrch = GridSearchCV(estimator = XGBClassifier( 
        objective= 'binary:logistic', 
        seed = 52718), 
    param_grid = params, 
    scoring = 'roc_auc',
    cv=4,
    verbose = 1)

gsrch.fit(trn_x, trn_y)
gsrch.best_score_
gsrch.best_params_
end_tm = time.time()
print('Execution Time: ' + str((end_tm - start_tm)/60) + str(' min.'))
# Notes on Grid Search Results

# Fitting Final Model
######################################################################################
# Parameters
params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'gbtree'
params['eval_metric'] = 'auc'
params['eta'] = 0.025
params['max_depth'] = 6
params['min_child_weight'] = 12
params['colsample_bytree'] = 0.65
params['colsample_bylevel'] = 0.25
params['subsample'] = 0.8
params['reg_lambda'] = .05
params['reg_alpha'] = 0
params['min_split_loss'] = 2
params['scale_pos_weight'] = float(331430 / 24825)
nrnd = 10000

dat_train = xgb.DMatrix(trn_x, label=trn_y)
dat_valid = xgb.DMatrix(val_x, label=val_y)
dat_test = xgb.DMatrix(tst_x, label = tst_y)
watchlist = [(dat_train, 'train'), (dat_valid, 'valid')]
xgb_trn = xgb.train(params, dat_train, nrnd, watchlist, early_stopping_rounds=40, verbose_eval=3)

# Predict on Test Set
pred = xgb_trn.predict(xgb.DMatrix(tst_x))

pred_output = [i for i in pred]
id_output = [i for i in tst_meta.SK_ID_CURR]

pred_df = pd.DataFrame({'SK_ID_CURR': id_output , 'TARGET': pred_output})

# Write Submission File
######################################################################################
pred_df.to_csv('.../submission4.csv', index = False)


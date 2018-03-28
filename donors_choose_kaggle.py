"""
Donors Choose Kaggle Competition
https://www.kaggle.com/c/donorschoose-application-screening
"""

# Import Packages
######################################################################################
import numpy as np, pandas as pd, xgboost as xgb
from xgboost.sklearn import XGBClassifier
from scipy import sparse
from scipy.sparse import vstack
import sklearn
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.cross_validation import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import re, string
from datetime import timedelta

# Import & Format Data
######################################################################################
trn_dir = ".../train.csv"
tst_dir = ".../test.csv"
rsc_dir = ".../resources.csv"
my_stop_words = text.ENGLISH_STOP_WORDS

# Import Data & Separate Columns by Type
######################################################################################

def trn_tst_rsc_agg(train_dir, test_dir, join_dir):
    trn = pd.read_csv(train_dir)
    trn['cv_subset'] = 'train'
    tst = pd.read_csv(test_dir)
    tst['cv_subset'] = 'test'
    trntst = pd.concat([trn, tst])
    rsc = pd.read_csv(rsc_dir)
    rsc[['price', 'quantity']] = rsc[['price', 'quantity']].astype(float)
    rsc['price_quant'] = rsc['price'] * rsc['quantity']
    rsc_pq = pd.DataFrame(rsc[['id', 'price_quant']].groupby('id').price_quant.agg(['sum'])).reset_index()
    rsc_pq.rename(columns = {'sum':'pq_sum'}, inplace = True)
    rsc_quant = pd.DataFrame(rsc[['id', 'quantity']].groupby('id').quantity.agg(['sum'])).reset_index()
    rsc_quant.rename(columns = {'sum':'quantity'}, inplace = True)
    #rsc_quant = pd.DataFrame(rsc.groupby('id'),as_index=False).agg(lambda x : x.sum() if x.dtype=='float64' else ' '.join(x))
    #rsc_txt = pd.DataFrame(rsc[['id', 'description']].groupby('id')['description'].apply(lambda x: "{%s}" % ', '.join(x)))
    agg = trntst.join(rsc_pq.set_index('id'), on = 'id', how = 'left')
    agg_ii = agg.join(rsc_quant.set_index('id'), on = 'id', how = 'left')
    #agg_iii = agg.join(rsc_txt.set_index('id'), on = 'id', how = 'left')
    return agg_ii

dat = trn_tst_rsc_agg(trn_dir, tst_dir, rsc_dir)
dat = dat.fillna('thisismyNAfiller')

# Create Variable for Days Since Previous Submission  
######################################################################################
dat['subm_dt'] = dat['project_submitted_datetime'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
dat_tm = dat.sort_values(['teacher_id', 'subm_dt']).groupby('teacher_id')['subm_dt'].diff()
dat_tm = dat_tm.fillna(0)
dat_tm_days = dat_tm.dt.total_seconds() / (24 * 60 * 60)
dat['days_since_last_sub'] = dat_tm_days.values

tmp_list = []
for i in dat['days_since_last_sub']:
    if i > 0:
        tmp_list.append(1)
    else:
        tmp_list.append(0)

dat['last_sub_exists'] = tmp_list

# Create Variable for Days Since Last Submission
######################################################################################
dat['subm_dt'] = dat['project_submitted_datetime'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
dat_tm = dat.sort_values(['teacher_id', 'subm_dt']).groupby('teacher_id')['subm_dt'].diff()
dat_tm = dat_tm.fillna(0)
dat_tm_days = dat_tm.dt.total_seconds() / (24 * 60 * 60)
dat['days_since_last_sub'] = dat_tm_days.values

# Group Variables by Transformation Type
######################################################################################
dat_labels = dat[['id','teacher_id']]
dat_subsets = pd.DataFrame({'cv_subset': dat['cv_subset']})
dat_y = pd.DataFrame({'project_is_approved' : dat['project_is_approved']})
dat_text = pd.DataFrame({'title': dat['project_title'],
                         'essays': dat['project_essay_1'].map(str) + dat['project_essay_2'].map(str) + dat['project_essay_3'].map(str) + dat['project_essay_4'].map(str),
                         'resource_summary': dat['project_resource_summary']})
dat_categ = dat[['teacher_prefix', 'school_state','project_grade_category', 'project_subject_categories']]
dat_dates = pd.DataFrame({'project_submitted_datetime' : dat['project_submitted_datetime']})
dat_contin = pd.DataFrame({'teacher_number_of_previously_posted_projects' : dat['teacher_number_of_previously_posted_projects'],
                           'resource_quantity': dat['quantity'],
                           'resource_price': dat['pq_sum'],
                           'days_since_last_sub': dat['days_since_last_sub']})

# Define Data Processing Functions
######################################################################################
# Text Fields
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def text_proc(dat, StopWords, max_freq_perc = 0.7, min_freq_perc = 0.01, max_vars = 5000, max_ngram = 1):
    vectoriser = TfidfVectorizer(ngram_range=(1, max_ngram),
                                 tokenizer=tokenize,
                                 min_df = min_freq_perc,
                                 max_df = max_freq_perc,
                                 max_features = max_vars,
                                 strip_accents='unicode',
                                 use_idf=1,
                                 smooth_idf=1,
                                 sublinear_tf=1,
                                 stop_words=set(StopWords))
    
    tmp_list = []
    for col in dat:
        dat_transf = vectoriser.fit_transform(dat[col].values.astype('U'))
        dat_sparse = sparse.coo_matrix(dat_transf)
        tmp_list.append(dat_sparse)
    return scipy.sparse.hstack(tmp_list)

# Categorical Fields
def categ_proc(dat):
    tmp_list = []
    for col in dat:
        tmp_list.append(sparse.coo_matrix(pd.get_dummies(dat[col], sparse = True, drop_first = True)))
    return scipy.sparse.hstack(tmp_list)

# Continuous Fields
def contin_proc(dat):
    scaler = StandardScaler()
    if len(dat.columns) > 1:
        tmp_list = []
        for col in dat:
            scaled_dat = scaler.fit(dat[col])
            tmp_list.append(sparse.coo_matrix.transpose(sparse.coo_matrix(scaled_dat), copy = False))
        return scipy.sparse.hstack(tmp_list)
    else:
        scaled_dat = scaler.fit_transform(dat)
        tmp_list = sparse.coo_matrix(scaled_dat)
    return tmp_list

def contin_proc(dat):
    scaler = StandardScaler()
    newdat = dat
    newdat[newdat.columns] = scaler.fit_transform(newdat[newdat.columns])
    x = sparse.coo_matrix(newdat)
    return x
    
# Date Field
def date_proc(dat, binary_event_dt, dt_format):
    event_list = []
    for d in dat:
        if d >= binary_event_dt:
            x = 1
        else:
            x = 0
        event_list.append(x)
    mth_list = []
    for d in dat:
        mth_list.append(datetime.datetime.strptime(d, dt_format).month)
    sparse_output = pd.DataFrame({'event': event_list, 'mth': mth_list})
    return sparse.coo_matrix(sparse_output)
    
# Apply Data Proc. Functions & Combine Features
######################################################################################
x_text = text_proc(dat_text, my_stop_words, max_vars = 10000, max_ngram = 2)
x_categ = categ_proc(dat_categ)
x_contin = contin_proc(dat_contin)
x_dt = date_proc(dat_dates['project_submitted_datetime'], "2010-02-18", "%Y-%m-%d %H:%M:%S")
features = scipy.sparse.hstack([x_text, x_categ, x_contin, x_dt])

# Separate Train & Test Subsets
######################################################################################
n = 0
train_index = []
test_index = []
for i in dat_subsets['cv_subset']:
    if i == 'train':
        train_index.append(n)
    else:
        test_index.append(n)
    n = n + 1

train_x = features.tocsc()[train_index,:]
test_x = features.tocsc()[test_index,:]
train_y = dat_y['project_is_approved'][:(max(train_index)+1)]
test_y = dat_y['project_is_approved'][(max(train_index)+1):]
test_labels = dat_labels['id'][(max(train_index)+1):]

# Validation Split
######################################################################################
train_y, val_y, train_x, val_x = train_test_split(train_y, train_x, test_size = .225, random_state = 3132018)

# Parameter Tuning
######################################################################################
# Parameters Attempted in Grid Search Cross Validation
params = {
 'max_depth': [10],
 'min_child_weight': [9,10,11],
 'colsample_bytree': [0.1,0.2,0.3],
 'colsample_bylevel': [0.3,0.4,0.5]
}

gsrch = GridSearchCV(estimator = XGBClassifier( 
        objective= 'binary:logistic', 
        seed = 3082018), 
    param_grid = params, 
    scoring = 'roc_auc',
    cv=4,
    verbose = 1)

gsrch.fit(train_x, train_y)
gsrch.best_score_
gsrch.best_params_
    
# Notes on Grid Search Results
"""

Grid Search 1:

params = {
 'max_depth': [4,6,8,10,12],
 'min_child_weight': [4,5,7,10,13],
 'colsample_bytree': [0.2,0.4,0.6,0.8],
 'colsample_bylevel': [0.2,0.4,0.6,0.8]
}

gsrch.best_params_
Out[114]: 
{'colsample_bylevel': 0.4,
 'colsample_bytree': 0.2,
 'max_depth': 12,
 'min_child_weight': 10}

Grid Search 2:
    
params = {
 'max_depth': [10],
 'min_child_weight': [9,10,11],
 'colsample_bytree': [0.1,0.2,0.3],
 'colsample_bylevel': [0.3,0.4,0.5]
}

gsrch.best_params_
Out[118]: 
{'colsample_bylevel': 0.3,
 'colsample_bytree': 0.3,
 'max_depth': 10,
 'min_child_weight': 11}

"""

# Fitting Final Model
######################################################################################
# Parameters
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'auc'
params['eta'] = 0.01
params['max_depth'] = 9
params['min_child_weight'] = 9
params['colsample_bytree'] = 0.2
params['colsample_bylevel'] = 0.4
params['subsample'] = 1
nrnd = 4500

dat_train = xgb.DMatrix(train_x, label=train_y)
dat_valid = xgb.DMatrix(val_x, label=val_y)
dat_test = xgb.DMatrix(test_x, label = test_y)
watchlist = [(dat_train, 'train'), (dat_valid, 'valid')]
xgb_trn = xgb.train(params, dat_train, nrnd, watchlist, early_stopping_rounds=45, verbose_eval=1)

# Predict on Test Set
pred = xgb_trn.predict(xgb.DMatrix(test_x))
pred_df = pd.DataFrame({'id': test_labels, 'project_is_approved': pred})

# Write Submission File
######################################################################################
pred_df.to_csv('C:/Users/user/Desktop/kaggle_data/donors/submission4.csv', index = False)

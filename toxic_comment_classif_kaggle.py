# Import Packages
####################################################################
import numpy as np, pandas as pd
from random import shuffle
from tqdm import tqdm
from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, re, pprint, re, string, sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import xgboost as xgb

# Import & Format Data
######################################################################################
# Import Data
train =  pd.read_csv(".../train.csv")
test =  pd.read_csv(".../test.csv")
imported_stopwords = set(stopwords.words('english'))

# Deal With Null Values
train = train.fillna("thisismyplaceholderforNAvaluesthatIdontthinkwillshowupotherwise")
test = test.fillna("thisismyplaceholderforNAvaluesthatIdontthinkwillshowupotherwise")

# Tokenize Function
comp = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def get_token(s): return comp.sub(r' \1 ', s).split()
tfid = TfidfVectorizer(ngram_range=(1,2),
                       min_df=5,
                       max_df = 0.8,
                       max_features = 20000,
                       strip_accents='unicode',
                       use_idf=True,
                       smooth_idf=True,
                       sublinear_tf=True,
                       tokenizer=get_token,
                       stop_words = imported_stopwords).fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))

# Split Into Train, Validation, and Test
offset = 10000
train_txt = tfid.transform(train['comment_text'])
test_txt = tfid.transform(test['comment_text'])

# Model Fitting & Projection
####################################################################
# Gradient Boosting Parameters
boostparams = {}
boostparams['objective'] = 'binary:logistic'
boostparams['eta'] = 0.12
boostparams['max_depth'] = 4
boostparams['silent'] = 1
boostparams['eval_metric'] = 'logloss'
boostparams['min_child_weight'] = 1
boostparams['subsample'] = 0.65
boostparams['colsample_bytree'] = 0.8
plst = list(boostparams.items()) 
num_round = 400

# Fit Model for Toxic
toxic_train = xgb.DMatrix(train_txt[offset:,:], label = train['toxic'][offset:])
toxic_validation = xgb.DMatrix(train_txt[:offset,:], label = train['toxic'][:offset])
toxic_watchlist = [(toxic_train, 'train'),(toxic_validation, 'val')]
toxic_test = xgb.DMatrix(test_txt)
toxic_fit = xgb.train(plst, toxic_train, num_round, toxic_watchlist, early_stopping_rounds = 20)
toxic_ypred = toxic_fit.predict(toxic_test)


# Gradient Boosting Parameters
boostparams = {}
boostparams['objective'] = 'binary:logistic'
boostparams['eta'] = 0.03
boostparams['max_depth'] = 5
boostparams['silent'] = 1
boostparams['eval_metric'] = 'logloss'
boostparams['min_child_weight'] = 1
boostparams['subsample'] = 0.65
boostparams['colsample_bytree'] = 0.8
plst = list(boostparams.items()) 
num_round = 3000

# Fit Model for Severe Toxic
severe_toxic_train = xgb.DMatrix(train_txt[offset:,:], label = train['severe_toxic'][offset:])
severe_toxic_validation = xgb.DMatrix(train_txt[:offset,:], label = train['severe_toxic'][:offset])
severe_toxic_watchlist = [(severe_toxic_train, 'train'),(severe_toxic_validation, 'val')]
severe_toxic_test = xgb.DMatrix(test_txt)
severe_toxic_fit = xgb.train(plst, severe_toxic_train, num_round, severe_toxic_watchlist, early_stopping_rounds = 15)
severe_toxic_ypred = severe_toxic_fit.predict(severe_toxic_test)

# Fit Model for Obscene
obscene_train = xgb.DMatrix(train_txt[offset:,:], label = train['obscene'][offset:])
obscene_validation = xgb.DMatrix(train_txt[:offset,:], label = train['obscene'][:offset])
obscene_watchlist = [(obscene_train, 'train'),(obscene_validation, 'val')]
obscene_test = xgb.DMatrix(test_txt)
obscene_fit = xgb.train(plst, obscene_train, num_round, obscene_watchlist, early_stopping_rounds = 15)
obscene_ypred = obscene_fit.predict(obscene_test)

# Fit Model for Threat
threat_train = xgb.DMatrix(train_txt[offset:,:], label = train['threat'][offset:])
threat_validation = xgb.DMatrix(train_txt[:offset,:], label = train['threat'][:offset])
threat_watchlist = [(threat_train, 'train'),(threat_validation, 'val')]
threat_test = xgb.DMatrix(test_txt)
threat_fit = xgb.train(plst, threat_train, num_round, threat_watchlist, early_stopping_rounds = 15)
threat_ypred = threat_fit.predict(threat_test)

# Fit Model for Insult
insult_train = xgb.DMatrix(train_txt[offset:,:], label = train['insult'][offset:])
insult_validation = xgb.DMatrix(train_txt[:offset,:], label = train['insult'][:offset])
insult_watchlist = [(insult_train, 'train'),(insult_validation, 'val')]
insult_test = xgb.DMatrix(test_txt)
insult_fit = xgb.train(plst, insult_train, num_round, insult_watchlist, early_stopping_rounds = 15)
insult_ypred = insult_fit.predict(insult_test)

# Fit Model for Identity Hate
identity_hate_train = xgb.DMatrix(train_txt[offset:,:], label = train['identity_hate'][offset:])
identity_hate_validation = xgb.DMatrix(train_txt[:offset,:], label = train['identity_hate'][:offset])
identity_hate_watchlist = [(identity_hate_train, 'train'),(identity_hate_validation, 'val')]
identity_hate_test = xgb.DMatrix(test_txt)
identity_hate_fit = xgb.train(plst, identity_hate_train, num_round, identity_hate_watchlist, early_stopping_rounds = 15)
identity_hate_ypred = identity_hate_fit.predict(identity_hate_test)

# Combine Predictions into Submission File
submission = pd.DataFrame({'id': test['id'],
                           'toxic': toxic_ypred,
                           'severe_toxic': severe_toxic_ypred,
                           'obscene': obscene_ypred,
                           'threat': threat_ypred,
                           'insult': insult_ypred,
                           'identity_hate': identity_hate_ypred})

col_order = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission = submission[col_order]
submission.to_csv("C:/Users/user/Desktop/py_scripts/kaggle/profanity/pysubmission3.csv", index = False)




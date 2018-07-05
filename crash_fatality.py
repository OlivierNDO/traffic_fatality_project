# Import Packages
#########################################################################################################################
import pandas as pd, numpy as np
import os, time
from datetime import datetime
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib import pyplot
from scipy import sparse
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix,roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance

# Folder Directories
#########################################################################################################################
dirs = ['.../extract_20170601-20170801Texas',
        '.../extract_20170802-20170930Texas',
        '.../extract_20171001-20171124Texas',
        '.../extract_20171125-20180123Texas',
        '.../extract_20180124-20180323Texas',
        '.../extract_20180324-20180519Texas',
        '.../extract_20180520-20180531Texas']

# Important and Concatenate Text Files
#########################################################################################################################
def read_crash_csvs(dir_list):
    charges_list = []
    crash_list = []
    damages_list = []
    endorsements_list = []
    lookup_list = []
    person_list = []
    primaryperson_list = []
    restrictions_list = []
    unit_list = []
    for direc in dir_list:
        direc_files = os.listdir(direc)
        for file in direc_files:
            if file.split('_')[4] == 'charges':
                charges_list.append(pd.read_csv(direc + '/' + file))
            elif file.split('_')[4] == 'crash':
                crash_list.append(pd.read_csv(direc + '/' + file))
            elif file.split('_')[4] == 'damages':
                damages_list.append(pd.read_csv(direc + '/' + file))
            elif file.split('_')[4] == 'endorsements':
                endorsements_list.append(pd.read_csv(direc + '/' + file))
            elif file.split('_')[4] == 'lookup':
                lookup_list.append(pd.read_csv(direc + '/' + file))
            elif file.split('_')[4] == 'person':
                person_list.append(pd.read_csv(direc + '/' + file))
            elif file.split('_')[4] == 'primaryperson':
                primaryperson_list.append(pd.read_csv(direc + '/' + file))
            elif file.split('_')[4] == 'restrictions':
                restrictions_list.append(pd.read_csv(direc + '/' + file))
            else:
                unit_list.append(pd.read_csv(direc + '/' + file))
    charges_df = pd.concat(charges_list)
    crash_df = pd.concat(crash_list)
    damages_df = pd.concat(damages_list)
    endorsements_df = pd.concat(endorsements_list)
    lookup_df = pd.concat(lookup_list)
    person_df = pd.concat(person_list)
    primaryperson_df = pd.concat(primaryperson_list)
    restrictions_df = pd.concat(restrictions_list)
    unit_df = pd.concat(unit_list)
    return charges_df,\
    crash_df,\
    damages_df,\
    endorsements_df,\
    lookup_df,\
    person_df,\
    primaryperson_df,\
    restrictions_df,\
    unit_df
    
charges_df, crash_df, damages_df, endorsements_df, lookup_df, person_df, primaryperson_df, restrictions_df, unit_df = read_crash_csvs(dirs)

# Define Data Manipulation & Exploration Functions
#########################################################################################################################    
def pandas_col_summary(dat):
    colname_list = []
    type_list = []
    num_uniq_list = []
    perc_uniq_list = []
    num_nan_list = []
    perc_nan_list = []
    first_five_list = []
    for col in dat.columns:
        colname_list.append(col)
        if dat[col].dtype == 'O':
            type_list.append('String')
        elif dat[col].dtype == 'f':
            type_list.append('Float')
        else:
            type_list.append('Integer')
        num_uniq_list.append(dat[col].agg('nunique'))
        perc_uniq_list.append(np.round(dat[col].agg('nunique') / len(dat[col]),3))
        num_nan_list.append(sum(pd.isnull(dat[col])))
        perc_nan_list.append(np.round(sum(pd.isnull(dat[col])) / len(dat[col]),3))
        first_five_list.append(' | '.join(str(dat[col][0:5])))
    output = pd.DataFrame({'Field_Name': colname_list,
                           'Value_Type': type_list,
                           'Num_Unique_Vals': num_uniq_list,
                           'Perc_Unique_Vals': perc_uniq_list,
                           'Num_NAN_Vals': num_nan_list,
                           'Perc_NAN_Vals': perc_nan_list,
                           'First_Five_Vals': first_five_list})
    return output

def nan_del_cols(dat_summary, label_excl = ['Crash_ID'], nan_threshold = .5, uniq_perc_thres = 0.5):
    delete_list = []
    reason_list = []
    for index, row in dat_summary.iterrows():
        if row['Field_Name'] in label_excl:
            pass
        else:
            if row['Perc_NAN_Vals'] > nan_threshold:
                delete_list.append(row['Field_Name'])
                reason_list.append('more than ' + str(np.round(nan_threshold * 100,0)) + '% of values missing')
            elif (row['Value_Type'] == 'String') and (row['Perc_Unique_Vals'] > uniq_perc_thres):
                delete_list.append(row['Field_Name'])
                reason_list.append('uninformative string values')
            else:
                pass
    output_df = pd.DataFrame({'Variable': delete_list,
                              'Deletion_Reason': reason_list})
    return output_df

def remove_del_cols(dat, del_col_dat):
    delete_cols = [i for i in del_col_dat['Variable']]
    output_df = dat.drop(delete_cols, axis = 1, inplace = False)
    return output_df

def remove_del_list_cols(dat, del_list):
    return dat.drop(del_list, axis = 1, inplace = False)

def float_time_of_day(time_vals):
    hr_list = []
    min_list = []
    am_pm_list = []
    for t in time_vals:
        hour = t[0:2]
        minute = t[3:5]
        am_pm = t[6:8]
        hr_list.append(hour)
        min_list.append(minute)
        am_pm_list.append(am_pm)
    output_df = pd.DataFrame({'Hour': hr_list,
                              'Minute': min_list,
                              'AM_PM': am_pm_list})
    temp_list = []
    for index, row in output_df.iterrows():
        hr_num = np.float64(row['Hour']) * 60
        min_num = np.float64(row['Minute'])
        if row['AM_PM'] == 'PM':
            temp_list.append((hr_num + min_num)*2)
        else:
            temp_list.append((hr_num + min_num))
    result = pd.DataFrame({'Time_Of_Day': temp_list})
    return result

def missing_contin_vals(dat, col):
    contin_vals = []
    dummy_missing = []
    for val in dat[col]:
        if (math.isnan(val)) or (val == ''):
            dummy_missing.append(1)
            contin_vals.append(0)
        else:
            dummy_missing.append(0)
            contin_vals.append(val)
    output_df = pd.DataFrame({col: contin_vals,
                              col+'_missing': dummy_missing})
    return output_df
            
def dummy_trans(dat, dummy_cols):
    temp_list = []
    for col in dummy_cols:
        temp_list.append(pd.get_dummies(dat[col], prefix = col, drop_first = True))
    output_df = pd.concat(temp_list, axis = 1)
    return output_df

def dv_string_trans(dat, dv_col, pos_string):
    temp_list = []
    for val in dat[dv_col]:
        if val == pos_string:
            temp_list.append(1)
        else:
            temp_list.append(0)
    output_df = pd.DataFrame({dv_col: temp_list})
    return output_df

def crash_df_transform(dat, lab_vars, categ_vars, tm_vars, dv = 'Crash_Fatal_Fl'):
    dummy_df = dummy_trans(dat, categ_vars)
    time_df = float_time_of_day(dat['Crash_Time'])
    age_df = missing_contin_vals(dat, 'Prsn_Age')
    speed_lim_df = dat['Crash_Speed_Limit']
    dv_df = dv_string_trans(dat, dv, 'Y')
    lab_df = dat[lab_vars]
    output_df = pd.concat([lab_df,
                           dv_df,
                           speed_lim_df,
                           time_df,
                           age_df,
                           dummy_df], axis = 1)
    return output_df
           
# Execute Initial Data Manipulation Functions
#########################################################################################################################
crash_col_summ = pandas_col_summary(crash_df)
delete_cols = nan_del_cols(crash_col_summ)
crash_df_ii = remove_del_cols(crash_df, delete_cols)
crash_col_summ_ii = pandas_col_summary(crash_df_ii)

# Manual Variable Deletions
#########################################################################################################################

# remove backward-looking variables
backward_looking_vars = ['Thousand_Damage_Fl', 'Investigat_Comp_Fl', 'Txdot_Rptable_Fl', 'Located_Fl',
                         'Crash_Sev_ID', 'Death_Cnt', 'Sus_Serious_Injry_Cnt', 'Harm_Evnt_ID',
                         'Nonincap_Injry_Cnt', 'Traffic_Cntl_ID', 'Poss_Injry_Cnt', 'Tot_Injry_Cnt',
                         'Unkn_Injry_Cnt', 'Non_Injry_Cnt', 'Investigat_Notify_Time', 'Investigat_Arrv_Time']

# remove non-generalizable variables
non_genrl_vars = ['Hwy_Sys', 'Rpt_Street_Sfx', 'Rpt_Sec_Street_Sfx', 'Rpt_Rdwy_Sys_ID', 'Rpt_Sec_Rdwy_Sys_ID',
                  'MPO_ID', 'Rpt_CRIS_Cnty_ID', 'Cnty_ID', 'Investigat_Service_ID', 'Crash_Date',
                  'Investigat_Region_ID', 'Report_Date', 'Investigat_Agency_ID', 'ORI_Number',
                  'City_ID', 'Rpt_City_ID', 'Hwy_Nbr', 'Investigat_Notify_Meth', 'Rpt_Sec_Block_Num',
                  'Rpt_Block_Num', 'Rpt_Street_Name', 'Street_Name', 'Rpt_Sec_Street_Name',
                  'Latitude', 'Longitude', 'Medical_Advisory_Fl', 'Amend_Supp_Fl']

crash_df_iii = remove_del_list_cols(crash_df_ii, backward_looking_vars + non_genrl_vars)

# Append Variables from Re: Persons Involved in Crash
#########################################################################################################################
crash_df_iv = pd.merge(crash_df_iii, person_df[['Crash_ID','Prsn_Nbr', 'Prsn_Occpnt_Pos_ID', 'Prsn_Age', 'Prsn_Ethnicity_ID',
                                                'Prsn_Gndr_ID', 'Prsn_Rest_ID', 'Prsn_Airbag_ID', 'Prsn_Helmet_ID',
                                                'Prsn_Alc_Rslt_ID', 'Prsn_Bac_Test_Rslt', 'Prsn_Drg_Rslt_ID']],
    on = 'Crash_ID',
    how = 'left')

# Categoization & Treatment of Remaining Variables
#########################################################################################################################
# List of Variables by Type
label_vars = ['Crash_ID', 'Prsn_Nbr']
dummy_code_vars = ['Schl_Bus_Fl', 'Rr_Relat_Fl', 'Active_School_Zone_Fl',
                   'Rpt_Outside_City_Limit_Fl', 'Private_Dr_Fl', 'Toll_Road_Fl', 'Road_Constr_Zone_Fl',
                   'Road_Constr_Zone_Wrkr_Fl', 'At_Intrsct_Fl', 'Onsys_Fl', 'Rural_Fl', 'Intrsct_Relat_ID',
                   'Rpt_Road_Part_ID', 'Road_Part_Adj_ID', 'Road_Relat_ID', 'Rpt_Sec_Road_Part_ID',
                   'Day_of_Week', 'Light_Cond_ID', 'Road_Algn_ID', 'Bridge_Detail_ID', 'Road_Cls_ID',
                   'Entr_Road_ID', 'Surf_Cond_ID', 'Pop_Group_ID', 'Wthr_Cond_ID', 'Phys_Featr_2_ID',
                   'Phys_Featr_1_ID', 'FHE_Collsn_ID', 'Obj_Struck_ID', 'Othr_Factr_ID', 'Prsn_Occpnt_Pos_ID',
                   'Prsn_Rest_ID',  'Prsn_Airbag_ID', 'Prsn_Helmet_ID', 'Prsn_Alc_Rslt_ID', 'Prsn_Bac_Test_Rslt',
                   'Prsn_Drg_Rslt_ID', 'Prsn_Ethnicity_ID', 'Prsn_Gndr_ID']
contin_vars = ['Prsn_Age', 'Crash_Speed_Limit']
time_vars = ['Crash_Time']

crash_df_v = crash_df_transform(crash_df_iv, label_vars, dummy_code_vars, time_vars)
crash_df_v['Prsn_Nbr'].fillna(1, inplace = True)
crash_df_v_summ = pandas_col_summary(crash_df_v)


# Aggregate from Person Level to Crash ID Level
#########################################################################################################################

person_cols = ['Crash_ID', 'Prsn_Nbr', 'Prsn_Occpnt_Pos_ID_2.0', 'Prsn_Occpnt_Pos_ID_3.0', 'Prsn_Occpnt_Pos_ID_4.0',
               'Prsn_Occpnt_Pos_ID_5.0', 'Prsn_Occpnt_Pos_ID_6.0', 'Prsn_Occpnt_Pos_ID_7.0', 'Prsn_Occpnt_Pos_ID_8.0',
               'Prsn_Occpnt_Pos_ID_9.0', 'Prsn_Occpnt_Pos_ID_10.0', 'Prsn_Occpnt_Pos_ID_11.0', 'Prsn_Occpnt_Pos_ID_12.0',
               'Prsn_Occpnt_Pos_ID_13.0', 'Prsn_Occpnt_Pos_ID_14.0', 'Prsn_Occpnt_Pos_ID_16.0', 'Prsn_Occpnt_Pos_ID_98.0',
               'Prsn_Rest_ID_2.0', 'Prsn_Rest_ID_3.0', 'Prsn_Rest_ID_4.0', 'Prsn_Rest_ID_5.0', 'Prsn_Rest_ID_6.0',
               'Prsn_Rest_ID_7.0', 'Prsn_Rest_ID_8.0', 'Prsn_Rest_ID_9.0', 'Prsn_Rest_ID_10.0', 'Prsn_Rest_ID_11.0',
               'Prsn_Airbag_ID_1.0', 'Prsn_Airbag_ID_2.0', 'Prsn_Airbag_ID_3.0', 'Prsn_Airbag_ID_4.0', 'Prsn_Airbag_ID_7.0',
               'Prsn_Airbag_ID_8.0', 'Prsn_Helmet_ID_2.0', 'Prsn_Helmet_ID_3.0', 'Prsn_Helmet_ID_4.0', 'Prsn_Helmet_ID_5.0',
               'Prsn_Helmet_ID_97.0', 'Prsn_Ethnicity_ID_1.0', 'Prsn_Ethnicity_ID_2.0', 'Prsn_Ethnicity_ID_3.0',
               'Prsn_Ethnicity_ID_4.0', 'Prsn_Ethnicity_ID_5.0', 'Prsn_Ethnicity_ID_6.0', 'Prsn_Gndr_ID_1.0',
               'Prsn_Gndr_ID_2.0', 'Prsn_Age', 'Prsn_Age_missing']

nonperson_cols = [x for x in crash_df_v.columns if x not in person_cols] + ['Crash_ID']
person_df = crash_df_v[person_cols].groupby('Crash_ID', as_index = False).max()
nonperson_df = crash_df_v[nonperson_cols].drop_duplicates(inplace = False)
crash_df_vi = pd.merge(nonperson_df, person_df, on = 'Crash_ID')

# Model Fitting
#########################################################################################################################
crash_df_vi = shuffle(crash_df_vi)
y = crash_df_vi.iloc[:,1]
x = crash_df_vi.iloc[:,2:319]

# Validation Split
######################################################################################
trn_y, tst_y, trn_x, tst_x = train_test_split(y, x, test_size = .3, random_state = 7042018)
trn_y, val_y, trn_x, val_x = train_test_split(trn_y, trn_x, test_size = .2, random_state = 7042018)

# Fitting Final Model
######################################################################################
yvar_imbalance = trn_y.count() / trn_y.sum()
# Parameters
params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'gbtree'
params['eval_metric'] = 'logloss'
params['eta'] = 0.025
params['max_depth'] = 6
params['min_child_weight'] = 12
params['colsample_bytree'] = 0.65
params['colsample_bylevel'] = 0.25
params['subsample'] = 0.8
params['reg_lambda'] = .05
params['reg_alpha'] = 0
params['min_split_loss'] = 2
params['scale_pos_weight'] = yvar_imbalance
nrnd = 1000 # n_estimators

dat_train = xgb.DMatrix(trn_x, label=trn_y)
dat_valid = xgb.DMatrix(val_x, label=val_y)
watchlist = [(dat_train, 'train'), (dat_valid, 'valid')]
xgb_trn = xgb.train(params, dat_train, nrnd, watchlist, early_stopping_rounds=40, verbose_eval=1)


# Predict on Test Set
pred = xgb_trn.predict(xgb.DMatrix(tst_x))
pred_df = pd.DataFrame({'Predicted': pred, 'Actual': tst_y})

# Out of Sample Performance
AUC = roc_auc_score(tst_y, pred)
tn, fp, fn, tp = confusion_matrix(tst_y, np.round(pred,0)).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print('\n\n        Out of Sample Performance: \n \n AUC: ' \
      + str(np.round(AUC, 4)) \
      + '\n sensitivity (true positive): ' \
      + str(np.round(sensitivity, 4)) \
      + '\n specificity (true negative): ' \
      + str(np.round(specificity, 4)))

"""
        Out of Sample Performance: 
 
 AUC: 0.8852
 sensitivity (true positive): 0.7171
 specificity (true negative): 0.8693
"""

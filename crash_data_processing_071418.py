# Import Packages
#########################################################################################################################
import pandas as pd, numpy as np
import pyodbc
from sqlalchemy import create_engine, MetaData, Table, select
import os, time
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import *

# Folder Directories
#########################################################################################################################
dirs = ['C:/Users/user/Desktop/crash_iii/extract_20141107-20141231Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20140908-20141106Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20140701-20140907Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20140515-20140630Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20140311-20140514Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20140101-20140310Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20131116-20131231Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20130912-20131115Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20130701-20130911Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20130522-20130630Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20130315-20130521Texas',
        'C:/Users/user/Desktop/crash_iii/extract_20130101-20130314Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20170601-20170801Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20170802-20170930Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20171001-20171124Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20171125-20180123Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20180124-20180323Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20180324-20180519Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20180520-20180531Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20170101-20170302Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20170303-20170427Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20170428-20170531Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20160926-20161119Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20161120-20161231Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20160731-20160925Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20160601-20160730Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20160426-20160531Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20160301-20160425Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20160101-20160229Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20151225-20151231Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20151030-20151224Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20150902-20151029Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20150701-20150901Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20150505-20150630Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20150305-20150504Texas',
        'C:/Users/user/Desktop/crash_ii/extract_20150101-20150304Texas']

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

#charges_summary = pandas_col_summary(charges_df)
#crashes_summary = pandas_col_summary(crash_df)
#damages_summary = pandas_col_summary(damages_df)
#lookup_summary = pandas_col_summary(lookup_df)
#person_summary = pandas_col_summary(person_df)
#primaryperson_summary = pandas_col_summary(primaryperson_df)
#restrictions_summary = pandas_col_summary(restrictions_df)
#unit_summary = pandas_col_summary(unit_df)

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

# Manual Feature Transformation of Individual Fields
#########################################################################################################################    
# Replace Prsn_Type_ID Variable - Combine 'Unknown' / 'Invalid' / etc.
def prsn_type_id_repl(prsn_type_id_col):
    temp_list = []
    for prsn in prsn_type_id_col:
        if prsn not in [1,2,3,4,5,6]:
            temp_list.append(94)
        else:
            temp_list.append(prsn)
    output_df = pd.DataFrame({'Prsn_Type_ID': temp_list})
    return output_df
    
# Replace Prsn_Occpnt_Pos_ID Variable - Combine 'Unknown' / 'Invalid' / etc.
def prsn_occpnt_pos_id_repl(prsn_occ_pos_col):
    temp_list = []
    for prsn in prsn_occ_pos_col:
        if prsn not in [1,2,3,4,5,6,7,8,9,10,11,13,14,16]:
            temp_list.append(94)
        else:
            temp_list.append(prsn)
    output_df = pd.DataFrame({'Prsn_Occpnt_Pos_ID': temp_list})
    return output_df

# Replace DFO ('distance of crash from highway') with itself + a missingness [0,1] variable
def dfo_repl(dfo_col):
    num_list = []
    nan_dummy_list = []
    for d in dfo_col:
        if pd.isnull(d):
            num_list.append(0)
            nan_dummy_list.append(1)
        else:
            num_list.append(d)
            nan_dummy_list.append(0)
    output_df = pd.DataFrame({'Dfo': num_list,
                              'Dfo_Miss': nan_dummy_list})
    return output_df
  
# Replace Trk_Aadt_Pct ('adj. avg. daily traff perc for trucks on highway') with itself + a missingness [0,1] variable
def trk_aadt_perc_repl(trk_aadt_col):
    num_list = []
    nan_dummy_list = []
    for d in trk_aadt_col:
        if pd.isnull(d):
            num_list.append(0)
            nan_dummy_list.append(1)
        else:
            num_list.append(d)
            nan_dummy_list.append(0)
    output_df = pd.DataFrame({'Trk_Aadt_Pct': num_list,
                              'Trk_Aadt_Pct_Miss': nan_dummy_list})
    return output_df

# Adt_Adj_Curnt_Amt
def adt_adj_curnt_amt_repl(aaca_col):
    aaca_list = []
    aaca_miss_list = []
    for p in aaca_col:
        if pd.isnull(p):
            aaca_list.append(0)
            aaca_miss_list.append(1)
        else:
            aaca_list.append(p)
            aaca_miss_list.append(0)
    output_df = pd.DataFrame({'Adt_Adj_Curnt_Amt': aaca_list,
                              'Adt_Adj_Curnt_Amt_Miss': aaca_miss_list})
    return output_df
            
# Pct_Combo_Trk_Adt
def pct_combo_trk_adt_repl(pcta_col):
    pcta_list = []
    pcta_miss_list = []
    for p in pcta_col:
        if pd.isnull(p):
            pcta_list.append(0)
            pcta_miss_list.append(1)
        else:
            pcta_list.append(p)
            pcta_miss_list.append(0)
    output_df = pd.DataFrame({'Pct_Combo_Trk_Adt': pcta_list,
                              'Pct_Combo_Trk_Adt_Miss': pcta_miss_list})
    return output_df

# Pct_Single_Trk_Adt
def pct_single_trk_adt_repl(psta_col):
    psta_list = []
    psta_miss_list = []
    for p in psta_col:
        if pd.isnull(p):
            psta_list.append(0)
            psta_miss_list.append(1)
        else:
            psta_list.append(p)
            psta_miss_list.append(0)
    output_df = pd.DataFrame({'Pct_Single_Trk_Adt': psta_list,
                              'Pct_Single_Trk_Adt_Miss': psta_miss_list})
    return output_df     

# Func_Sys_ID
def func_sys_id_repl(fsi_col):
    temp_list = []
    for f in fsi_col:
        if f not in [1,2,6,7,8,9,11,12,14,16,17,19]:
            temp_list.append(99)
        else:
            temp_list.append(f)
    output_df = pd.DataFrame({'Func_Sys_ID': temp_list})
    return output_df

# Rural_Urban_Type_ID
def rural_urban_type_id_repl(ruti_col):
    temp_list = []
    for r in ruti_col:
        if r not in [1,2,3,4]:
            temp_list.append(99)
        else:
            temp_list.append(r)
    output_df = pd.DataFrame({'Rural_Urban_Type_ID': temp_list})
    return output_df

# Median_Width
def median_width_repl(med_col):
    med_list = []
    med_miss_list = []
    for m in med_col:
        if pd.isnull(m):
            med_list.append(0)
            med_miss_list.append(1)
        else:
            med_list.append(m)
            med_miss_list.append(0)
    output_df = pd.DataFrame({'Median_Width': med_list,
                              'Median_Width_Miss': med_miss_list})
    return output_df

# Shldr_Use_Right_ID
def shldr_use_right_id_repl(suri_col):
    suri_list = []
    for s in suri_col:
        if s not in [0,1,2,3,4,5,6,7]:
            suri_list.append(99)
        else:
            suri_list.append(s)
    output_df = pd.DataFrame({'Shldr_Use_Right_ID': suri_list})
    return output_df

# Shldr_Use_Left_ID
def shldr_use_left_id_repl(suri_col):
    suri_list = []
    for s in suri_col:
        if s not in [0,1,2,3,4,5,6,7]:
            suri_list.append(99)
        else:
            suri_list.append(s)
    output_df = pd.DataFrame({'Shldr_Use_Left_ID': suri_list})
    return output_df

# Shldr_Type_Right_ID / Shldr_Type_Left_ID
def shldr_type_id_repl(shldr_col):
    temp_list = []
    for s in shldr_col:
        if s in [1,2,3,4,5]:
            temp_list.append(s)
        else:
            temp_list.append(99)
    colnm = shldr_col.name
    output_df = pd.DataFrame({colnm: temp_list})
    return output_df

# Row_Width_Usual
def row_width_usual_repl(rwu_col):
    rwu_list = []
    rwu_miss_list = []
    for r in rwu_col:
        if pd.isnull(r):
            rwu_list.append(0)
            rwu_miss_list.append(1)
        else:
            rwu_list.append(r)
            rwu_miss_list.append(0)
    output_df = pd.DataFrame({'Row_Width_Usual': rwu_list,
                              'Row_Width_Usual_Miss': rwu_miss_list})
    return output_df

# Road_Type_ID
def road_type_repl(rt_col):
    temp_list = []
    for r in rt_col:
        if r in [0,1,2,3]:
            temp_list.append(r)
        else:
            temp_list.append(99)
    output_df = pd.DataFrame({'Road_Type_ID': temp_list})
    return output_df

# Median_Type_ID
def median_type_repl(med_col):
    temp_list = []
    for m in med_col:
        if m not in [0,1,2,3,4]:
            temp_list.append(0)
        else:
            temp_list.append(m)
    output_df = pd.DataFrame({'Median_Type_ID': temp_list})
    return output_df

# Surf_Cond_ID
def surf_cond_repl(surf_col):
    temp_list = []
    for s in surf_col:
        if s in [1,2,3,4,5,6,9,10]:
            temp_list.append(s)
        else:
            temp_list.append(99)
    output_df = pd.DataFrame({'Surf_Cond_ID': temp_list})
    return output_df

# Surf_Type_ID
def surf_type_repl(surf_col):
    temp_list = []
    for s in surf_col:
        if s in [1,2,3,4,5,6]:
            temp_list.append(s)
        else:
            temp_list.append(99)
    output_df = pd.DataFrame({'Surf_Type_ID': temp_list})
    return output_df

# Nbr_Of_Lane
def nbr_of_lane_repl(lane_col):
    nol_list = []
    nol_miss_list = []
    for l in lane_col:
        if pd.isnull(l):
            nol_list.append(0)
            nol_miss_list.append(1)
        else:
            nol_list.append(l)
            nol_miss_list.append(0)
    output_df = pd.DataFrame({'Nbr_Of_Lane': nol_list,
                              'Nbr_Of_Lane_Miss': nol_miss_list})
    return output_df

# Shldr_Width_Left & Shldr_Width_Right
def shldr_width_repl(shldr_col):
    s_list = []
    s_miss_list = []
    for s in shldr_col:
        if pd.isnull(s):
            s_list.append(0)
            s_miss_list.append(1)
        else:
            s_list.append(s)
            s_miss_list.append(0)
    colnm = shldr_col.name
    colnm_ii = colnm + '_Miss'
    output_df = pd.DataFrame({colnm: s_list,
                              colnm_ii: s_miss_list})
    return output_df

# Curb_Type_Left_ID & Curb_Type_Right_ID
def curb_type_repl(curb_col):
    curb_list = []
    curb_miss_list = []
    for c in curb_col:
        if pd.isnull(c):
            curb_list.append(0)
            curb_miss_list.append(1)
        else:
            curb_list.append(c)
            curb_miss_list.append(0)
    colnm = curb_col.name
    colnm_ii = colnm + '_Miss'
    output_df = pd.DataFrame({colnm: curb_list,
                              colnm_ii: curb_miss_list})
    return output_df

# Surf_Width
def surf_width_repl(surf_col):
    surf_list = []
    surf_list_miss = []
    for s in surf_col:
        if pd.isnull(s):
            surf_list.append(0)
            surf_list_miss.append(1)
        else:
            surf_list.append(s)
            surf_list_miss.append(0)
    output_df = pd.DataFrame({'Surf_Width': surf_list,
                              'Surf_Width_Miss': surf_list_miss})
    return output_df

def roadbed_repl(rb_col):
    rb_list = []
    rb_miss_list = []
    for r in rb_col:
        if pd.isnull(r):
            rb_list.append(0)
            rb_miss_list.append(1)
        else:
            rb_list.append(r)
            rb_miss_list.append(0)
    output_df = pd.DataFrame({'Roadbed_Width': rb_list,
                             'Roadbed_Width_Miss': rb_miss_list})
    return output_df

def hp_med_repl(hp_col):
    hp_list = []
    hp_miss_list = []
    for h in hp_col:
        if pd.isnull(h):
            hp_list.append(0)
            hp_miss_list.append(1)
        else:
            hp_list.append(h)
            hp_miss_list.append(0)
    output_df = pd.DataFrame({'Hp_Median_Width': hp_list,
                              'Hp_Median_Width_Miss': hp_miss_list})
    return output_df

def hp_shldr_repl(hps_col):
    hp_list = []
    hp_miss_list = []
    for h in hps_col:
        if pd.isnull(h):
            hp_list.append(0)
            hp_miss_list.append(1)
        else:
            hp_list.append(h)
            hp_miss_list.append(0)
    colnm = hps_col.name
    colnm_ii = colnm + '_Miss'
    output_df = pd.DataFrame({colnm: hp_list,
                              colnm_ii: hp_miss_list})
    return output_df

def hwy_ln_repl(hl_col):
    hl_list = []
    for h in hl_col:
        if pd.isnull(h):
            hl_list.append(99)
        else:
            hl_list.append(h)
    output_df = pd.DataFrame({'Hwy_Dsgn_Lane_ID': hl_list})
    return output_df

def hwy_hi_repl(hl_col):
    hl_list = []
    for h in hl_col:
        if pd.isnull(h):
            hl_list.append(99)
        else:
            hl_list.append(h)
    output_df = pd.DataFrame({'Hwy_Dsgn_Hrt_ID': hl_list})
    return output_df

# Use 'damages' File to Extract 'hit' features (e.g. if 'pole' is listed in damages, car is likely to have hit a pole)
#########################################################################################################################    
def pandas_stemword_freq(panda_col):
    #stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    word_list = []
    for r in damages['Damaged_Property']:
        for s in str(r).split(' '):
            word_list.append(stemmer.stem(re.sub(r'[^\w\s]','',s)))
    #word_list = [w for w in word_list if w not in stop_words]
    word_count_df = pd.DataFrame.from_dict(Counter(word_list), orient='index').reset_index()
    word_count_df.columns = ['word', 'freq']
    word_count_df['freq_perc'] = word_count_df['freq'] / len(panda_col)
    word_count_df = word_count_df.sort_values(by='freq', ascending=False)
    return word_count_df

def damage_features(dmg_df, str_col = 'Damaged_Property', id_col = 'Crash_ID'):
    fence_list = []
    pole_list = []
    rail_list = []
    sign_list = []
    light_list = []
    conc_list = []
    wall_list = []
    tree_list = []
    metal_list = []
    brick_list = []
    wood_list = []
    elec_list = []
    teleph_list = []
    hydrant_list = []
    curb_list = []
    for s in dmg_df[str_col]:
        if ('FENC' in str(s)) or ('GATE' in str(s)) or ('POST' in str(s)) or ('UTIL' in str(s)):
            fence_list.append(1)
        else:
            fence_list.append(0)
        if ('POLE' in str(s)) or ('SIGNAL' in str(s)):
            pole_list.append(1)
        else:
            pole_list.append(0)
        if ('RAIL' in str(s)) or ('GUARD' in str(s)):
            rail_list.append(1)
        else:
            rail_list.append(0)
        if ('SIGN' in str(s)):
            sign_list.append(1)
        else:
            sign_list.append(0)
        if ('LIGHT' in str(s)):
            light_list.append(1)
        else:
            light_list.append(0)
        if ('CEMENT' in str(s)) or ('CONCRETE' in str(s)):
            conc_list.append(1)
        else:
            conc_list.append(0)
        if ('WALL' in str(s)) or ('BARRIER' in str(s)):
            wall_list.append(1)
        else:
            wall_list.append(0)
        if ('TREE' in str(s)):
            tree_list.append(1)
        else:
            tree_list.append(0)
        if ('METAL' in str(s)):
            metal_list.append(1)
        else:
            metal_list.append(0)
        if ('BRICK' in str(s)):
            brick_list.append(1)
        else:
            brick_list.append(0)
        if ('WOOD' in str(s)):
            wood_list.append(1)
        else:
            wood_list.append(0)
        if ('ELECTR' in str(s)):
            elec_list.append(1)
        else:
            elec_list.append(0)
        if ('TELEPHON' in str(s)):
            teleph_list.append(1)
        else:
            teleph_list.append(0)
        if ('HYDRANT' in str(s)):
            hydrant_list.append(1)
        else:
            hydrant_list.append(0)
        if ('CURB' in str(s)):
            curb_list.append(1)
        else:
            curb_list.append(0)
    output_df = pd.DataFrame({id_col: dmg_df[id_col],
                              'HIT_FENCE': fence_list,
                              'HIT_POLE': pole_list,
                              'HIT_RAIL': rail_list,
                              'HIT_SIGN': sign_list,
                              'HIT_LIGHT': light_list,
                              'HIT_CONC': conc_list,
                              'HIT_WALL': wall_list,
                              'HIT_TREE': tree_list,
                              'HIT_METAL': metal_list,
                              'HIT_BRICK': brick_list,
                              'HIT_WOOD': wood_list,
                              'HIT_ELEC': elec_list,
                              'HIT_TELEPH': teleph_list,
                              'HIT_HYDRANT': hydrant_list,
                              'HIT_CURB': curb_list})
    output_df = output_df.groupby([id_col]).max().reset_index()
    return output_df

# Define endorsement functions: i.e. did driver have a non-standard license type
#########################################################################################################################
def endorse_dummies(end_df, lic_col = 'Drvr_Lic_Endors_ID', id_col = 'Crash_ID'):
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    for i in end_df[lic_col]:
        if i == 2:
            list2.append(1); list3.append(0); list4.append(0);
            list5.append(0); list6.append(0); list7.append(0);
            list8.append(0)
        elif i == 3:
            list2.append(0); list3.append(1); list4.append(0);
            list5.append(0); list6.append(0); list7.append(0);
            list8.append(0)
        elif i == 4:
            list2.append(0); list3.append(0); list4.append(1);
            list5.append(0); list6.append(0); list7.append(0);
            list8.append(0)
        elif i == 5:
            list2.append(0); list3.append(0); list4.append(0);
            list5.append(1); list6.append(0); list7.append(0);
            list8.append(0)
        elif i == 6:
            list2.append(0); list3.append(0); list4.append(0);
            list5.append(0); list6.append(1); list7.append(0);
            list8.append(0)
        elif i == 7:
            list2.append(0); list3.append(0); list4.append(0);
            list5.append(0); list6.append(0); list7.append(1);
            list8.append(0)
        elif i == 0:
            list2.append(0); list3.append(0); list4.append(0);
            list5.append(0); list6.append(0); list7.append(0);
            list8.append(1)
        else:
            list2.append(0); list3.append(0); list4.append(0);
            list5.append(0); list6.append(0); list7.append(0);
            list8.append(0)
    output_df = pd.DataFrame({id_col: end_df[id_col],
                              'CDL2': list2,
                              'CDL3': list3,
                              'CDL4': list4,
                              'CDL5': list5,
                              'CDL6': list6,
                              'CDL7': list7,
                              'CDL8': list8})
    output_df = output_df.groupby([id_col], sort = False).max().reset_index()
    return output_df

# Execute Individual Column Processing Functions
#########################################################################################################################
Pct_Combo_Trk_Adt = pct_combo_trk_adt_repl(crash_df['Pct_Combo_Trk_Adt'])
Prsn_Type_ID = prsn_type_id_repl(person_df['Prsn_Type_ID'])
Prsn_Occpnt_Pos_ID = prsn_occpnt_pos_id_repl(person_df['Prsn_Occpnt_Pos_ID'])
DFO = dfo_repl(crash_df['Dfo'])
Trk_Aadt_Pct = trk_aadt_perc_repl(crash_df['Trk_Aadt_Pct'])
Pct_Combo_Trk_Adt = pct_combo_trk_adt_repl(crash_df['Pct_Combo_Trk_Adt'])
Pct_Single_Trk_Adt = pct_single_trk_adt_repl(crash_df['Pct_Single_Trk_Adt'])
Adt_Adj_Curnt_Amt = adt_adj_curnt_amt_repl(crash_df['Adt_Adj_Curnt_Amt'])
Func_Sys_ID = func_sys_id_repl(crash_df['Func_Sys_ID'])
Rural_Urban_Type_ID = rural_urban_type_id_repl(crash_df['Rural_Urban_Type_ID'])
Median_Width = median_width_repl(crash_df['Median_Width'])
Shldr_Use_Right_ID = shldr_use_right_id_repl(crash_df['Shldr_Use_Right_ID'])
Shldr_Use_Left_ID = shldr_use_left_id_repl(crash_df['Shldr_Use_Left_ID'])
Shldr_Type_Right_ID = shldr_type_id_repl(crash_df['Shldr_Type_Right_ID'])
Shldr_Type_Left_ID = shldr_type_id_repl(crash_df['Shldr_Type_Left_ID'])
Row_Width_Usual = row_width_usual_repl(crash_df['Row_Width_Usual'])
Road_Type_ID = road_type_repl(crash_df['Road_Type_ID'])
Median_Type_ID = median_type_repl(crash_df['Median_Type_ID'])
Surf_Cond_ID = surf_cond_repl(crash_df['Surf_Cond_ID'])
Surf_Type_ID = surf_type_repl(crash_df['Surf_Type_ID'])
Nbr_Of_Lane = nbr_of_lane_repl(crash_df['Nbr_Of_Lane'])
Shldr_Width_Left = shldr_width_repl(crash_df['Shldr_Width_Left'])
Shldr_Width_Right = shldr_width_repl(crash_df['Shldr_Width_Right'])
Curb_Type_Left_ID = curb_type_repl(crash_df['Curb_Type_Left_ID'])
Curb_Type_Right_ID = curb_type_repl(crash_df['Curb_Type_Right_ID'])
Surf_Width = surf_width_repl(crash_df['Surf_Width'])
Roadbed_Width = roadbed_repl(crash_df['Roadbed_Width'])
Hp_Median_Width = hp_med_repl(crash_df['Hp_Median_Width'])
Hp_Shldr_Left = hp_shldr_repl(crash_df['Hp_Shldr_Left'])
Hp_Shldr_Right = hp_shldr_repl(crash_df['Hp_Shldr_Right'])
Hwy_Dsgn_Lane_ID = hwy_ln_repl(crash_df['Hwy_Dsgn_Lane_ID'])
Hwy_Dsgn_Hrt_ID = hwy_hi_repl(crash_df['Hwy_Dsgn_Hrt_ID'])

crash_df['Pct_Combo_Trk_Adt'] = Pct_Combo_Trk_Adt['Pct_Combo_Trk_Adt']
crash_df['Pct_Combo_Trk_Adt_Miss'] = Pct_Combo_Trk_Adt['Pct_Combo_Trk_Adt_Miss']
person_df['Prsn_Type_ID'] = Prsn_Type_ID['Prsn_Type_ID']
person_df['Prsn_Occpnt_Pos_ID'] = Prsn_Occpnt_Pos_ID['Prsn_Occpnt_Pos_ID']
crash_df['Dfo'] = DFO['Dfo']
crash_df['Dfo_Miss'] = DFO['Dfo_Miss']
crash_df['Trk_Aadt_Pct'] = Trk_Aadt_Pct['Trk_Aadt_Pct']
crash_df['Trk_Aadt_Pct_Miss'] = Trk_Aadt_Pct['Trk_Aadt_Pct_Miss']
crash_df['Pct_Combo_Trk_Adt'] = Pct_Combo_Trk_Adt['Pct_Combo_Trk_Adt']
crash_df['Pct_Combo_Trk_Adt_Miss'] = Pct_Combo_Trk_Adt['Pct_Combo_Trk_Adt_Miss']
crash_df['Pct_Single_Trk_Adt'] = Pct_Single_Trk_Adt['Pct_Single_Trk_Adt']
crash_df['Pct_Single_Trk_Adt_Miss'] = Pct_Single_Trk_Adt['Pct_Single_Trk_Adt_Miss']
crash_df['Adt_Adj_Curnt_Amt'] = Adt_Adj_Curnt_Amt['Adt_Adj_Curnt_Amt']
crash_df['Adt_Adj_Curnt_Amt_Miss'] = Adt_Adj_Curnt_Amt['Adt_Adj_Curnt_Amt_Miss']
crash_df['Func_Sys_ID'] = Func_Sys_ID['Func_Sys_ID']
crash_df['Rural_Urban_Type_ID'] = Rural_Urban_Type_ID['Rural_Urban_Type_ID']
crash_df['Median_Width'] = Median_Width['Median_Width']
crash_df['Median_Width_Miss'] = Median_Width['Median_Width_Miss']
crash_df['Shldr_Use_Right_ID'] = Shldr_Use_Right_ID['Shldr_Use_Right_ID']
crash_df['Shldr_Use_Left_ID'] = Shldr_Use_Left_ID['Shldr_Use_Left_ID']
crash_df['Shldr_Type_Right_ID'] = Shldr_Type_Right_ID['Shldr_Type_Right_ID']
crash_df['Shldr_Type_Left_ID'] = Shldr_Type_Left_ID['Shldr_Type_Left_ID']
crash_df['Row_Width_Usual'] = Row_Width_Usual['Row_Width_Usual']
crash_df['Row_Width_Usual_Miss'] = Row_Width_Usual['Row_Width_Usual_Miss']
crash_df['Road_Type_ID'] = Road_Type_ID['Road_Type_ID']
crash_df['Median_Type_ID'] = Median_Type_ID['Median_Type_ID']
crash_df['Surf_Cond_ID'] = Surf_Cond_ID['Surf_Cond_ID']
crash_df['Surf_Type_ID'] = Surf_Type_ID['Surf_Type_ID']
crash_df['Nbr_Of_Lane'] = Nbr_Of_Lane['Nbr_Of_Lane']
crash_df['Nbr_Of_Lane_Miss'] = Nbr_Of_Lane['Nbr_Of_Lane_Miss']
crash_df['Shldr_Width_Left'] = Shldr_Width_Left['Shldr_Width_Left']
crash_df['Shldr_Width_Left_Miss'] = Shldr_Width_Left['Shldr_Width_Left_Miss']
crash_df['Shldr_Width_Right'] = Shldr_Width_Right['Shldr_Width_Right']
crash_df['Shldr_Width_Right_Miss'] = Shldr_Width_Right['Shldr_Width_Right_Miss']
crash_df['Curb_Type_Left_ID'] = Curb_Type_Left_ID['Curb_Type_Left_ID']
crash_df['Curb_Type_Left_ID_Miss'] = Curb_Type_Left_ID['Curb_Type_Left_ID_Miss']
crash_df['Curb_Type_Right_ID'] = Curb_Type_Right_ID['Curb_Type_Right_ID']
crash_df['Curb_Type_Right_ID_Miss'] = Curb_Type_Right_ID['Curb_Type_Right_ID_Miss']
crash_df['Surf_Width'] = Surf_Width['Surf_Width']
crash_df['Surf_Width_Miss'] = Surf_Width['Surf_Width_Miss']
crash_df['Roadbed_Width'] = Roadbed_Width['Roadbed_Width']
crash_df['Roadbed_Width_Miss'] = Roadbed_Width['Roadbed_Width_Miss']
crash_df['Hp_Median_Width'] = Hp_Median_Width['Hp_Median_Width']
crash_df['Hp_Median_Width_Miss'] = Hp_Median_Width['Hp_Median_Width_Miss']
crash_df['Hp_Shldr_Left'] = Hp_Shldr_Left['Hp_Shldr_Left']
crash_df['Hp_Shldr_Left_Miss'] = Hp_Shldr_Left['Hp_Shldr_Left_Miss']
crash_df['Hp_Shldr_Right'] = Hp_Shldr_Right['Hp_Shldr_Right']
crash_df['Hp_Shldr_Right_Miss'] = Hp_Shldr_Right['Hp_Shldr_Right_Miss']
crash_df['Hwy_Dsgn_Lane_ID'] = Hwy_Dsgn_Lane_ID['Hwy_Dsgn_Lane_ID']
crash_df['Hwy_Dsgn_Hrt_ID'] = Hwy_Dsgn_Hrt_ID['Hwy_Dsgn_Hrt_ID']

del Pct_Combo_Trk_Adt; del Prsn_Type_ID; del Prsn_Occpnt_Pos_ID;del DFO; del Trk_Aadt_Pct; del Pct_Combo_Trk_Adt;
del Pct_Single_Trk_Adt; del Adt_Adj_Curnt_Amt; del Func_Sys_ID; del Rural_Urban_Type_ID; del Median_Width;
del Shldr_Use_Right_ID; del Shldr_Use_Left_ID; del Shldr_Type_Right_ID; del Shldr_Type_Left_ID;
del Row_Width_Usual; del Road_Type_ID; del Median_Type_ID; del Surf_Cond_ID; del Surf_Type_ID;
del Nbr_Of_Lane;del Shldr_Width_Left;del Shldr_Width_Right; del Curb_Type_Left_ID;
del Curb_Type_Right_ID; del Surf_Width; del Roadbed_Width; del Hp_Median_Width; del Hp_Shldr_Left;
del Hp_Shldr_Right; del Hwy_Dsgn_Lane_ID; del Hwy_Dsgn_Hrt_ID;

# Execute Initial Data Manipulation Functions
#########################################################################################################################
crash_col_summ = pandas_col_summary(crash_df)
delete_cols = nan_del_cols(crash_col_summ)
crash_df_ii = remove_del_cols(crash_df, delete_cols)
crash_col_summ_ii = pandas_col_summary(crash_df_ii)

# Execute Damage & Endorsement Functions
#########################################################################################################################
hit_dummies = damage_features(dmg_df = damages_df, str_col = 'Damaged_Property', id_col = 'Crash_ID')
endo_dummies = endorse_dummies(endorsements_df)
crash_df_ii = pd.merge(crash_df_ii, hit_dummies, on = 'Crash_ID', how = 'left')
crash_df_ii[['HIT_FENCE', 'HIT_POLE', 'HIT_RAIL', 'HIT_SIGN', 'HIT_LIGHT', 'HIT_CONC', 'HIT_WALL', 'HIT_TREE', 'HIT_METAL',
       'HIT_BRICK', 'HIT_WOOD', 'HIT_ELEC', 'HIT_TELEPH', 'HIT_HYDRANT', 'HIT_CURB']] = crash_df_ii[['HIT_FENCE', 'HIT_POLE',
       'HIT_RAIL', 'HIT_SIGN', 'HIT_LIGHT', 'HIT_CONC', 'HIT_WALL', 'HIT_TREE', 'HIT_METAL',
       'HIT_BRICK', 'HIT_WOOD', 'HIT_ELEC', 'HIT_TELEPH', 'HIT_HYDRANT', 'HIT_CURB']].fillna(value = 0)

crash_df_ii = pd.merge(crash_df_ii, endo_dummies, on = 'Crash_ID', how = 'left')

crash_df_ii[['Crash_ID', 'CDL2', 'CDL3', 'CDL4', 'CDL5', 'CDL6', 'CDL7', 'CDL8']] = crash_df_ii[['Crash_ID', 'CDL2', 'CDL3', 'CDL4', 'CDL5', 'CDL6', 'CDL7', 'CDL8']].fillna(value = 0)

# Look at Fatality % Over Dates
date_fatal = crash_df_ii[['Crash_ID', 'Crash_Fatal_Fl', 'Crash_Date']]
date_fatal['Crash_Date'] = pd.to_datetime(date_fatal['Crash_Date'], format = '%m/%d/%Y')
date_fatal_agg = date_fatal.groupby(['Crash_Date', 'Crash_Fatal_Fl']).Crash_ID.nunique().reset_index()
date_nonfatal = date_fatal_agg[(date_fatal_agg.Crash_Fatal_Fl) == 'N']
date_nonfatal.columns = ['Crash_Date', 'Crash_Fatal_Fl', 'Num_NonFatal']
date_fatal = date_fatal_agg[(date_fatal_agg.Crash_Fatal_Fl) == 'Y']
date_fatal.columns = ['Crash_Date', 'Crash_Fatal_Fl', 'Num_Fatal']
date_fatal_perc = pd.merge(date_nonfatal[['Crash_Date', 'Num_NonFatal']],
                           date_fatal[['Crash_Date', 'Num_Fatal']],
                           on = 'Crash_Date',
                           how = 'left')

date_fatal_perc['Num_Crashes'] = date_fatal_perc['Num_Fatal'] + date_fatal_perc['Num_NonFatal']
date_fatal_perc['Perc_Fatal'] = date_fatal_perc['Num_Fatal'] / date_fatal_perc['Num_Crashes']
fig, ax = plt.subplots()
ax.plot(date_fatal_perc['Crash_Date'], date_fatal_perc['Perc_Fatal'])
date_fatal_perc.to_csv('C:/Users/user/Desktop/date_fatal_perc.csv')

# Date Functions
crash_df_ii['Crash_Date'] = pd.to_datetime(crash_df_ii['Crash_Date'], format = '%m/%d/%Y')

def date_vars(dat, dt_col = 'Crash_Date', id_col = 'Crash_ID'):
    tm_tuples = []
    for i in dat[dt_col]:
        tm_tuples.append(i.timetuple())
    yr_list = []
    mth_list = []
    day_list = []
    yr_day_list = []
    wday_list = []
    dt_contin = []
    for t in tm_tuples:
        yr_list.append(t[0])
        mth_list.append(t[1])
        day_list.append(t[2])
        yr_day_list.append(t[7] / 366)
        wday_list.append(t[6])
        dt_contin.append(t[0] + (t[7] / 366))
    jul4_list = []
    jul3_list = []
    jan1_list = []
    dec21_25 = []
    for m, i in enumerate(mth_list):
        if (m == 7) and (day_list[i] == 4):
            jul4_list.append(1)
        else:
            jul4_list.append(0)
        if (m == 7) and (day_list[i] == 3):
            jul3_list.append(1)
        else:
            jul3_list.append(0)
        if (m == 1) and (day_list[i] == 1):
            jan1_list.append(1)
        else:
            jan1_list.append(0)
        if (m == 12) and (day_list[i] in [21,22,23,24,25]):
            dec21_25.append(1)
        else:
            dec21_25.append(0)
    output_df = pd.DataFrame({id_col: dat[id_col],
                              'Dt_Contin': dt_contin,
                              'Day_Yr_Contin': yr_day_list,
                              'Jul_3': jul3_list,
                              'Jul_4': jul4_list,
                              'Jan_1': jan1_list,
                              'Dec21_Dec25': dec21_25})
    output_df = output_df.groupby([id_col], sort = False).max().reset_index()
    return output_df
            
crash_date_vars =  date_vars(crash_df_ii, dt_col = 'Crash_Date', id_col = 'Crash_ID')
crash_df_ii = pd.merge(crash_df_ii, crash_date_vars, on = 'Crash_ID', how = 'left')

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

backward_looking_vars = [i for i in backward_looking_vars if i in crash_df_ii.columns]
non_genrl_vars = [i for i in non_genrl_vars if i in crash_df_ii.columns]
crash_df_iii = remove_del_list_cols(crash_df_ii, backward_looking_vars + non_genrl_vars)

# Append Variables from Re: Persons Involved in Crash
#########################################################################################################################
crash_df_iv = pd.merge(crash_df_iii, person_df[['Crash_ID','Prsn_Nbr', 'Prsn_Occpnt_Pos_ID', 'Prsn_Age', 'Prsn_Ethnicity_ID',
                                                'Prsn_Gndr_ID', 'Prsn_Rest_ID', 'Prsn_Airbag_ID', 'Prsn_Helmet_ID',
                                                'Prsn_Alc_Rslt_ID', 'Prsn_Bac_Test_Rslt', 'Prsn_Drg_Rslt_ID']],
    on = 'Crash_ID',
    how = 'left')

crash_df_iv = pd.merge(crash_df_iv,
                       primaryperson_df[['Crash_ID', 'Drvr_Lic_Type_ID', 'Drvr_Lic_Cls_ID']],
                       on = 'Crash_ID',
                       how = 'left')

crash_df_iv_summ = pandas_col_summary(crash_df_iv)

# Categoization & Treatment of Remaining Variables
#########################################################################################################################
# List of Variables by Type
label_vars = ['Crash_ID', 'Prsn_Nbr']
dummy_code_vars = ['Crash_Fatal_Fl', 'Cmv_Involv_Fl', 'Schl_Bus_Fl', 'Rr_Relat_Fl', 'Active_School_Zone_Fl',
                   'Rpt_Outside_City_Limit_Fl', 'Private_Dr_Fl', 'Toll_Road_Fl', 'Road_Constr_Zone_Fl',
                   'Road_Constr_Zone_Wrkr_Fl', 'At_Intrsct_Fl', 'Onsys_Fl', 'Rural_Fl', 'Pct_Combo_Trk_Adt_Miss',
                   'Dfo_Miss', 'Trk_Aadt_Pct_Miss', 'Pct_Single_Trk_Adt_Miss', 'Adt_Adj_Curnt_Amt_Miss',
                   'Median_Width_Miss', 'Row_Width_Usual_Miss', 'Nbr_Of_Lane_Miss', 'Shldr_Width_Left_Miss',
                   'Shldr_Width_Right_Miss', 'Curb_Type_Left_ID_Miss', 'Curb_Type_Right_ID_Miss', 'Surf_Width_Miss',
                   'Roadbed_Width_Miss', 'Hp_Median_Width_Miss', 'Hp_Shldr_Left_Miss', 'Hp_Shldr_Right_Miss',
                   'Hwy_Dsgn_Hrt_ID', 'Road_Type_ID', 'Intrsct_Relat_ID', 'Curb_Type_Left_ID', 'Curb_Type_Right_ID',
                   'Median_Type_ID', 'Rural_Urban_Type_ID', 'Rpt_Road_Part_ID', 'Road_Part_Adj_ID', 'Road_Relat_ID', 
                   'Surf_Type_ID', 'Shldr_Type_Left_ID', 'Shldr_Type_Right_ID', 'Day_of_Week',
                   'Hwy_Dsgn_Lane_ID', 'Drvr_Lic_Type_ID', 'Light_Cond_ID', 'Road_Algn_ID', 'Surf_Cond_ID',
                   'Bridge_Detail_ID', 'Road_Cls_ID', 'Shldr_Use_Left_ID', 'Shldr_Use_Right_ID', 'Entr_Road_ID',
                   'Pop_Group_ID', 'Wthr_Cond_ID', 'Drvr_Lic_Cls_ID', 'Phys_Featr_2_ID', 'Func_Sys_ID', 'Phys_Featr_1_ID',
                   'FHE_Collsn_ID', 'Obj_Struck_ID', 'Othr_Factr_ID', 'Prsn_Occpnt_Pos_ID',
                   'Prsn_Rest_ID',  'Prsn_Airbag_ID', 'Prsn_Helmet_ID', 'Prsn_Alc_Rslt_ID', 'Prsn_Bac_Test_Rslt',
                   'Prsn_Drg_Rslt_ID', 'Prsn_Ethnicity_ID', 'Prsn_Gndr_ID','Jul_3', 'Jul_4', 'Jan_1', 'Dec21_Dec25',
                   'CDL2', 'CDL3', 'CDL4', 'CDL5', 'CDL6', 'CDL7', 'CDL8','HIT_FENCE', 'HIT_POLE', 'HIT_RAIL',
                   'HIT_SIGN', 'HIT_LIGHT', 'HIT_CONC', 'HIT_WALL', 'HIT_TREE', 'HIT_METAL', 'HIT_BRICK', 'HIT_WOOD',
                   'HIT_ELEC', 'HIT_TELEPH', 'HIT_HYDRANT', 'HIT_CURB']
contin_vars = ['Hp_Shldr_Right', 'Hp_Shldr_Left', 'Shldr_Width_Right', 'Shldr_Width_Left', 'Crash_Speed_Limit', 'Prsn_Age',
               'Surf_Width', 'Median_Width', 'Pct_Single_Trk_Adt', 'Roadbed_Width', 'Hp_Median_Width', 'Pct_Combo_Trk_Adt',
               'Row_Width_Usual', 'Trk_Aadt_Pct', 'Crash_Time', 'Adt_Adj_Curnt_Amt', 'Dfo', 'Dt_Contin', 'Day_Yr_Contin']
time_vars = ['Crash_Time']
crash_df_v = crash_df_transform(crash_df_iv, label_vars, dummy_code_vars, time_vars)
crash_df_v['Prsn_Nbr'].fillna(1, inplace = True)
crash_df_v_summ = pandas_col_summary(crash_df_v)

# Garbage Collection
#########################################################################################################################
del crash_df_iii; del crash_df_iv; del crash_df_ii; del charges_df; del damages_df
del crash_date_vars; del crash_df; del endorsements_df; del hit_dummies; del lookup_df
del primaryperson_df; del restrictions_df; del unit_df; del endo_dummies;

# Aggregate from Person Level to Crash ID Level
#########################################################################################################################
age_df = dummy_trans(pd.DataFrame({'Prsn_Age':pd.cut(crash_df_v['Prsn_Age'], 15)}), ['Prsn_Age'])
age_df.columns = ['Age_Bucket_1', 'Age_Bucket_2', 'Age_Bucket_3', 'Age_Bucket_4', 'Age_Bucket_5',
                  'Age_Bucket_6', 'Age_Bucket_7', 'Age_Bucket_8', 'Age_Bucket_9', 'Age_Bucket_10',
                  'Age_Bucket_11', 'Age_Bucket_12', 'Age_Bucket_13', 'Age_Bucket_14']
age_df['Crash_ID'] = crash_df_v['Crash_ID']
#age_df['Prsn_Nbr'] = crash_df_v['Prsn_Nbr']
age_df = age_df.groupby(['Crash_ID']).sum()
crash_df_vi = crash_df_v.drop(['Prsn_Nbr', 'Prsn_Age'], axis = 1, inplace = False)
#crash_df_vi_summ = pandas_col_summary(crash_df_vi)

person_cols = ['Crash_ID', 'Prsn_Age_missing', 'Prsn_Airbag_ID_1.0', 'Prsn_Airbag_ID_2.0', 'Prsn_Airbag_ID_3.0',
               'Prsn_Airbag_ID_4.0', 'Prsn_Airbag_ID_7.0', 'Prsn_Airbag_ID_8.0', 'Prsn_Ethnicity_ID_1.0',
               'Prsn_Ethnicity_ID_2.0', 'Prsn_Ethnicity_ID_3.0', 'Prsn_Ethnicity_ID_4.0', 'Prsn_Ethnicity_ID_5.0',
               'Prsn_Ethnicity_ID_6.0', 'Prsn_Gndr_ID_1.0', 'Prsn_Gndr_ID_2.0', 'Prsn_Helmet_ID_2.0', 'Prsn_Helmet_ID_3.0',
               'Prsn_Helmet_ID_4.0', 'Prsn_Helmet_ID_5.0', 'Prsn_Helmet_ID_97.0', 'Prsn_Occpnt_Pos_ID_10.0',
               'Prsn_Occpnt_Pos_ID_11.0', 'Prsn_Occpnt_Pos_ID_13.0', 'Prsn_Occpnt_Pos_ID_14.0',
               'Prsn_Occpnt_Pos_ID_16.0', 'Prsn_Occpnt_Pos_ID_2.0', 'Prsn_Occpnt_Pos_ID_3.0',
               'Prsn_Occpnt_Pos_ID_4.0', 'Prsn_Occpnt_Pos_ID_5.0', 'Prsn_Occpnt_Pos_ID_6.0', 'Prsn_Occpnt_Pos_ID_7.0',
               'Prsn_Occpnt_Pos_ID_8.0', 'Prsn_Occpnt_Pos_ID_9.0', 'Prsn_Occpnt_Pos_ID_94.0', 'Prsn_Rest_ID_10.0',
               'Prsn_Rest_ID_11.0', 'Prsn_Rest_ID_2.0', 'Prsn_Rest_ID_3.0', 'Prsn_Rest_ID_4.0', 'Prsn_Rest_ID_5.0',
               'Prsn_Rest_ID_6.0', 'Prsn_Rest_ID_7.0', 'Prsn_Rest_ID_8.0', 'Prsn_Rest_ID_9.0']

nonperson_cols = [x for x in crash_df_vi.columns if x not in person_cols] + ['Crash_ID']
person_df = crash_df_vi[person_cols].groupby('Crash_ID', as_index = False).sum()
nonperson_df = crash_df_v[nonperson_cols].drop_duplicates()
nonperson_df = nonperson_df.groupby('Crash_ID', as_index = False).max()
crash_df_vii = pd.merge(nonperson_df, person_df, on = 'Crash_ID').drop(columns = ['Crash_Fatal_Fl_Y'])

# Create Final, Model-Ready Dataframes
#########################################################################################################################
def scale_cont_cols(dat, id_col):
    colname_list = []
    num_uniq_list = []
    for col in dat.columns:
        colname_list.append(col)
        num_uniq_list.append(dat[col].agg('nunique'))
    count_df = pd.DataFrame({'Field_Name': colname_list,
                             'Num_Unique': num_uniq_list})
    continuous_vars = [x for x in count_df[count_df.Num_Unique > 2]['Field_Name']]
    continuous_vars = [x for x in continuous_vars if x != id_col]
    other_vars = [x for x in count_df['Field_Name'] if x not in continuous_vars]
    continuous_df = dat[continuous_vars]
    other_df = dat[other_vars]
    scaler = StandardScaler()
    scaler.fit(continuous_df)
    continuous_df[continuous_vars] = scaler.transform(continuous_df[continuous_vars])
    continuous_df[id_col] = dat[id_col]
    scaled_output = pd.merge(other_df, continuous_df, on = id_col)
    return scaled_output, dat
    
final_data_scaled, final_data_unscaled = scale_cont_cols(crash_df_vii, 'Crash_ID')

# Create Final, Model-Ready Dataframes
#########################################################################################################################
final_data_scaled.to_csv("C:/Users/user/Desktop/school/dddm/trf_fatality_scaled_07142018.csv", index = False)
final_data_unscaled.to_csv("C:/Users/user/Desktop/school/dddm/trf_fatality_unscaled_07142018.csv", index = False)


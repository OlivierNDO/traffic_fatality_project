# Import Packages
#########################################################################################################################
import pandas as pd, numpy as np, tensorflow as tf
import time, math, itertools
from datetime import datetime
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from scipy import sparse
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, log_loss
from tqdm import tqdm
from random import choices
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# Data Import
#########################################################################################################################
save_path = '<save_path>'
agg_scaled_data = pd.read_csv('.../trf_fatality_scaled_07142018.csv')
y = agg_scaled_data.iloc[:,1]
x = agg_scaled_data.iloc[:,2:447]

# Make Five List of Indices for Cross Validation
#########################################################################################################################
def batch_iters(lst,n):
    return [lst[i::n] for i in range(n)]

def five_fold_cv_with_val(y_col):
    n_obs = len(y_col)
    obs_shuffled = shuffle(range(0,n_obs))
    fold_positions = [obs_shuffled[i::5] for i in range(5)]
    train_list = []
    test_list = []
    val_list = []
    for i, f in enumerate(fold_positions):
        train_folds = [0,1,2,3,4][:i] + [0,1,2,3,4][(i + 1):]
        train_list.append(list(itertools.chain.from_iterable([fold_positions[q] for q in train_folds])))
        #train_list.append(fold_positions[train_folds])
        nontrain = shuffle(fold_positions[i])
        nontrain_split = batch_iters(nontrain, 2)
        nontrain_testsub = nontrain_split[0]
        nontrain_valsub = nontrain_split[1]
        test_list.append(nontrain_testsub)
        val_list.append(nontrain_valsub)
    return train_list, test_list, val_list

train_pos, test_pos, val_pos = five_fold_cv_with_val(y)

# Neural Network
######################################################################################
def balanced_batch_pos(train_y, size_batch):
    pos_0 = []
    pos_1 = []
    for x, i in enumerate(train_y):
        if i == 0:
            pos_0.append(x)
        else:
            pos_1.append(x)
    num_batch = (len(pos_0) + len(pos_1)) // size_batch
    batch_list = []
    for i in range(0, num_batch):
        sample0 = choices(pos_0, k = size_batch//2)
        sample1 = choices(pos_1, k = size_batch//2)
        batch_list.append(shuffle(sample0 + sample1))
    return batch_list

def nnet_eval_cv5(X, Y, Trn_Pos, Tst_Pos, Val_Pos):
    # Placeholders for Evaluation Metrics
    Cv_List = []
    Auc_List = []
    for i, v in enumerate(Trn_Pos):
        # Train, Test, Validation CV Split
        ######################################################################################
        Train_X = X.iloc[Trn_Pos[i]].astype(np.float32)
        Test_X = X.iloc[Tst_Pos[i]].astype(np.float32)
        Val_X = X.iloc[Val_Pos[i]].astype(np.float32)
        Train_Y = Y[Trn_Pos[i]].as_matrix()
        Test_Y = Y[Tst_Pos[i]].as_matrix()
        Val_Y = Y[Val_Pos[i]].as_matrix()
        # Neural Net Parameters
        ######################################################################################
        tf.reset_default_graph()
        # Learning Rate Decay
        initial_learning_rate = 0.1
        decay_steps = 10000
        decay_rate = 1/10
        scale = 0.001
        global_step = tf.Variable(0, trainable = False, name = 'global_step')
        learn_rt = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
        # Size of Inputs, Output Classes, & Hidden Layers
        n_inputs = Train_X.shape[1]
        n_classes = 2
        n_hidden1 = 120
        n_hidden2 = 120
        dropout = 0.3
        # Iterations & Batches
        n_epochs = 100
        batch_size = 1000
        early_stop_epocs = 3
        batch_i = balanced_batch_pos(Train_Y, batch_size)
        # Neural Net Structure
        ######################################################################################
        x = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'x')
        y = tf.placeholder(tf.int64)
        # Regularization
        my_dense_layer = partial(tf.layers.dense, activation = tf.nn.relu, kernel_regularizer = tf.contrib.layers.l1_regularizer(scale))
        wt1 = tf.get_variable("wt1", shape=[5],initializer=tf.contrib.layers.xavier_initializer())
        wt2 = tf.get_variable("wt2", shape=[5],initializer=tf.contrib.layers.xavier_initializer())
        # Define Layers
        with tf.name_scope('dnn'):
            hidden1 = my_dense_layer(x, n_hidden1, name = 'hidden1')
            d_out1 = tf.layers.dropout(hidden1, rate=dropout)
            hidden2 = my_dense_layer(d_out1, n_hidden2, name = 'hidden2')
            d_out = tf.layers.dropout(hidden2, rate=dropout)
            logits = my_dense_layer(d_out, n_classes, activation = None, name = 'outputs')
        # Define Loss Function : tf.nn.sparse_softmax_cross_entropy_with_logits
        with tf.name_scope('loss'):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
            base_loss = tf.reduce_mean(xentropy, name = 'avg_xentropy')
            reg_losses = tf.reduce_sum(tf.abs(wt1) + tf.reduce_sum(tf.abs(wt2)))
            loss = tf.add(base_loss, scale * reg_losses, name = 'loss')
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([base_loss] + reg_losses, name = 'loss')
        # Define Gradient Descent Optimizer
        with tf.name_scope('train'): 
            optimizer = tf.train.MomentumOptimizer(learning_rate = learn_rt, momentum = 0.9)
            training_op = optimizer.minimize(loss, global_step = global_step)    
        # Define Performance Evaluation
        with tf.name_scope('eval'):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        # Fit Neural Network
        ######################################################################################
        # Initialize TF Session
        init = tf.global_variables_initializer()
        pred_list = []
        epoch_list = []
        trn_acc = []
        val_acc = []
        val_auc = []
        # Execute
        with tf.Session() as sess:
            init.run()
            best_auc = 0
            auc_list = []
            break_list = []
            for epoch in range(n_epochs):
                for b in range(0,len(batch_i)):
                    batch_x = Train_X.iloc[batch_i[b]]
                    batch_y = Train_Y[batch_i[b]]
                    sess.run(training_op, feed_dict = {x: batch_x, y: batch_y})
                acc_train = accuracy.eval(feed_dict = {x: batch_x, y: batch_y})
                acc_test = accuracy.eval(feed_dict = {x: Val_X, y: Val_Y})
                prob_test = logits.eval(feed_dict={x: Val_X, y: Val_Y})
                epoch_auc = roc_auc_score(Val_Y, prob_test[:,1])
                print(epoch, 'Training Accuracy: ', acc_train,
                      ' Validation Accuracy: ', acc_test,
                      ' Validation AUC: ', epoch_auc)
                epoch_list.append(epoch)
                trn_acc.append(acc_train)
                val_acc.append(acc_test)
                val_auc.append(epoch_auc)
                auc_list.append(epoch_auc)
                best_auc = max(auc_list)
                if epoch_auc < best_auc:
                    break_list.append(1)
                else:
                    break_list = []
                if sum(break_list) >= early_stop_epocs:
                    print("Stopping after " + str(epoch) + "\
                          epochs because AUC hasn't improved in \
                          " + str(early_stop_epocs) + " rounds.")
                    break
            make_pred = logits.eval(feed_dict = {x: Test_X})
            pred_list.append(make_pred)
            y_pred = [i for i in pred_list[0][:,1]]
            Cv_List.append(i)
            Auc_List.append(roc_auc_score(Test_Y, y_pred))
    output_df = pd.DataFrame({'CV_Iteration': Cv_List,
                              'AUC': Auc_List})
    return output_df

# Gradient Boosting
######################################################################################
def xgb_eval_cv5(X, Y, Trn_Pos, Tst_Pos, Val_Pos):
    # Placeholders for Evaluation Metrics
    Cv_List = []
    Auc_List = []
    for i, v in enumerate(Trn_Pos):
        # Train, Test, Validation CV Split
        Train_X = X.iloc[Trn_Pos[i]]
        Test_X = X.iloc[Tst_Pos[i]]
        Val_X = X.iloc[Val_Pos[i]]
        Train_Y = Y[Trn_Pos[i]]
        Test_Y = Y[Tst_Pos[i]]
        Val_Y = Y[Val_Pos[i]]
        yvar_imbalance = (Train_Y.count() - Train_Y.sum()) / Train_Y.sum()
        params = {}
        params['objective'] = 'binary:logistic'
        params['booster'] = 'gbtree'
        params['eval_metric'] = 'logloss'
        params['eta'] = 0.03
        params['max_depth'] = 6
        params['min_child_weight'] = 12
        params['colsample_bytree'] = 0.55
        params['colsample_bylevel'] = 0.25
        params['subsample'] = 0.5
        params['reg_lambda'] = .05
        params['reg_alpha'] = 0
        params['min_split_loss'] = 10
        params['scale_pos_weight'] = yvar_imbalance
        nrnd = 3400 # n_estimators
        dat_train = xgb.DMatrix(Train_X, label = Train_Y)
        dat_valid = xgb.DMatrix(Val_X, label = Val_Y)
        watchlist = [(dat_train, 'train'), (dat_valid, 'valid')]
        xgb_trn = xgb.train(params, dat_train, nrnd, watchlist, early_stopping_rounds=12, verbose_eval=1)
        pred = xgb_trn.predict(xgb.DMatrix(Test_X))
        pred_df = pd.DataFrame({'Predicted': pred, 'Actual': Test_Y})
        Cv_List.append(i)
        Auc_List.append(roc_auc_score(pred_df['Actual'], pred_df['Predicted']))
    output_df = pd.DataFrame({'CV_Iteration': Cv_List, 'AUC': Auc_List})
    return output_df

# Random Forest
######################################################################################
def rf_eval_cv5(X, Y, Trn_Pos, Tst_Pos, Val_Pos):
    # Placeholders for Evaluation Metrics
    Cv_List = []
    Auc_List = []
    # Parameters
    for i, v in enumerate(Trn_Pos):
        # Train, Test, Validation CV Split
        Train_X = X.iloc[Trn_Pos[i]]
        Test_X = X.iloc[Tst_Pos[i]]
        Train_Y = Y[Trn_Pos[i]]
        Test_Y = Y[Tst_Pos[i]]
        rf = RandomForestClassifier(n_jobs = -1,
                                    max_features = 'auto',
                                    n_estimators = 500,
                                    max_depth = 5,
                                    oob_score = True,
                                    class_weight = 'balanced',
                                    min_samples_split = 10,
                                    min_samples_leaf = 5,
                                    verbose = 3)
        rf.fit(Train_X, Train_Y)
        pred = rf.predict_proba(Test_X)
        rf_prediction = []
        for r in pred:
            rf_prediction.append(r[1])
        pred_df = pd.DataFrame({'Predicted': rf_prediction, 'Actual': Test_Y})
        Cv_List.append(i)
        Auc_List.append(roc_auc_score(pred_df['Actual'], pred_df['Predicted']))
    output_df = pd.DataFrame({'CV_Iteration': Cv_List,
                              'AUC': Auc_List})
    return output_df

# Execute Cross Validation Functions
######################################################################################
# Neural Network
start_tm = time.time()
nnet_cv5 = nnet_eval_cv5(x, y, train_pos, test_pos, val_pos)
end_tm = time.time()
print('Neural Network 5-Fold CV Time (sec.): ' + str(np.round(end_tm - start_tm,0)))
nnet_cv5.to_csv(save_path + 'nnet_cv5.csv')

# XGBoost
start_tm = time.time()
xgb_cv5 = xgb_eval_cv5(x, y, train_pos, test_pos, val_pos)
end_tm = time.time()
print('XGBoost 5-Fold CV Time (sec.): ' + str(np.round(end_tm - start_tm,0)))
xgb_cv5.to_csv(save_path + 'xgb_cv5.csv')

# Random Forest
start_tm = time.time()
rf_cv5 = rf_eval_cv5(x, y, train_pos, test_pos, val_pos)
end_tm = time.time()
print('Random Forest 5-Fold CV Time (sec.): ' + str(np.round(end_tm - start_tm,0)))
rf_cv5.to_csv(save_path + 'rf_cv5.csv')

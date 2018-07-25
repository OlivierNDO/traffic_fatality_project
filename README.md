Overview:
  - classication algorithm estimating probability of fatality in Texas car crashes

Data Source: 
  - Texas Department of Transportation

Sections:
  - crash_data_processing.py
      > aggregate, reformat, create new features
  - crash_tensorflow_earlystop.py
      > try different parameters & architectures in Tensorflow
  - cross_val_cv5.py
      > five-fold cross validation comparing Tensoflow deep neural network, gradient boosting, and random forest models

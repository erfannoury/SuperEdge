import numpy as np
import scipy.sparse
import xgboost as xgb
from sklearn.externals import joblib
from os import path
def main():
    cache_path = 'cachelarge/'
    dtrain_name = 'data.train.txt#dtrain.cache'
    dtrain = xgb.DMatrix(path.join(cache_path, dtrain_name))
    param = {'max_depth':15, 
             'eta':1,
             'learning_rate':0.1,
             'gamma':0,
             'silent':1,
             'objective':'binary:logistic',
             'n_estimators':50,
             'nthread':24 }

    bst = xgb.train(param, dtrain)
    bst.save_model('../../../Models/external16_all_XGBC_d15_n50_int_th.xgb')



if __name__ == '__main__':
    main()
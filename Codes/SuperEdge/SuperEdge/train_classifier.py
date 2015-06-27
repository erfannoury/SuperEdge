import numpy as np
from extract_feature import VGG16Extractor
from poisson_disc import PoissonDiskSampler
from datetime import datetime
from bsds500 import BSDS
from sklearn.externals import joblib
import xgboost as xgb
import os.path as path

def main():
    cache_path = 'cache/'
    feat_name = 'feat.dat'
    lbl_name = 'lbl.dat'
    feat_len = 1088 #4224
    now = datetime.now()
    lbl_memmap = np.memmap(path.join(cache_path, lbl_name), dtype='float32', mode='r')
    feat_memmap = np.memmap(path.join(cache_path, feat_name), dtype='float32', mode='r', shape=(lbl_memmap.shape[0], feat_len))
    print 'loading dataset took ', (datetime.now() - now)
    now = datetime.now()
    print 'starting training XGBoost classifier'
    clf = xgb.XGBClassifier(max_depth=20, nthread=24, n_estimators=150)
    print 'XGBoost classifier parameters: ', clf
    clf.fit(features, (labels > 0).astype(uint8))
    print 'training xgboost classifier took: ', (datetime.now() - now)
    print 'saving trained classifier'
    joblib.dump(clf, '../../../Models/ondisk_20_XGBC.pkl', compress=True)


if __name__ == '__main__':
    main()
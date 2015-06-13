import numpy as np
import cv2
from extract_feature import VGG16Extractor
from poisson_disc import PoissonDiskSampler
from datetime import datetime
from bsds500 import BSDS
from sklearn.externals import joblib
import xgboost as xgb

def main():
    now = datetime.now()    
    vgg = VGG16Extractor()
    pds = PoissonDiskSampler(vgg.image_width, vgg.image_height, 2)
    samples = pds.get_sample()
    Xtrain, ytrain = BSDS.load(which='train')
    features = []
    labels = []
    for i in xrange(50):
        hyperimage = vgg.transform(Xtrain[i,...])
        print i, ' ', hyperimage.shape
        for y in xrange(hyperimage.shape[0]):
            for x in xrange(hyperimage.shape[1]):
                if ytrain[i, y, x] > 0:
                    features.append(hyperimage[y,x,:])
                    labels.append(ytrain[i,y,x])
        for s in samples:
            if ytrain[i, s[1], s[0]] == 0:
                features.append(hyperimage[s[1],s[0],:])
                labels.append(ytrain[i,s[1], s[0]])
        hyperimage = None
    print 'transforming training set took ', (datetime.now() - now)
    features = np.asarray(features, dtype=np.float32)
    print 'features.shape: ', features.shape
    labels = np.asarray(labels, dtype=np.float32)
    print 'labels.shape: ', labels.shape

    now = datetime.now()
    print 'starting training XGBoost regressor'
    clf = xgb.XGBRegressor(max_depth=10, nthread=12, min_child_weight=2)
    print 'XGBoost regressor Parameters: ', clf
    clf.fit(features, labels)
    print 'training xgboost regressor took: ', (datetime.now() - now)
    print 'saving trained regressor'
    joblib.dump(clf, '../../../Models/XGBR.pkl', compress=True)


if __name__ == '__main__':
    main()

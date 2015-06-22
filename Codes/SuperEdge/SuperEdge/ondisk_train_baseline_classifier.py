import numpy as np
import cv2
from extract_feature import VGG16Extractor
from poisson_disc import PoissonDiskSampler
from datetime import datetime
from bsds500 import BSDS
from sklearn.externals import joblib
import xgboost as xgb
import os.path as path

def main():
    cache_path = 'cache/'
    batch_size = 50
    feat_batch_name = '_feat.dat'
    feat_count = 0
    feat_len = 0
    lbl_batch_name = '_lbl.dat'
    now = datetime.now()    
    vgg = VGG16Extractor()
    pds = PoissonDiskSampler(vgg.image_width, vgg.image_height, 4)
    samples = pds.get_sample()
    Xtrain, ytrain, _ = BSDS.load(which='train')
    idx = -1
    for b in xrange(int(np.ceil(Xtrain.shape[0] * 1.0 / batch_size))):
        features = []
        labels = []
        for _ in xrange(batch_size):
            idx += 1
            if idx >= Xtrain.shape[0]:
                break
            hyperimage = vgg.transform(Xtrain[idx,...])
#            print idx, ' ', hyperimage.shape
            for y in xrange(hyperimage.shape[0]):
                for x in xrange(hyperimage.shape[1]):
                    if ytrain[idx, y, x] > 0:
                        features.append(hyperimage[y,x,:])
                        labels.append(ytrain[idx,y,x])
            for s in samples:
                if ytrain[idx, s[1], s[0]] == 0:
                    features.append(hyperimage[s[1],s[0],:])
                    labels.append(ytrain[idx,s[1], s[0]])
            hyperimage = None
        print 'transforming batch ', b, ' of training set took ', (datetime.now() - now)
        features = np.asarray(features, dtype=np.float32)
        print 'batch ', b, ' features.shape: ', features.shape
        feat_count += features.shape[0]
        feat_len = features.shape[1]
        labels = np.asarray(labels, dtype=np.float32)
        print 'batch ', b, ' labels.shape: ', labels.shape
        feat_memmap = np.memmap(path.join(cache_path, str.format('%d%s' % (b, feat_batch_name))), dtype='float32', mode='w+', shape=features.shape)
        lbl_memmap = np.memmap(path.join(cache_path, str.format('%d%s' % (b, lbl_batch_name))), dtype='float32', mode='w+', shape=labels.shape)
        feat_memmap[:] = features
        del feat_memmap
        lbl_memmap[:] = labels
        del lbl_memmap

    print 'transforming all batches of training set finished'
    print 'creating the final training set'
    now = datetime.now()
    feat_memmap = np.memmap(path.join(cache_path, feat_batch_name[1:]), dtype='float32', mode='w+', shape=(feat_count, feat_len))
    lbl_memmap = np.memmap(path.join(cache_path, lbl_batch_name[1:]), dtype='float32', mode='w+', shape=(feat_count,))
    count = 0
    for b in xrange(int(np.ceil(Xtrain.shape[0] * 1.0 / batch_size))):
        bfeat_memmap = np.memmap(path.join(cache_path, str.format('%d%s' % (b, feat_batch_name))), dtype='float32', mode='r')
        blbl_memmap = np.memmap(path.join(cache_path, str.format('%d%s' % (b, lbl_batch_name))), dtype='float32', mode='r')
        feat_memmap[count:count + bfeat_memmap.shape[0],:] = bfeat_memmap
        lbl_memmap[count:count + bfeat_memmap.shape[0]] = blbl_memmap
        count += bfeat_memmap.shape[0]
        feat_memmap.flush()
        lbl_memmap.flush()
    print 'creating the final training set took: ', (datetime.now() - now)
    print 'fianl training set size: ', feat_memmap.shape
    now = datetime.now()
    print 'starting training XGBoost regressor'
    clf = xgb.XGBRegressor(max_depth=20, nthread=24, n_estimators=150, objective='reg:logistic')
    print 'XGBoost regressor Parameters: ', clf
    clf.fit(features, labels)
    print 'training xgboost regressor took: ', (datetime.now() - now)
    print 'saving trained regressor'
    joblib.dump(clf, '../../../Models/ondisk_XGBR.pkl', compress=True)


if __name__ == '__main__':
    main()

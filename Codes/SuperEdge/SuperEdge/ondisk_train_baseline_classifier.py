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
    cache_path = 'cachelarge/'
    batch_size = 25
    feat_batch_name = '_feat.dat'
    lbl_batch_name = '_lbl.dat'
    now = datetime.now()    
    vgg = VGG16Extractor()
    pds = PoissonDiskSampler(vgg.image_width, vgg.image_height, 3)
    samples = pds.get_sample()
    Xtrain, ytrain, _ = BSDS.load(which='train')
    shapes = {}
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
        shapes[b] = features.shape
        labels = np.asarray(labels, dtype=np.float32)
        print 'batch ', b, ' labels.shape: ', labels.shape
        feat_memmap = np.memmap(path.join(cache_path, str.format('%d%s' % (b, feat_batch_name))), dtype='float32', mode='w+', shape=features.shape)
        lbl_memmap = np.memmap(path.join(cache_path, str.format('%d%s' % (b, lbl_batch_name))), dtype='uint8', mode='w+', shape=labels.shape)
        feat_memmap[:] = features
	feat_memmap.flush()
        del feat_memmap
	del features
        lbl_memmap[:] = (labels > 0.3).astype(np.uint8)
	lbl_memmap.flush()
        del lbl_memmap
	del labels

    feat_count = 0
    feat_len = 0
    for b in shapes.keys():
        feat_count += shapes[b][0]
        feat_len = shapes[b][1]
    print 'transforming all batches of training set finished'
    print 'creating the final training set'
    now = datetime.now()
    feat_memmap = np.memmap(path.join(cache_path, feat_batch_name[1:]), dtype='float32', mode='w+', shape=(feat_count, feat_len))
    lbl_memmap = np.memmap(path.join(cache_path, lbl_batch_name[1:]), dtype='uint8', mode='w+', shape=(feat_count,))
    count = 0
    for b in xrange(int(np.ceil(Xtrain.shape[0] * 1.0 / batch_size))):
        bfeat_memmap = np.memmap(path.join(cache_path, str.format('%d%s' % (b, feat_batch_name))), dtype='float32', mode='r', shape=shapes[b])
        blbl_memmap = np.memmap(path.join(cache_path, str.format('%d%s' % (b, lbl_batch_name))), dtype='uint8', mode='r', shape=(shapes[b][0],))
        feat_memmap[count:count + bfeat_memmap.shape[0],:] = bfeat_memmap
        lbl_memmap[count:count + bfeat_memmap.shape[0]] = blbl_memmap
        count += shapes[b][0]
        feat_memmap.flush()
        lbl_memmap.flush()
    print 'creating the final training set took: ', (datetime.now() - now)
    print 'final training set size: ', feat_memmap.shape
    now = datetime.now()
    print 'starting training the classifier'
    clf = xgb.XGBClassifier(max_depth=10, nthread=24, n_estimators=30)
    print 'Classifier parameters: ', clf
    clf.fit(feat_memmap, lbl_memmap)
    print 'training classifier took: ', (datetime.now() - now)
    print 'saving trained classifier'
    joblib.dump(clf, '../../../Models/ondisk16_25_XGBC_d10_n30_int_th.pkl', compress=True)


if __name__ == '__main__':
    main()

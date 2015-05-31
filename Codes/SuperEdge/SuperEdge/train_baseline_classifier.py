import numpy as np
import cv2
from extract_feature import VGG16Extractor
from poisson_disc import PoissonDiskSampler
from datetime import datetime
from bsds500 import BSDS
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.externals import joblib

def main():
    now = datetime.now()    
    vgg = VGG16Extractor()
    pds = PoissonDiskSampler(vgg.image_width, vgg.image_height, 4)
    samples = pds.get_sample()
    Xtrain, ytrain = BSDS.load(which='train')
    features = []
    labels = []
    for i in xrange(Xtrain.shape[0]):
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
    print 'starting training RandomForest classifier'
    rf = RF(n_jobs=12)
    print 'RandomForest Parameters: ', rf
    rf.fit(features, (labels > 0).astype(np.int32))
    print 'training RandomForest classifier took: ', (datetime.now() - now)
    print 'saving trained classifier'
    joblib.dump(rf, '../../../Models/RandomForest.pkl', compress=True)


if __name__ == '__main__':
    main()
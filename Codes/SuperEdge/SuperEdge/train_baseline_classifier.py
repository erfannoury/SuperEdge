import numpy as np
import cv2
from extract_feature import VGG16Extractor
from poisson_disc import PoissonDiskSampler
from datetime import datetime
from bsds500 import BSDS
from sklearn.svm import LinearSVC
import cPickle

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
                labels.append(0)
        hyperimage = None
    print 'transforming training set took ', (datetime.now() - now)
    features = np.asarray(features, dtype=np.float32)
    print 'features.shape: ', features.shape
    labels = np.asarray(labels, dtype=np.float32)
    print 'labels.shape: ', labels.shape

    now = datetime.now()
    print 'starting training LinearSVM classifier'
    svc = LinearSVC()
    print 'LinearSVM Parameters: ', svc
    svc.fit(features, int(labels > 0))
    print 'training LinearSVM classifier took: ', (datetime.now() - now)
    print 'saving trained classifier'
    f = file('../../../Models/linearSVC.pkl', 'wb')
    cPickle.dump(svc, f, protocol=2)
    f.close()


if __name__ == '__main__':
    main()
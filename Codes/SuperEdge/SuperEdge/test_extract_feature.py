import numpy as np
from scipy.misc import imread
from extract_feature import VGG16Extractor
from datetime import datetime
from bsds500 import BSDS

def main():
    now = datetime.now()    
    vgg = VGG16Extractor()
    Xtrain, ytrain = BSDS.load(which='train')
    for i in xrange(Xtrain.shape[0]):
        hyperimage = vgg.transform(Xtrain[i,...])
        print i, ' ', hyperimage.shape
    print 'transforming training set took ', (datetime.now() - now)



if __name__ == '__main__':
    main()
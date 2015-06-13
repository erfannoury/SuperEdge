import numpy as np
from bsds500 import BSDS
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    now = datetime.now()
    Xtrain, ytrain, _ = BSDS.load(which='train')
    print Xtrain.shape, ytrain.shape, len(_)
    Xtest, ytest, _ = BSDS.load(which='test')
    print Xtest.shape, ytest.shape, len(_)
    Xval, yval, _ = BSDS.load(which='val')
    print Xval.shape, yval.shape, len(_)
    print 'loading all data took ', (datetime.now() - now)


    print 'displaying a random sample from training set'
    idx = np.random.randint(0, Xtrain.shape[0])
    plt.subplot(1,2,1)
    plt.imshow(Xtrain[idx,...].astype(np.uint8))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(ytrain[idx,...], cmap='gray')
    plt.axis('off')

    plt.show()
    print 'displaying a random sample from testing set'
    idx = np.random.randint(0, Xtest.shape[0])
    plt.subplot(1,2,1)
    plt.imshow(Xtest[idx,...].astype(np.uint8))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(ytest[idx,...], cmap='gray')
    plt.axis('off')

    plt.show()
    print 'displaying a random sample from validation set'
    idx = np.random.randint(0, Xval.shape[0])
    plt.subplot(1,2,1)
    plt.imshow(Xval[idx,...].astype(np.uint8))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(yval[idx,...], cmap='gray')
    plt.axis('off')

    plt.show()

    Xtrain, ytrain, _ = BSDS.list_load(which='train')
    print len(Xtrain), len(ytrain), len(_)
    Xtest, ytest, _ = BSDS.list_load(which='test')
    print len(Xtest), len(ytest), len(_)
    Xval, yval, _ = BSDS.list_load(which='val')
    print len(Xval), len(yval), len(_)

    print 'displaying a random sample from training set'
    idx = np.random.randint(0, len(Xtrain))
    plt.subplot(1,2,1)
    plt.imshow(Xtrain[idx].astype(np.uint8))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(ytrain[idx], cmap='gray')
    plt.axis('off')

    plt.show()
    print 'displaying a random sample from testing set'
    idx = np.random.randint(0, len(Xtest))
    plt.subplot(1,2,1)
    plt.imshow(Xtest[idx].astype(np.uint8))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(ytest[idx], cmap='gray')
    plt.axis('off')

    plt.show()
    print 'displaying a random sample from validation set'
    idx = np.random.randint(0, len(Xval))
    plt.subplot(1,2,1)
    plt.imshow(Xval[idx].astype(np.uint8))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(yval[idx], cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()

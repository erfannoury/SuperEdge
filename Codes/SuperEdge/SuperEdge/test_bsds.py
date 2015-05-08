import numpy as np
from bsds500 import BSDS
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    now = datetime.now()
    Xtrain, ytrain = BSDS.load(which='train')
    print Xtrain.shape, ytrain.shape
    Xtest, ytest = BSDS.load(which='test')
    print Xtest.shape, ytest.shape
    Xval, yval = BSDS.load(which='val')
    print Xval.shape, yval.shape
    print 'loading all data took ', (datetime.now() - now)

    fig = plt.figure(figsize=(14, 7))

    ax = fig.add_subplot(121)
    ax.imshow(Xtest[10,...].astype(np.uint8))
    ax.axis('off')

    ax = fig.add_subplot(122)
    ax.imshow(ytest[10,...], cmap='gray')

    plt.show()
if __name__ == '__main__':
    main()

import numpy as np
import cv2
from extract_feature import VGG16Extractor
from datetime import datetime
from bsds500 import BSDS
#from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy import io
from sklearn.externals import joblib
import cv2
import xgboost as xgb

def main():
    # load pre-trained model
    print 'loading pretrained model'
    model_addr = '../../../Models/XGBR_50_d10_r2.pkl'
    clf = joblib.load(model_addr)
        
    now = datetime.now()    
    print 'loading test dataset'
    Xtest, ytest, ntest = BSDS.list_load(which='test')
    vgg = VGG16Extractor()
    show_results = False
    save_results = True
    for i in xrange(15):
        tic = datetime.now()
        print 'Xtest.shape = ', Xtest[i].shape
        im = cv2.resize(Xtest[i], (2 * Xtest[i].shape[1], 2 * Xtest[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        print 'im.shape = ', im.shape
        hyperimage = vgg.transform(im)
        print str.format('%d - %s : ' % (i, ntest[i])), Xtest[i].shape
        
        ypred = clf.predict(hyperimage.reshape((hyperimage.shape[0] * hyperimage.shape[1], hyperimage.shape[2])))
        ypred = ypred.reshape((hyperimage.shape[0], hyperimage.shape[1]))
        ypred = (ypred - ypred.min()) / (ypred.max() - ypred.min())
        ypred = cv2.resize(ypred, (Xtest[i].shape[1], Xtest[i].shape[0]), interpolation=cv2.INTER_NEAREST)
        print 'ypred.shape = ', ypred.shape

        print 'single image prediction took: ', (datetime.now() - tic)
        if show_results:
            print 'displaying prediction result'
            plt.subplot(1,3,1)
            plt.imshow(Xtest[i].astype(np.uint8))
            plt.title(str.format('Image %s' % ntest[i]))
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(ytest[i], cmap='gray')
            plt.title(str.format('Ground truth %s' % ntest[i]))
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(ypred, cmap='gray')
            plt.title(str.format('Prediction %s' % ntest[i]))
            plt.axis('off')


            plt.show()
        if save_results:
            print 'saving prediction result'
            cv2.imwrite(str.format('../../../Datasets/BSDS500/result/%s.png' % ntest[i]), (ypred * 255))
            #cv2.imwrite(str.format('../../../Datasets/BSDS500/result/test_%d_gt.png' % i), (255 * ytest[i]))
            #io.savemat(str.format('../../../Datasets/BSDS500/result/%s.mat' % ntest[i]), {'ucm2': ypred}, do_compression=True, appendmat=False)

        hyperimage = None
    print 'predicting output for test set took ', (datetime.now() - now)


if __name__ == '__main__':
    main()

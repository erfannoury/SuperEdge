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
from scipy import misc
import cv2
import xgboost as xgb

def main():
    # load pre-trained model
    print 'loading pretrained model'
    model_addr = '../../../Models/XGBR_50_d10_r2.pkl'
    clf = joblib.load(model_addr)
    impath = 'C:/Users/Erfan/Pictures/nobody-cares.jpg'        
    im = misc.imread(impath).astype(dtype=np.float32)
    vgg = VGG16Extractor()
    show_results = True
    save_results = False

    tic = datetime.now()
    print 'im.shape = ', im.shape
    hyperimage = vgg.transform(im)
        
    ypred = clf.predict(hyperimage.reshape((hyperimage.shape[0] * hyperimage.shape[1], hyperimage.shape[2])))
    ypred = ypred.reshape((hyperimage.shape[0], hyperimage.shape[1]))
    ypred = (ypred - ypred.min()) / (ypred.max() - ypred.min())
    print 'ypred.shape = ', ypred.shape

    print 'single image prediction took: ', (datetime.now() - tic)
    if show_results:
        print 'displaying prediction result'
        plt.subplot(1,2,1)
        plt.imshow(im.astype(np.uint8))
        plt.title(str.format('Image'))
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(ypred, cmap='gray')
        plt.title(str.format('Prediction'))
        plt.axis('off')


        plt.show()

    hyperimage = None
    print 'predicting output for test set took ', (datetime.now() - now)


if __name__ == '__main__':
    main()

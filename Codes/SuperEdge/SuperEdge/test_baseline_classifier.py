import numpy as np
import cv2
from extract_feature import VGG16Extractor
from datetime import datetime
from bsds500 import BSDS
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from mpltools import style
from scipy import io
from sklearn.externals import joblib
style.use(['ggplot'])

def main():
    # load pre-trained model
    print 'loading pretrained model'
    model_addr = '../../../Models/linearSVC.pkl'
    svc = joblib.load(model_addr)
        
    now = datetime.now()    
    print 'loading test dataset'
    Xtest, ytest = BSDS.load(which='test')
    vgg = VGG16Extractor()
    show_results = False
    for i in xrange(Xtest.shape[0]):
        tic = datetime.now()
        hyperimage = vgg.transform(Xtest[i,...])
        ypred = np.zeros(ytest[0].shape, dtype=np.float32)
        print i, ' ', hyperimage.shape
        for y in xrange(hyperimage.shape[0]):
            for x in xrange(hyperimage.shape[1]):
                    ypred[y,x] = svc.predict(hyperimage[y,x,:])

        print 'single image prediction took: ', (datetime.now() - tic)
        if show_results:
            print 'displaying prediction result'
            plt.subplot(1,3,1)
            plt.imshow(Xtest[i,...])
            plt.title('Image')
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(ytest[i], cmap='gray')
            plt.title('Ground truth')
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(ypred, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')


            plt.show()
        else:
            print 'saving prediction result'
            cv2.imwrite(str.format('../../../Dataset/BSDS500/result/test_%d.png' % i), ypred)
            io.savemat(str.format('../../../Dataset/BSDS500/result/test_%d.mat' % i), {'pred': ypred}, do_compression=True)

        hyperimage = None
    print 'predicting output for test set took ', (datetime.now() - now)


if __name__ == '__main__':
    main()
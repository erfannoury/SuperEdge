import numpy as np
from extract_feature import VGG16Extractor
from datetime import datetime
import h5py
from scipy import io

def main():
    now = datetime.now()    
    vgg = VGG16Extractor(width=640, height=480, use_mean_pixel=False)
    nyud_images = '../../../Datasets/nyu_images.mat'
    nyud_images = h5py.File(nyud_images, 'r')
    images = nyud_images['images']
    for i in xrange(5):
        hyperimage = vgg.transform(images[i,:,:,:].transpose((2,1,0)))
        print i, ' ', hyperimage.shape
        f_name = str.format('../../../Datasets/NYUD/%d.mat' % i)
        io.savemat(f_name, {'feature': hyperimage})
        hyperimage = None
    nyud_images.close()
    print 'transforming the images set took ', (datetime.now() - now)



if __name__ == '__main__':
    main()
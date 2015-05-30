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
    for i in xrange(images.shape[0]):
        hyperdict = vgg.transform_unscaled(images[i,:,:,:].transpose((2,1,0)))
        print i
        f_name = str.format('../../../Datasets/NYUD/%d.mat' % i)
        io.savemat(f_name, hyperdict, do_compression=True)
        hyperimage = None
    nyud_images.close()
    print 'transforming the images set took ', (datetime.now() - now)



if __name__ == '__main__':
    main()
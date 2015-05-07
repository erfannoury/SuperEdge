import random
import numpy as np
from scipy import misc
import scipy.io as sio
import os

class BSDS(object):
    """
    This is a class for loading BSDS500 dataset
    """
    def __init__(self):
        pass

    @staticmethod
    def load(which='train'):
        """
        This will load the images and ground truth files for the specified subfolder.

        Parameters
        ----------
        which: str
                determine which set of data to load (train, test, val)

        Returns
        -------
        (X,y): tuple of ndarrays
               This tuple will contain the images and repective ground truth contours of images.
               X is of size (nb * 3 * width * height)
               y is of size (width * height)
               where width = 321 and height = 481
               ground truth contour map is created by averaging all of the human-annotated boundary maps

        """
        data_path = '../../../Datasets/BSDS500/data'
        images_path = os.path.join(data_path, 'images')
        gt_path = os.path.join(data_path, 'groundTruth')
        sub_folders = ['test', 'train', 'val']

        if which not in sub_folders:
            print 'defined folder is not in subfolders.'
            return
        
        nb = 200        # size of train/test set
        if which == sub_folders[-1]:
            nb = 100    # size of validation set

        nb_xchannels = 3    # number of images channels
        width = 321
        height = 481
        # using the bc01 ordering
        X = np.zeros((nb, nb_xchannels, width, height), dtype=np.float32)
        y = np.zeros((nb, width, height), dtype=np.float32)

        imdir = os.path.join(images_path, which)
        gtdir = os.path.join(gt_path, which)
        idx = -1
        for i, g in zip(os.listdir(imdir), os.listdir(gtdir)):
            idx += 1
            impath = os.path.join(imdir, i)
            gtpath = os.path.join(gtdir, g)
            img = misc.imread(impath).astype(dtype=np.float32)
            gt = sio.loadmat(gtpath)
            gt = gt["groundTruth"].flatten()
            bnds = [b["Boundaries"][0, 0] for b in gt]
            prob_bnd = np.zeros((321, 481), dtype=np.float32)
            if img.shape[0] == 321:
                X[idx,:,:,:] = img.transpose((2,0,1))
                for j in xrange(len(bnds)):
                    prob_bnd += bnds[j]
            else:
                X[idx,:,:,:] = img.transpose((2,1,0))
                for j in xrange(len(bnds)):
                    prob_bnd += bnds[j].transpose()
            y[idx,:,:] = prob_bnd / len(bnds)
        return (X, y)
import random
import numpy as np
from scipy import misc
import scipy.io as sio
import os


def load_subfolder(imdir, gtdir, validation=False):
    """
    This will load the images and ground truth files for the specified subfolder.

    Parameters
    ----------
    imdir: str
           address of the images folder
    gtdir: str
           address of the groundtruth files folder
    validation: bool
           since validation set's size is smaller (100 instead of 200 for train/test sets), this bool value indicates the validation folder

    Returns
    -------
    (X,y): tuple on ndarrays
           This tuple will contain the images and repective ground truth contours of images.
           X is of size (nb * 3 * width * height)
           y is of size (width * height)
           where width = 321 and height = 481
           ground truth contour map is created by averaging all of the human-marked boundary maps

    """

    nb = 200        # size of train/test set
    if validation:
        nb = 100    # size of validation set

    nb_xchannels = 3    # number of images channels
    width = 321
    height = 481

    # using the bc01 ordering
    X = np.zeros((nb, nb_xchannels, width, height), dtype=np.float32)
    y = np.zeros((nb, width, height), dtype=np.float32)
    idx = -1
    for i, g in zip(os.listdir(imdir), os.listdir(gtdir)):
        idx += 1

        impath = os.path.join(imdir, i)
        gtpath = os.path.join(gtdir, g)
        img = misc.imread(impath).astype(dtype=np.float32)
        gt = sio.loadmat(gtpath)
        gt = gt["groundTruth"].flatten()
        bnds = [b["Boundaries"][0, 0] for b in gt]

        if img.shape[0] == 321:
            X[idx, :, :, :] = img.transpose((2,0,1))
            for j in xrange(len(bnds)):
            	y += bnds[j]
        else:
            X[idx, :, :, :] = img.transpose((2,1,0))
            for j in xrange(len(bnds)):
            	y += bnds[j]
                
        y /= len(bnds)
    return (X, y)

def load_data(sub_folders):
    """
    This will load and return images and their respective contour images for the list of sub_folders provided.

    Parameters
    ----------
    sub_folders: list of str
                this list will contain the name of the subfolders that their data is to be loaded.
                The list should be a subset of this list: ['test', 'train', 'val']

    Returns
    -------
    data: list of tuples of form (X,y)
    """


    data_path = 'E:/University Central/IPLab/ELM Edge Detection/BSR/BSDS500/data'
    images_path = os.path.join(data_path, 'images')
    gt_path = os.path.join(data_path, 'groundTruth')
    def_sub_folders = ['test', 'train', 'val']

    data = []
    for sub_folder in sub_folders:
        if sub_folder not in def_sub_folders:
            raise Exception('Subfolder not found')
            return data
        impath = os.path.join(images_path, sub_folder)
        gtpath = os.path.join(gt_path, sub_folder)
        val = sub_folder == 'val'
        data.append(load_subfolder(impath, gtpath, validation=val))

    return data

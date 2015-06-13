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
		This will load the images and ground truth files for the specified subfolder. (RGB ordering)

		Parameters
		----------
		which: str
				determine which set of data to load (train, test, val)

		Returns
		-------
		(X,y,n): tuple of two ndarrays and a list
			   This tuple will contain the images and respective ground truth contours of images. It will also contain a list of image names.
			   X is of size (nb * width * height * 3)
			   y is of size (nb * width * height)
			   n is a list of size (nb)
			   where width = 480 and height = 320
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
		width = 320
		height = 480
		# don't need to use bc01 ordering, feature extractor will take care of the ordering
		X = np.zeros((nb, width, height, nb_xchannels), dtype=np.float32)
		y = np.zeros((nb, width, height), dtype=np.float32)
		n = []

		imdir = os.path.join(images_path, which)
		gtdir = os.path.join(gt_path, which)
		idx = -1
		for i in os.listdir(imdir):
			if i[-4:] == '.jpg':
				idx += 1
				imname = i[:-4]
				n.append(imname)
				impath = os.path.join(imdir, i)
				g = imname + '.mat'
				gtpath = os.path.join(gtdir, g)
				img = misc.imread(impath)[:-1,:-1].astype(dtype=np.float32)
				gt = sio.loadmat(gtpath)
				gt = gt["groundTruth"].flatten()
				bnds = [b["Boundaries"][0, 0][:-1,:-1] for b in gt]
				prob_bnd = np.zeros((320, 480), dtype=np.float32)
				if img.shape[0] == 320:
					X[idx,:,:,:] = img
				else:
					X[idx,:,:,:] = img.transpose((1,0,2))
				for j in xrange(len(bnds)):
					if bnds[j].shape[0] == 320 and bnds[j].shape[1] == 480:
						prob_bnd += bnds[j]
					else:
						prob_bnd += bnds[j].transpose()
				y[idx,:,:] = prob_bnd / len(bnds)
		return (X, y, n)

	@staticmethod
	def list_load(which='train'):
		"""
		This will load the images and ground truth files for the specified subfolder. (RGB ordering)

		Parameters
		----------
		which: str
				determine which set of data to load (train, test, val)

		Returns
		-------
		(X,y,n): tuple of three lists
			X: list of numpy.ndarray items
				First list contains the images.
			y: list of numpy.ndarray items
				Second list contains the ground truth contour image of the corresponding image.
			n: list of str
				Third list contains the names of the corresponding images.	   
		"""
		data_path = '../../../Datasets/BSDS500/data'
		images_path = os.path.join(data_path, 'images')
		gt_path = os.path.join(data_path, 'groundTruth')
		sub_folders = ['test', 'train', 'val']

		if which not in sub_folders:
			print 'defined folder is not in subfolders.'
			return
		
		# don't need to use bc01 ordering, feature extractor will take care of the ordering
		X = []
		y = []
		n = []

		imdir = os.path.join(images_path, which)
		gtdir = os.path.join(gt_path, which)
		idx = -1
		for i in os.listdir(imdir):
			if i[-4:] == '.jpg':
				idx += 1
				imname = i[:-4]
				n.append(imname)
				impath = os.path.join(imdir, i)
				g = imname + '.mat'
				gtpath = os.path.join(gtdir, g)
				img = misc.imread(impath).astype(dtype=np.float32)
				gt = sio.loadmat(gtpath)
				gt = gt["groundTruth"].flatten()
				bnds = [b["Boundaries"][0, 0] for b in gt]
				prob_bnd = np.zeros(bnds[0].shape, dtype=np.float32)
				X.append(img)
				for j in xrange(len(bnds)):
					prob_bnd += bnds[j]
				y.append(prob_bnd / len(bnds))
		return (X, y, n)
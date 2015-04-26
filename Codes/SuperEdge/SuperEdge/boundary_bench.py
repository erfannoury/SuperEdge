import os
import numpy as np
import scipy as sc
from skimage import io
import scipy.io as sio


def boundaryBench(img_dir, gt_dir, res_dir, out_dir, nthresh=99, max_dist=0.0075, thinpb=True):
    """
    Parameters
    ----------
    img_dir: str
        folder containing original images
    gt_dir: str
        folder containing ground truth data
    res_dir: str
        folder containing boundary detection results for all the images in img_dir
        Format can be one of the following:
            - a soft or hard boundary map in PNG format
            - a collection of segmentations in a cell 'segs' stored in a mat file
            - an ultrametric contour map in 'doubleSize' format, 'ucm2' stored in a mat file with values in [0 1]
    out_dir: str
        folder where evaluation results will be stored
    nthresh: int
        number of points in the precision/recall curve
    max_dist: float
        for computing precision/recall
    thinpb: bool
        option to apply morphological thinning on segmentation boundaries before benchmarking
    """
    images = [img for img in os.listdir(img_dir) if img.endswith('jpg')]
    for img in images:
        id = img[:-4]
        ev_file = os.path.join(out_dir, id + '_ev.txt')
        #ev = open(os.path.join(out_dir, evfile), 'w')
        res_img = os.path.join(res_dir, id + '.png')
        if os.path.exists(res_img):
            #img = io.imread(res_img)
            gt_file = os.path.join(gt_dir, id + '.mat')
            evaluate_boundary_image(res_img, gt_file, ev_file, nthresh, max_dist, thinpb)
        ev.close()
    collect_eval_boundary(out_dir)


def evaluate_boundary_image(res_file, gt_file, pr_file, nthresh=99, max_dist=0.0075, thinkpb=True):
    """
    Parameters
    ----------

    res_img: str
        path to the boundary detection result image
    gt_file: str
        path to the ground truth mat file
    pr_file: str
        path to the temporary output for this image
    nthresh: int
        number of points in the precision/recall curve
    max_dist: float
        for computing precision/recall
    thinpb: bool
        option to apply morphological thinning on segmentation boundaries before benchmarking
    
        Returns
        -------
        thresh: list
            list of threshold values
        cntR,sumR: tuple
            ratio gives recall
        cntP,sumP: tuple
            ratio gives precision
    """
    img = io.imread(res_file, True) *1.0 / 255

    gt = sio.loadmat(gt_file)
    gt = gt["groundTruth"].flatten()
    bnds = [b["Boundaries"][0, 0] for b in gt]
    
    # I will use ground truth boundaries for evaluation instead of the segmentations
    #segs = [s["Segmentation"][0, 0] for s in gt]

    thresh = None
    #if len(segs) <= 0:
    #    thresh = np.linspace(1.0/(nthresh+1), 1-1.0/(nthresh+1), nthresh)
    ## TODO: not sure about this
    #else:
    #    nthresh = len(segs)
    #    thresh = xrange(len(segs))

    thresh = np.linspace(1.0/(nthresh+1), 1-1.0/(nthresh+1), nthresh)

    cntR = [0] * nthresh
    sumR = [0] * nthresh
    cntP = [0] * nthresh
    sumP = [0] * nthresh

    for t in xrange(nthresh):
        bmap = img >= thresh[t]

        # TODO: do morphological thinning to make sure that boundaries are standard thickness (thin the bmap)

        # accumulate machine matches, since the machine pixels are allowed to match with any segmentation
        accP = np.zeros(*bmap.shape, dtype=bool)

        for bnd in bnds:
            match1, match2 = correspondPixels(bmap, bnd, max_dist)

            # accumulate machine matches
            accP = accP | match1.astype(bool)

            #computer recall
            sumR[t] += bnd.sum()
            cntR[t] += (match2 > 0).sum()
        
        # compute precision
        sumP[t] += bmap.sum()
        cntP[t] += accP.sum()

        # TODO return results


def collect_eval_boundary(out_dir):
    """
    Calculate P, R and F-measure from individual evaluation files
    """
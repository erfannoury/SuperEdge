import numpy as np
from datetime import datetime
from sklearn.datasets import dump_svmlight_file
import os.path as path

def main():
    cache_path = 'cachelarge/'
    feat_name = 'feat.dat'
    lbl_name = 'lbl.dat'
    feat_len = 4224 #1088
    now = datetime.now()
    lbl_memmap = np.memmap(path.join(cache_path, lbl_name), dtype='uint8', mode='r')
    feat_memmap = np.memmap(path.join(cache_path, feat_name), dtype='float32', mode='r', shape=(lbl_memmap.shape[0], feat_len))
    print 'loading dataset took ', (datetime.now() - now)
    now = datetime.now()
    print 'starting dumping feature files to libsvm format'
    libsvm_file = 'data.train.txt'
    dump_svmlight_file(feat_memmap, lbl_memmap, path.join(cache_path, libsvm_file))
    print 'dumping feature file took ', (datetime.now() - now)

if __name__ == '__main__':
    main()

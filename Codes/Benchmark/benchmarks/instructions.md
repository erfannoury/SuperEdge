It is assumed that MATLAB is currently in this directory.

Predictions can be in two formats.
1. Predictions as probability maps saved as `png` image files. 
2. Predictions as `mat` files according to the ucm2 convention.

Only the first method is covered in this instruction note for now. Also note that output file's name should be the same as the input image's name.

To run benchmark codes, addresses to original images, ground truth `mat` files and also output images should be known beforehand.

We assume that all predictions are put in `../data/png`. Also original images for the test set are in `../../../Datasets/BSDS500/data/images/test/` and their corresponding ground truth annotations are at `../../../Datasets/BSDS500/data/groundTruth/test/`. 
To save benchmark results, `../data/results` is selected.

To run the boundary benchmark code on predictions, assuming that you are still in the current folder, run this command in MATLAB:
```Matlab
boundaryBench("../../../Datasets/BSDS500/data/images/test/", "../../../Datasets/BSDS500/data/groundTruth/test/", "../data/png", "../data/results") 
```
This will run the boundary benchmark code with default settings for the _number of thresholds_ (default: 99), _max distance_ (default: 0.0075),  and _thinning_ (default: true) options.  

After running this command, three text files will be created in the results folder. Now using those files, you can run this command to obtain the OIS, ODS, and AP values and also the Precision-Recall curve plot.

```Matlab
plot_eval("../data/results")
```

## Overview

| Index | ODS  | ODS threshold | OIS  | AP   |
|-------|------|---------------|------|------|
| 1     | 0.64 | 0.57          | 0.67 | 0.42 |
| 2     | 0.64 | 0.42          | 0.66 | 0.50 |
| 3     | 0.66 | 0.41          | 0.71 | 0.62 |

## Model Descriptions and Charts
### 1. XGBRegressor (2015/06/02)
`XGBRegressor(max_depth=10, nthread=12)` trained on first 100 images using VGG CNN F.

![1.XGBRegressor](Result Charts/1.isoF.png)

### 2. XGBRegressor (2015/06/09)
`XGBRegressor(max_depth=10, nthread=12, min_child_weight=2)` trained on first 50 images using VGG CNN F with Poisson-disk sampling with radius 2.

![2.XGBRegressor](Result Charts/2.isoF.png)

### 3. XGBRegressor (2015/06/13)
`XGBRegressor(max_depth=10, nthread=12, min_child_weight=2)` trained on first 50 images using VGG CNN F with Poisson-disk sampling with radius 2. Classifier used in this test is exactly the same as the classifier used in test #2, though the testing setup is different. Deep features suffer from low spatial resolutions. To experiment with possible solutions to this issue, in this test setup, I upscaled test images by a factor of 2 and then extracted deep features for the upscaled image and classified pixels in the upscaled image. Afterwards, the classification output was downscaled to original image size and the result was saved. As is apparent from the results table, this method has been beneficial.

![3.XGBRegressor](Result Charts/3.isoF.png)

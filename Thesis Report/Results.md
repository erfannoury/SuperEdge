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
`XGBRegressor(max_depth=10, nthread=12, min_child_weight=2)` trained on first 50 images using VGG CNN F with Poisson-disk sampling with radius 2. The only difference with test #2 is that test images are upscaled (2x), classification is performed on the upscaled image and final result is downscaled to the original image size. This is done to increase spatial resolution of features, though this method isn't proved to produce the best results.

![3.XGBRegressor](Result Charts/3.isoF.png)
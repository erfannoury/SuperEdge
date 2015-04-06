1. Number of edge maps for images isn't constant.

    * For the test set, these are the results:

    |Number of contour maps| Number of images|
    |----------------------|-----------------|
    |4 | 1|
    |5 | 145|
    |6 | 47|
    |7 | 4|
    |8 | 3|

    * For the training set, there are the numbers:

    |Number of contour maps| Number of images|
    |-----------------|--------------|
    | 4 | 2|
    | 5 | 131|
    | 6 | 49 |
    | 7 | 15 |
    | 8 | 2 |

    * For the validation set, these are the numbers:

    | Number of contour maps | Number of images |
    |----------------|----------------|
    | 5 | 69 |
    | 6 | 18 |
    | 7 | 11 |
    | 8 | 2  |

    The question is how to handle the missing and extra contour maps?

2. How multiple contour maps are used in the training and testing phase? If I stick with 5 contour maps per image, I will lose lots of contour maps in the training phase (I will lose 85 images and I would have 2 redundant contour map images).

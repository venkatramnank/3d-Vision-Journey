# Introduction and starting off by scourging around resources (in Progress!!!)

The first post I read about introduction about 3D is here: https://towardsdatascience.com/intro-to-3d-deep-learning-e992f7efa6ee. 

## Difference between 3D and 2D data

2D data is essentially images. They are respresented as 1D or 2D matrices. 


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


```python
img = cv2.imread(r'/Users/venkatramnankalyanakumar/Desktop/3DVision/notebooks/dog.jpg')
```


```python
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x123d5d4b0>




    
![[https://github.com/venkatramnank/3d-Vision-Journey/blob/main/images/output_3_1.png](https://github.com/venkatramnank/3d-Vision-Journey/blob/main/_posts/output_3_1.png)
    



```python
img.shape # here (height, width, channels)
```




    (4919, 7375, 3)



But 3D data is reperesented differently. Some commonly used representations are: multi-view, volumetric, point cloud, mesh and volumetric display

### Multi view representation

This representation is captured by positioning multiple cameras that take photos from different angles of the same object or scene. 
ShapeNet is a good ecample of multi view representation.

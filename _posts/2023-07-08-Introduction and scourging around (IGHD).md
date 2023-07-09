# Introduction and starting off by scourging around resources

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


<img src="{{site.baseurl | prepend: site.url}}images/output_3_1.png" alt="Untitled" />



<img src="https://github.com/venkatramnank/3d-Vision-Journey/blob/main/_posts/images/dog.jpg" alt="Untitled" />


```python
img.shape # here (height, width, channels)
```




    (4919, 7375, 3)



But 3D data is reperesented differently. Some commonly used representations are: multi-view, volumetric, point cloud, mesh and volumetric display

### Multi view representation

This representation is captured by positioning multiple cameras that take photos from different angles of the same object or scene. 
ShapeNet is a good example of multi view representation.

### Point Cloud representation

This representation is commonly used (ask any roboticist, they love them). In this representation each image is represented by a set of points (x, y, z coordinates), which are collected from raw sensors. Point cloud data are typically captured by LiDAR sensors or converted from mesh data.

### Mesh

The standard building block for 3D modeling with programs like Blender, Autodesk Maya, Unreal Engine, etc. is a mesh. The mesh representation consists of a set of points as well as the relationship between these points (edges and faces), unlike a point cloud where every 3D object is made up of individual points. Polygon mesh, which has faces shaped like triangles or quads, is one sort of mesh.

### Volumetric display

In the volumetric representation, each image is solid and made of voxels: the 3D equivalent of pixels in 2D images.


# Some initial webistes to have a look at:
- 3D Tutorial CVPR: https://www.youtube.com/watch?v=8CenT_4HWyY
- Math : https://cse291-i.github.io/schedule.html
- Pytorch 3D: https://www.youtube.com/watch?v=MOBAJb5nJRI  and https://pytorch3d.org/



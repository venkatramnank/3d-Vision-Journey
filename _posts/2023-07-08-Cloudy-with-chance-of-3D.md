# Cloudy with a Chance of 3D: Unraveling the Mysteries of Point Clouds 
## In Progress

Righto !! This is my attempt to learn from various websites and resources (linked in references).

### What are point clouds?
“A Point Cloud is an unordered set of 3 dimensional points in a frame of reference (Cartesian coordinate system) on the surface of objects.”
```math
P = {(x_{i}, y_{i}, z_{i}) \ | \ i \in N}
```
Each point represents a single spatial measurement on the object's surface. Taken together, a point cloud represents the entire external surface of an object. Point clouds are obtained using 3D scanners (like LIDAR).

### Starting off with point clouds (using python of course, duh)

We need to open3d library.
```
$ pip install open3d
```

> **_NOTE:_**  If you run across the ImportError in open3d in a mac, use the following fix ``` brew install libomp ```.


```python
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import plotly.graph_objects as go
import plotly.express as px
```

Point clouds are represented as (n x 3), where n is the number of points. Let us randomly create a 7 point cloud.


```python
number_points = 7
# uniform distribution over [0, 1)
pcd = np.random.rand(number_points, 3)  
print(pcd)
```

    [[0.9865411  0.08488266 0.95627717]
     [0.25042176 0.46572193 0.44011125]
     [0.34306293 0.64666564 0.90510456]
     [0.67587728 0.33338794 0.22865515]
     [0.15539856 0.70407214 0.75977207]
     [0.65352276 0.2598785  0.07726434]
     [0.92681409 0.22140909 0.2261651 ]]



```python
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter3D(pcd[:, 0], pcd[:, 1], pcd[:, 2]) # x, y, z
# label the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Random Point Cloud")
# display:
plt.show()
```


    

<img src="{{site.baseurl | prepend: site.url}}images/output_4_0.png" alt="Untitled" />
    


Let us see some cool stuff with open3d. Let us import Armadillo 3D mesh from open3D.


```python
armadillo = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo.path)
```


```python
# Visualizing
mesh.compute_vertex_normals() # compute normals for vertices or faces
o3d.visualization.draw_geometries([mesh])
```

Looks like some weird Armadillo man. Well, whatever. Wait, it looks like a **Pokemon** !!!



<img src="{{site.baseurl | prepend: site.url}}images/armadillo3d.png" alt="Untitled" />

Let us see what this mesh variable actually looks like.


```python
type(mesh)
```




    open3d.cpu.pybind.geometry.TriangleMesh



This mesh can be saved in .ply format. 


```python
pcd = mesh.sample_points_uniformly(number_of_points=1000) #Sampling 1000 points from the mesh
o3d.io.write_point_cloud("armadillo_pcd.ply", pcd)
```




    True



Now we can load this using the following code:


```python
pcd_o3d = o3d.io.read_point_cloud("armadillo_pcd.ply")
```


```python
type(pcd_o3d)
```




    open3d.cpu.pybind.geometry.PointCloud



open3d.cpu.pybind.geometry.PointCloud is a Point cloud class. It has multiple useful functions in it, which can be explored here: http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html


```python
pcd_np = np.asarray(pcd_o3d.points) # Converting to numpy array
```


```python
pcd_np.shape # n x 3, here n is 1000 points 
```




    (1000, 3)




```python
# Display using matplotlib:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter3D(pcd_np[:, 0], pcd_np[:, 2], pcd_np[:, 1])
# label the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Armadillo man Point Cloud")
# display:
plt.show()
```


    


<img src="{{site.baseurl | prepend: site.url}}images/output_20_0.png" alt="Untitled" />

## A very famous dataset

A much commonly known dataset is http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip. This dataset contains 9 classes of data including bathtub, bed ,etc. This dataset is used for building deep learning models. Well let us just say, this is common like the MNIST dataset. **But**, the format for each file is **.off** file.

(From Wikipedia) OFF (Object File Format) is a geometry definition file format containing the description of the composing polygons of a geometric object. It can store 2D or 3D objects, and simple extensions allow it to represent higher-dimensional objects as well.Though originally developed for Geomview, a geometry visualization software, other software has adapted the simple standard.

Let us see one of these files. 
> **__Note (shamelessness alert)__**: the code is shamelessly stolen from : https://github.com/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb


```python
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

with open("bathtub_0001.off", 'r') as f:
    vert, faces = read_off(f)

```

In 3D data, vertices and faces are fundamental components used to represent the geometry and structure of objects or surfaces.

Vertices are individual points in three-dimensional space. They define the corners or endpoints of geometric shapes, such as triangles, polygons, or more complex surfaces. Each vertex has coordinates (x, y, z) that specify its position in the 3D space.

Faces are flat polygons formed by connecting three or more vertices. They represent the surface of an object or the boundary between different regions. Faces are typically defined by a collection of vertices connected in a specific order, often forming triangles (three-sided faces) or quadrilaterals (four-sided faces). These polygons can be further combined to create more complex surfaces or objects.

In 3D models or point clouds, vertices and faces work together to define the shape, structure, and surface of the represented object. The vertices provide the spatial coordinates, while the faces connect the vertices to form the surface topology. Together, they enable the rendering, visualization, and analysis of 3D data.


```python
i,j,k = np.array(faces).T
x,y,z = np.array(vert).T
```

Here is a pretty cool visualization function


```python
def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig
```


```python

visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50, i=i,j=j,k=k)]).show()
```


```python

visualize_rotate([go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers')]).show()
```

### References :
- https://betterprogramming.pub/introduction-to-point-cloud-processing-dbda9b167534


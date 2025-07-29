import torch
import trimesh
import numpy as np
from e3nn.o3 import rand_matrix
from torch_cluster import knn_graph
from recon.lib.segmentator import segment_point

# mesh = trimesh.creation.box(extents=(1, 1, 1)) # equivariance
mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=5, sections=100) # no equivariance for smooth object
# mesh_path = "/data/lab-lei.jiabao/ShapeNet_GT/ShapenetV1_tpami/03001627/183974726abab0454aa6191ddaf4b676/mesh_gt_simplified.ply"
# mesh = trimesh.load_mesh(mesh_path, process=False) # equivariance

points, face_index = trimesh.sample.sample_surface(mesh, count=10000)
normals = mesh.face_normals[face_index]

points_before = torch.from_numpy(points).float()
normals_before = torch.from_numpy(normals).float()
edges_before = knn_graph(points_before, k=50).T
index_before = segment_point(points_before, normals_before, edges_before, kThresh=0.01, segMinVerts=20)
color_table = torch.randint(0, 256, size=(1 + index_before.max(), 3))
np.savetxt("result_before.txt", torch.cat([points_before, color_table[index_before]], dim=1).numpy())

R = rand_matrix()
points_after = points_before @ R
normals_after = normals_before @ R
edges_after = knn_graph(points_after, k=50).T
index_after = segment_point(points_after, normals_after, edges_after, kThresh=0.01, segMinVerts=20)
assert index_after.max() == index_before.max()
np.savetxt("result_after.txt", torch.cat([points_after @ R.T, color_table[index_after]], dim=1).numpy())

# compare num of each
unique_val_before, cnt_before = torch.unique(index_before, return_counts=True)
unique_val_after,  cnt_after  = torch.unique(index_after,  return_counts=True)

mismatch = torch.sort(cnt_before).values == torch.sort(cnt_after).values # number distribution should be the same
print(torch.where(mismatch==False))
if mismatch.all():
    print("has equivariance!") 


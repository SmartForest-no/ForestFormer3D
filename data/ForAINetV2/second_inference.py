import os
import numpy as np
from plyfile import PlyData, PlyElement
def read_ply(filename):
    """Read a PLY file and return its contents as a dictionary."""
    ply_data = PlyData.read(filename)
    data = ply_data['vertex'].data
    return {key: data[key] for key in data.dtype.names}


input_file = "/workspace/work_dirs/oneformer3d_outputfolder/BlueCat_RN_merged_trees_test_final_results.ply"
output_file = "/workspace/work_dirs/oneformer3d_outputfolder/BlueCat_RN_filtered_results.ply"

"""Filter points with instance_pred = -1 and save to a new PLY file."""
# Read input PLY file
pcd = read_ply(input_file)
points = np.vstack((pcd['x'], pcd['y'], pcd['z'])).astype(np.float32).T
semantic_seg = pcd["semantic_gt"].astype(np.int64)
treeID = pcd["instance_gt"].astype(np.int64)
instance_pred = pcd["instance_pred"].astype(np.int64)

# Filter points where instance_pred = -1
mask = (instance_pred == -1)
points_filtered = points[mask]
semantic_seg_filtered = semantic_seg[mask]
treeID_filtered = treeID[mask]
instance_pred_filtered = instance_pred[mask]

# Define output data type
dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('semantic_seg', 'i4'), ('treeID', 'i4')]

# Prepare vertex data
vertex = np.array(
    [tuple(points_filtered[i]) + (semantic_seg_filtered[i], treeID_filtered[i])
        for i in range(points_filtered.shape[0])],
    dtype=dtype
)

# Create output directory if it doesn't exist
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

# Write to new PLY file
el = PlyElement.describe(vertex, 'vertex')
PlyData([el], text=True).write(output_file)
print(f"Filtered PLY file saved to: {output_file}")
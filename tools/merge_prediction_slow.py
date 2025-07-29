import numpy as np
from plyfile import PlyData, PlyElement
import os
from scipy.spatial import cKDTree
import laspy

def load_ply(file_path):
    plydata = PlyData.read(file_path)
    points = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
    
    if {'semantic_pred', 'instance_pred', 'score', 'semantic_gt', 'instance_gt'}.issubset(plydata['vertex'].data.dtype.names):
        labels = np.vstack((
            plydata['vertex']['semantic_pred'],
            plydata['vertex']['instance_pred'],
            plydata['vertex']['score'],
            plydata['vertex']['semantic_gt'],
            plydata['vertex']['instance_gt']
        )).T
    
    elif {'semantic_pred', 'semantic_seg', 'treeID'}.issubset(plydata['vertex'].data.dtype.names):
        semantic_pred = plydata['vertex']['semantic_pred']
        semantic_seg = plydata['vertex']['semantic_seg'] - 1
        treeID = plydata['vertex']['treeID']
        instance_pred = np.full_like(semantic_pred, -1)
        score = np.zeros_like(semantic_pred, dtype=np.float32)
        labels = np.column_stack((
            semantic_pred,
            instance_pred,
            score,
            semantic_seg,
            treeID
        ))
    
    else:
        raise ValueError("Unsupported PLY format")
    return points, labels

def save_ply(file_path, points, labels):
    dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
             ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('score', 'f4'),
             ('semantic_gt', 'i4'), ('instance_gt', 'i4')]
    vertex = np.array([tuple(points[i]) + tuple(labels[i]) for i in range(points.shape[0])], dtype=dtype)
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(file_path)

def save_las(file_path, points, labels, offset):
    header = laspy.LasHeader(version="1.2", point_format=3)

    header.offsets = offset

    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    las.add_extra_dim(laspy.ExtraBytesParams(name="semantic_pred", type=np.int32))
    las.semantic_pred = labels[:, 0]

    las.add_extra_dim(laspy.ExtraBytesParams(name="instance_pred", type=np.int32))
    las.instance_pred = labels[:, 1]

    las.add_extra_dim(laspy.ExtraBytesParams(name="score", type=np.float32))
    las.score = labels[:, 2]

    las.add_extra_dim(laspy.ExtraBytesParams(name="semantic_gt", type=np.int32))
    las.semantic_gt = labels[:, 3]

    las.add_extra_dim(laspy.ExtraBytesParams(name="instance_gt", type=np.int32))
    las.instance_gt = labels[:, 4]

    las.write(file_path)

def downsample_ground_points(ground_points, voxel_size=0.5):
    voxel_coords = np.floor(ground_points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
    return ground_points[unique_idx]

def compute_instance_values(points, semantic_labels, instance_labels):

    ground_mask = (semantic_labels == 0)
    ground_points = points[ground_mask]

    if ground_points.shape[0] > 100000:
        ground_points = downsample_ground_points(ground_points, voxel_size=0.5)


    if ground_points.shape[0] > 0:
        ground_tree = cKDTree(ground_points)
    else:
        ground_tree = None  

    instance_values = {}

    instance_ids = np.unique(instance_labels[instance_labels >= 0])  
    
    for instance_id in instance_ids:
        instance_mask = (instance_labels == instance_id)
        instance_points = points[instance_mask]

        voxel_coords = np.floor(instance_points / 0.2).astype(np.int32)  
        volume = len(np.unique(voxel_coords, axis=0)) 

        if ground_tree is not None:
            lowest_point = np.min(instance_points[:, 2])  
            _, nearest_ground_idx = ground_tree.query([[np.mean(instance_points[:, 0]), 
                                                        np.mean(instance_points[:, 1]), 
                                                        lowest_point]], k=1)
            H_min = lowest_point - ground_points[nearest_ground_idx[0], 2]
        else:
            H_min = np.nan  
        
        instance_values[instance_id] = volume / H_min if H_min > 0 else 1000000

    return instance_values


def process_and_save_ply(points, labels, output_file):
    semantic_pred = labels[:, 0]
    instance_pred = labels[:, 1]

    instance_values = compute_instance_values(points, semantic_pred, instance_pred)

    instance_scores = np.full(instance_pred.shape, 1000000, dtype=np.float32)
    for instance_id, value in instance_values.items():
        instance_scores[instance_pred == instance_id] = value

    vertex_data = np.empty(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('instance_score', 'f4')
    ])
    
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    vertex_data['semantic_pred'] = semantic_pred
    vertex_data['instance_pred'] = instance_pred
    vertex_data['instance_score'] = instance_scores

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=False).write(output_file)
    print(f"Saved processed PLY: {output_file}")

    return instance_scores

def main(scan_name, output_dir, iterations):
    print(f"▶️ Starting merge process for scan: {scan_name} with {iterations} iterations")
    
    base_file = os.path.join(output_dir, f"{scan_name}_1.ply")

    print(f"🔹 Loading base file: {base_file}")
    base_points, base_labels = load_ply(base_file)
    all_points = base_points.copy()
    all_labels = base_labels.copy()
    
    instance_offset = base_labels[:, 1].max() + 1  # 

    #cumulative_offset = np.zeros(3, dtype=np.float32)
    
    #existing_coords = set(tuple(np.round(p, 5)) for p in all_points)

    for i in range(1, iterations+1):
        print(f"\n=== 🔄 Iteration {i}/{iterations} ===")
        if i > 1:
            pred_file = os.path.join(output_dir, f"{scan_name}_{i}.ply")

            if os.path.exists(pred_file): 
                print(f"📥 Loading prediction: {pred_file}")
                pred_points, pred_labels = load_ply(pred_file)
                
                pred_labels[:, 1] += instance_offset
                instance_offset = pred_labels[:, 1].max() + 1
                
                all_points = np.vstack((all_points, pred_points))
                all_labels = np.vstack((all_labels, pred_labels))
            else:
                print(f"Warning: Prediction file {pred_file} not found, skipping.")
        
        blue_file = os.path.join(output_dir, f"{scan_name}_bluepoints_{i}.ply")

        final_all_points = all_points
        final_all_labels = all_labels
        if os.path.exists(blue_file):
            print(f"🔹 Adding bluepoints: {blue_file}")
            blue_points, blue_labels = load_ply(blue_file)
            final_all_points = np.vstack((all_points, blue_points))
            final_all_labels = np.vstack((all_labels, blue_labels))
        else:
            print(f"Warning: Bluepoints file {blue_file} not found, skipping.")
        
        
        offset_path = os.path.join('/workspace/data/ForAINetV2/forainetv2_instance_data', scan_name + '_offsets.npy')
        offsets = np.load(offset_path)
        #final_all_points[:, 0] += offsets[0]
        #final_all_points[:, 1] += offsets[1]
        #final_all_points[:, 2] += offsets[2]
        #cumulative_offset += offsets

        output_dir_round = os.path.join(output_dir, f"round_{i}")
        os.makedirs(output_dir_round, exist_ok=True)
        output_path = os.path.join(output_dir_round, f"{scan_name}_round{i}.ply")
        print(f"Saving merged PLY (before filtering): {output_path}")
        save_ply(output_path, final_all_points, final_all_labels)

        output_dir_round = os.path.join(output_dir, f"round_{i}_noisy_score")
        os.makedirs(output_dir_round, exist_ok=True)
        file_path = os.path.join(output_dir_round, f"{scan_name}_noisysegments.ply")
        print(f"Computing instance score & saving noisysegments: {file_path}")
        instance_scores = process_and_save_ply(final_all_points, final_all_labels, file_path)
        threshold = 200
        print(f"Removing instances with score < {threshold}")
        mask_low_score = (instance_scores < threshold)
        final_all_labels[mask_low_score,1] = -1
        output_dir_round = os.path.join(output_dir, f"round_{i}_after_remove_noise_{threshold}")
        os.makedirs(output_dir_round, exist_ok=True)
        output_path = os.path.join(output_dir_round, f"{scan_name}_round{i}.ply")
        print(f"Saving filtered PLY: {output_path}")
        save_ply(output_path, final_all_points, final_all_labels)

        output_dir_round = os.path.join(output_dir, f"round_{i}")
        output_path = os.path.join(output_dir_round, f"{scan_name}_round{i}.las")
        final_all_points = final_all_points.astype(np.float64)
        final_all_points[:, 0] += offsets[0]
        final_all_points[:, 1] += offsets[1]
        final_all_points[:, 2] += offsets[2]

        las_offset = np.floor(offsets)
        print(f"Saving LAS file: {output_path}")
        save_las(output_path, final_all_points, final_all_labels, las_offset)
 
if __name__ == '__main__':
    import sys
    scan_name = sys.argv[1]
    output_dir = sys.argv[2]
    iterations = int(sys.argv[3])
    main(scan_name, output_dir, iterations)
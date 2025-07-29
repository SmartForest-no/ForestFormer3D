# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Load FOR-instance dataset with ground truth labels for semantic and
instance segmentations."""
import argparse
import inspect
import json
import os

import numpy as np
from plyutils import read_ply

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i][
                'objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def extract_bbox(mesh_vertices, label_ids, instance_ids, bg_sem=np.array([0])):
    # Filter out background points
    valid_mask = ~np.isin(label_ids, bg_sem)
    mesh_vertices = mesh_vertices[valid_mask]
    instance_ids = instance_ids[valid_mask]
    label_ids = label_ids[valid_mask]

    # Get the number of unique instances
    unique_instance_ids = np.unique(instance_ids)
    num_instances = len(unique_instance_ids)

    # Initialize instance_bboxes
    instance_bboxes = np.zeros((num_instances, 7))

    for i, instance_id in enumerate(unique_instance_ids):
        # Select points corresponding to the current instance
        mask = instance_ids == instance_id
        pts = mesh_vertices[mask, :3]

        if pts.shape[0] == 0:
            continue

        # Calculate min_pts, max_pts, locations, and dimensions
        min_pts = pts.min(axis=0)
        max_pts = pts.max(axis=0)
        locations = (min_pts + max_pts) / 2
        dimensions = max_pts - min_pts

        # Store the results in instance_bboxes
        instance_bboxes[i, :3] = locations
        instance_bboxes[i, 3:6] = dimensions
        instance_bboxes[i, 6] = 1

    return instance_bboxes


def export(ply_file,
           output_file=None,
           test_mode=False):
    """Export original files to vert, ins_label, sem_label and bbox file.

    Args:
        ply_file (str): Path of the ply_file.
        output_file (str): Path of the output folder.
            Default: None.
        test_mode (bool): Whether is generating test data without labels.
            Default: False.

    It returns a tuple, which contains the the following things:
        np.ndarray: Vertices of points data.
        np.ndarray: Indexes of label.
        np.ndarray: Indexes of instance.
        np.ndarray: Instance bboxes.
        dict: Map from object_id to label_id.
    """

    #from plyfile import PlyData, PlyElement
    #def read_ply(filename):
    #    """Read a PLY file and return its contents as a dictionary."""
    #    ply_data = PlyData.read(filename)
    #    data = ply_data['vertex'].data
    #    return {key: data[key] for key in data.dtype.names}

    pcd = read_ply(ply_file)
    #points = np.vstack((pcd['x'], pcd['y'], pcd['z'])).astype(np.float32).T

    points = np.vstack((pcd['x'], pcd['y'], pcd['z'])).astype(np.float64).T

    is_blue = 'bluepoints' in os.path.basename(ply_file)

    if is_blue:
        offsets = np.zeros(3, dtype=np.float64)
        print("-------------------has blue point-----------------------")
    else:
        print("-------------------no blue point-----------------------")
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])
        min_z = np.min(points[:, 2])
        offsets = np.array([mean_x, mean_y, min_z], dtype=np.float64)

        points[:, 0] -= mean_x
        points[:, 1] -= mean_y
        points[:, 2] -= min_z

    points = points.astype(np.float32)

    #semantic_seg = np.ones((points.shape[0],), dtype=np.int64)
    #treeID = np.zeros((points.shape[0],), dtype=np.int64)
    semantic_seg = pcd["semantic_seg"].astype(np.int64)
    treeID = pcd["treeID"].astype(np.int64)
    #semantic_seg = pcd["semantic"].astype(np.int64)
    #treeID = pcd["instance"].astype(np.int64)

    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    # perform global alignment of mesh vertices
    pts = np.ones((points.shape[0], 4))
    pts[:, 0:3] = points[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = pts[:, 0:3]

    # Load semantic and instance labels
    if not test_mode:
        # semantic label
        bg_sem=np.array([0])   #####wythan wood np.array([1])
        label_ids = semantic_seg - 1
        instance_ids = treeID  # 0: unannotated

        # Set instance_ids of background points to -1
        instance_ids[np.isin(label_ids, bg_sem)] = -1
    

        '''
        # Get unique instance IDs that are not -1
        unique_instance_ids = np.unique(instance_ids[instance_ids != -1])
    
        # Create a mapping from old instance IDs to new instance IDs
        new_instance_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_instance_ids,start=1)} #####wythan wood start=0)}
    
        # Update instance_ids with new instance IDs
        new_instance_ids = np.zeros_like(instance_ids)
        for old_id, new_id in new_instance_id_map.items():
            new_instance_ids[instance_ids == old_id] = new_id
        # Set background points back to 0
        new_instance_ids[instance_ids == -1] = 0
        instance_ids = new_instance_ids
        '''
        # Create a mask for non-background points
        valid_mask = instance_ids != -1

        # Create a new array for instance IDs, initialized to 0
        new_instance_ids = np.zeros_like(instance_ids)

        # Keep the original instance IDs without making them continuous
        new_instance_ids[valid_mask] = instance_ids[valid_mask]

        # Set background points back to 0
        new_instance_ids[instance_ids == -1] = 0

        # Assign the result back to instance_ids
        instance_ids = new_instance_ids

        unaligned_bboxes = extract_bbox(points, label_ids, instance_ids, bg_sem)
        aligned_bboxes = extract_bbox(aligned_mesh_vertices, label_ids, instance_ids, bg_sem)
    else:
        label_ids = None
        instance_ids = None
        unaligned_bboxes = None
        aligned_bboxes = None

    if output_file is not None:
        np.save(output_file + '_vert.npy', points)
        if not test_mode:
            np.save(output_file + '_sem_label.npy', label_ids)
            np.save(output_file + '_ins_label.npy', instance_ids)
            np.save(output_file + '_unaligned_bbox.npy', unaligned_bboxes)
            np.save(output_file + '_aligned_bbox.npy', aligned_bboxes)
            np.save(output_file + '_axis_align_matrix.npy', axis_align_matrix)

    return points, label_ids, instance_ids, unaligned_bboxes, \
        aligned_bboxes, axis_align_matrix, offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scan_path',
        required=True,
        help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument(
        '--label_map_file',
        required=True,
        help='path to scannetv2-labels.combined.tsv')
    parser.add_argument(
        '--scannet200',
        action='store_true',
        help='Use it for scannet200 mapping')

    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path,
                            scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(
        opt.scan_path, scan_name +
        '.txt')  # includes axisAlignment info for the train set scans.
    export(mesh_file, agg_file, seg_file, meta_file, opt.label_map_file,
           opt.output_file, scannet200=opt.scannet200)


if __name__ == '__main__':
    main()

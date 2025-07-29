# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batch mode in loading FOR-instance dataset with ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_ForAINetV2_data.py
"""
import argparse
import datetime
import os
from os import path as osp

import torch
import segmentator
import open3d as o3d
import numpy as np
from load_forainetv2_data import export
from plyutils import read_ply
from scipy.spatial import Delaunay

DONOTCARE_CLASS_IDS = np.array([])

FORAINETV2_OBJ_CLASS_IDS = np.array(
    [1])

def create_ply_with_superpoints(points, superpoints, filename):
    # Combine points and superpoints into a single array
    points_with_superpoints = np.hstack((points, superpoints[:, np.newaxis]))

    # Define the ply header
    header = f"""ply
            format ascii 1.0
            element vertex {points.shape[0]}
            property float x
            property float y
            property float z
            property float superpoint
            end_header
            """
    # Open file and write header
    with open(filename, 'w') as f:
        f.write(header)
        # Write points and superpoints
        for point, sp in zip(points, superpoints):
            f.write(f"{point[0]} {point[1]} {point[2]} {sp}\n")
    
    print(f"Point cloud saved to {filename}")

def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    forainetv2_dir,
                    test_mode=False):
    ply_file = osp.join(forainetv2_dir, scan_name + '.ply')
    mesh_vertices, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, axis_align_matrix, offsets = export(
            ply_file, None, test_mode)

    if not test_mode:
        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')

        OBJ_CLASS_IDS = FORAINETV2_OBJ_CLASS_IDS

        bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f'Num of care instances: {unaligned_bboxes.shape[0]}')

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]
    
    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)
    np.save(f'{output_filename_prefix}_offsets.npy', offsets)

    if not test_mode:
        #assert superpoints.shape == semantic_labels.shape
        np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
        np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
        np.save(f'{output_filename_prefix}_unaligned_bbox.npy',
                unaligned_bboxes)
        np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)
        np.save(f'{output_filename_prefix}_axis_align_matrix.npy',
                axis_align_matrix)
    

def batch_export(max_num_point,
                 output_folder,
                 scan_names_file,
                 forainetv2_dir,
                 test_mode=False
                 ):
    if test_mode and not os.path.exists(forainetv2_dir):
        # test data preparation is optional
        return
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    scan_names = [line.rstrip() for line in open(scan_names_file)]
    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        try:
            export_one_scan(scan_name, output_filename_prefix, max_num_point,
                            forainetv2_dir, test_mode)
        except Exception:
            print(f'Failed export scan: {scan_name}')
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='./forainetv2_instance_data',
        help='output folder of the result.')
    parser.add_argument(
        '--train_forainetv2_dir', default='train_val_data', help='forainetv2 data directory.')
    parser.add_argument(
        '--test_forainetv2_dir',
        default='test_data',
        help='forainetv2 data directory.')
    parser.add_argument(
        '--train_scan_names_file',
        default='meta_data/train_list.txt',
        help='The path of the file that stores the train scan names.')
    parser.add_argument(
        '--val_scan_names_file',
        default='meta_data/val_list.txt',
        help='The path of the file that stores the val scan names.')
    parser.add_argument(
        '--test_scan_names_file',
        default='meta_data/test_list.txt',
        help='The path of the file that stores the test scan names.')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.train_scan_names_file,
        args.train_forainetv2_dir,
        test_mode=False
        )
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.val_scan_names_file,
        args.train_forainetv2_dir,
        test_mode=False
        )
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.test_scan_names_file,
        args.test_forainetv2_dir,
        test_mode=False
        )


if __name__ == '__main__':
    main()

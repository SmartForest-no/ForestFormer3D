from os import path as osp
import numpy as np
import random

from mmdet3d.datasets.scannet_dataset import ScanNetSegDataset, ScanNetDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class ForAINetV2SegDataset_(ScanNetDataset):
    """We just add super_pts_path."""
    METAINFO = {
        'classes':
        ('ground','wood','leaf'),
        'palette': [[0, 255, 0],[0, 0, 255], [0, 255, 255]],
        'seg_valid_class_ids':
        (0, 1, 2),
        'seg_all_class_ids':
        (0, 1, 2)  # possibly with 'stair' class
    }

    def get_scene_idxs(self, *args, **kwargs):
        """Compute scene_idxs for data sampling."""
        return np.arange(len(self)).astype(np.int32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        #info['super_pts_path'] = osp.join(
        #    self.data_prefix.get('sp_pts_mask', ''), info['super_pts_path'])

        info = super().parse_data_info(info)

        return info
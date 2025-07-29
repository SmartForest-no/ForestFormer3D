import numpy as np
import scipy
import torch
from torch_scatter import scatter_mean
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import PointSample

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ElasticTransfrom(BaseTransform):
    """Apply elastic augmentation to a 3D scene. Required Keys:

    Args:
        gran (List[float]): Size of the noise grid (in same scale[m/cm]
            as the voxel grid).
        mag (List[float]): Noise multiplier.
        voxel_size (float): Voxel size.
        p (float): probability of applying this transform.
    """

    def __init__(self, gran, mag, voxel_size, p=1.0):
        self.gran = gran
        self.mag = mag
        self.voxel_size = voxel_size
        self.p = p

    def transform(self, input_dict):
        """Private function-wrapper for elastic transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after elastic, 'points' is updated
            in the result dict.
        """
        coords = input_dict['points'].tensor[:, :3].numpy() / self.voxel_size
        if np.random.rand() < self.p:
            coords = self.elastic(coords, self.gran[0], self.mag[0])
            coords = self.elastic(coords, self.gran[1], self.mag[1])
        input_dict['elastic_coords'] = coords
        return input_dict

    def elastic(self, x, gran, mag):
        """Private function for elastic transform to a points.

        Args:
            x (ndarray): Point cloud.
            gran (List[float]): Size of the noise grid (in same scale[m/cm]
                as the voxel grid).
            mag: (List[float]): Noise multiplier.
        
        Returns:
            dict: Results after elastic, 'points' is updated
                in the result dict.
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        noise_dim = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(noise_dim[0], noise_dim[1],
                            noise_dim[2]).astype('float32') for _ in range(3)
        ]

        for blur in [blur0, blur1, blur2, blur0, blur1, blur2]:
            noise = [
                scipy.ndimage.filters.convolve(
                    n, blur, mode='constant', cval=0) for n in noise
            ]

        ax = [
            np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in noise_dim
        ]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0) for n in noise
        ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * mag

@TRANSFORMS.register_module()
class AddSuperPointAnnotations(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        assert idxs[0] == -1

        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1
        pts_instance_mask = mapping[pts_instance_mask]
        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()


        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()
        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        num_stuff_cls = len(self.stuff_classes)
        insts = new_idxs[1:] - 1
        if self.merge_non_stuff_cls:
            gt_labels = insts.new_zeros(len(insts) + num_stuff_cls + 1)
        else:
            gt_labels = insts.new_zeros(len(insts) + self.num_classes + 1)

        for inst in insts:
            index = pts_semantic_mask[pts_instance_mask == inst][0]
            gt_labels[inst] = index - num_stuff_cls
        
        input_dict['gt_labels_3d'] = gt_labels.numpy()

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        sem_mask = torch.nn.functional.one_hot(sem_mask, 
                                    num_classes=self.num_classes + 1)
       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
        
        if self.merge_non_stuff_cls:
            sp_masks_seg = torch.vstack((
                sp_masks_seg[:num_stuff_cls, :], 
                sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))
        
        sp_masks_all = torch.vstack((sp_masks_inst, sp_masks_seg))

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict


@TRANSFORMS.register_module()
class SwapChairAndFloor(BaseTransform):
    """Swap two categories for ScanNet200 dataset. It is convenient for
    panoptic evaluation. After this swap first two categories are
    `stuff` and other 198 are `thing`.
    """
    def transform(self, input_dict):
        """Private function-wrapper for swap transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after swap, 'pts_semantic_mask' is updated
                in the result dict.
        """
        mask = input_dict['pts_semantic_mask'].copy()
        mask[input_dict['pts_semantic_mask'] == 2] = 3
        mask[input_dict['pts_semantic_mask'] == 3] = 2
        input_dict['pts_semantic_mask'] = mask
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_semantic_mask'] = mask
        return input_dict


@TRANSFORMS.register_module()
class PointInstClassMapping_(BaseTransform):
    """Delete instances from non-instaces classes.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Added Keys:
    - gt_labels_3d (int)

    Args:
        num_classes (int): Number of classes.
    """

    def __init__(self, num_classes, structured3d=False):
        self.num_classes = num_classes
        self.structured3d = structured3d

    def transform(self, input_dict):
        """Private function for deleting 
            instances from non-instaces classes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            are updated in the result dict. 'gt_labels_3d' is added.
        """

        # because pts_instance_mask contains instances from non-instaces 
        # classes
        pts_instance_mask = np.array(input_dict['pts_instance_mask'])
        pts_semantic_mask = input_dict['pts_semantic_mask']

        if self.structured3d:
            # wall as one instance
            pts_instance_mask[pts_semantic_mask == 0] = \
                pts_instance_mask.max() + 1
            # floor as one instance
            pts_instance_mask[pts_semantic_mask == 1] = \
                pts_instance_mask.max() + 1
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        pts_semantic_mask[pts_semantic_mask == self.num_classes] = -1

        idxs = np.unique(pts_instance_mask)
        mapping = np.zeros(np.max(idxs) + 2, dtype=int)
        new_idxs = np.arange(len(idxs))
        if idxs[0] == -1:
            mapping[idxs] = new_idxs - 1
            new_idxs = new_idxs[:-1]
        else:
            mapping[idxs] = new_idxs
        pts_instance_mask = mapping[pts_instance_mask]

        input_dict['pts_instance_mask'] = pts_instance_mask
        input_dict['pts_semantic_mask'] = pts_semantic_mask

        gt_labels = np.zeros(len(new_idxs), dtype=int)
        for inst in new_idxs:
            gt_labels[inst] = pts_semantic_mask[pts_instance_mask == inst][0]

        input_dict['gt_labels_3d'] = gt_labels

        return input_dict

@TRANSFORMS.register_module()
class PointSample_(PointSample):

    def _points_random_sampling(self, points, num_samples):
        """Points random sampling. Sample points to a certain number.
        
        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:
                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """

        point_range = range(len(points))
        choices = np.random.choice(point_range, 
                                   min(num_samples, len(points)))
        
        return points[choices], choices

    def transform(self, input_dict):
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask',
            'pts_semantic_mask', sp_pts_mask' keys are updated in the 
            result dict.
        """
        points = input_dict['points']

        # if point number smaller than num_point, skip
        if len(points) < self.num_points:
            return input_dict

        points, choices = self._points_random_sampling(
            points, self.num_points)
        input_dict['points'] = points
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        vote_label = input_dict.get('vote_label', None)
        instance_mask = input_dict.get('instance_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            
            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]

            input_dict['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if vote_label is not None:
            vote_label = vote_label[choices]
            input_dict['vote_label'] = vote_label

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict['instance_mask'] = instance_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
            input_dict['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            input_dict['eval_ann_info']['instance_mask'] = instance_mask
            
        return input_dict
    
@TRANSFORMS.register_module()
class SkipEmptyScene(BaseTransform):
    """Skip empty scene during training.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    """

    def transform(self, input_dict):
        """Private function for skipping empty scene during training.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            'points', 'gt_labels_3d' are updated in the result dict.
        """

        if len(input_dict['gt_labels_3d']) != 0:
            self.inst = input_dict['pts_instance_mask']
            self.sem = input_dict['pts_semantic_mask']
            self.gt_labels = input_dict['gt_labels_3d']
            self.points = input_dict['points']
        else:
            input_dict['pts_instance_mask'] = self.inst
            input_dict['pts_semantic_mask'] = self.sem 
            input_dict['gt_labels_3d'] = self.gt_labels
            input_dict['points'] = self.points

        return input_dict

@TRANSFORMS.register_module()
class SkipEmptyScene_(BaseTransform):
    """Skip empty scene during training.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    """

    def transform(self, input_dict):
        """Private function for skipping empty scene during training.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            'points', 'gt_labels_3d' are updated in the result dict.
        """

        if len(input_dict["points"]) == 0:
            return None
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if len(np.unique(pts_instance_mask)) < 2:
            return None

        return input_dict

@TRANSFORMS.register_module()
class CylinderCrop(BaseTransform):
    def __init__(self, radius=8):
        self.radius = radius

    def transform(self, input_dict):

        assert "points" in input_dict.keys()
        
        # Get the tensor of points
        points_tensor = input_dict["points"].tensor.numpy()
        
        # Select a random center point
        center = points_tensor[np.random.randint(points_tensor.shape[0])]
        
        # Calculate indices of points within the radius
        choices = np.where(
            (np.sum(np.square(points_tensor[:, :2] - center[:2]), 1) < self.radius**2)
        )[0]
        
        # Update points tensor
        if "points" in input_dict.keys():
            input_dict["points"] = input_dict["points"][choices]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        # Initialize the instance mask
        instance_mask = pts_semantic_mask != 0  # Background points have -1 in pts_instance_mask
        pts_instance_mask = pts_instance_mask.copy()
        pts_instance_mask[~instance_mask] = -1
        
        if pts_instance_mask is not None:
            original_pts_instance_mask = pts_instance_mask
            pts_instance_mask = pts_instance_mask[choices]

            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]
            input_dict['pts_instance_mask'] = pts_instance_mask

            # Initialize vote_label without zero initialization
            vote_label = np.empty((len(choices), 3))
            vote_label[:] = np.nan  # Set initial values to NaN for easier debugging and identification

            # Calculate the ratio_inspoint
            ratio_inspoint = {}
            for idx in idxs:
                if idx != -1:  # Skip the background points
                    original_count = np.sum(original_pts_instance_mask == idx)
                    new_count = np.sum(pts_instance_mask == mapping[idx])
                    ratio_inspoint[mapping[idx]] = new_count / original_count if original_count > 0 else 0

                    # Calculate the vote_label for the instance
                    ind = np.where(original_pts_instance_mask == idx)[0]
                    if len(ind) > 0:
                        pos = points_tensor[ind, :3]
                        max_pos = pos.max(0)
                        min_pos = pos.min(0)
                        center = 0.5 * (min_pos + max_pos)
                        
                        # Find the points in the cylinder that belong to this instance
                        cylinder_ind = np.where(pts_instance_mask == mapping[idx])[0]
                        vote_label[cylinder_ind, :] = center - points_tensor[choices[cylinder_ind], :3]
            
            
            input_dict['ratio_inspoint'] = ratio_inspoint
            input_dict['vote_label'] = torch.tensor(vote_label, dtype=torch.float32)
            input_dict['instance_mask'] = torch.tensor(instance_mask, dtype=torch.bool)

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict['instance_mask'] = instance_mask 

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
            input_dict['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            input_dict['eval_ann_info']['instance_mask'] = instance_mask

        return input_dict


@TRANSFORMS.register_module()
class CylinderCrop_RemoveOutpoints(BaseTransform):
    def __init__(self, radius=8):
        self.radius = radius

    def transform(self, input_dict):
        assert "points" in input_dict.keys()

        # Get the tensor of points
        points_tensor = input_dict["points"].tensor.numpy()

        # Select a random center point
        center = points_tensor[np.random.randint(points_tensor.shape[0])]

        # Calculate indices of points within the radius
        choices = np.where(
            (np.sum(np.square(points_tensor[:, :2] - center[:2]), 1) < self.radius**2)
        )[0]

        # Update the points tensor
        if "points" in input_dict.keys():
            input_dict["points"] = input_dict["points"][choices]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        # Initialize the instance mask based on semantic labels
        instance_mask = pts_semantic_mask != 0  # Background points have -1 in pts_instance_mask
        pts_instance_mask = pts_instance_mask.copy()
        pts_instance_mask[~instance_mask] = -1  # Set background points to -1 in instance mask
        

        if pts_instance_mask is not None:
            original_pts_instance_mask = pts_instance_mask
            pts_instance_mask = pts_instance_mask[choices]  # Only work with selected points after cropping

            # Initialize vote_label and ratio_inspoint
            vote_label = np.empty((len(choices), 3))
            vote_label[:] = np.nan  # Set initial values to NaN for easier debugging
            ratio_inspoint = {}

            # Find all unique instance indices after cropping
            valid_choices = []
            unique_instances = np.unique(pts_instance_mask)

            for instance in unique_instances:
                if instance != -1:  # Only process non-background instances
                    original_count = np.sum(original_pts_instance_mask == instance)
                    new_count = np.sum(pts_instance_mask == instance)

                    if original_count > 0:
                        ratio = new_count / original_count
                    else:
                        ratio = 0

                    if ratio == 1:  # Only keep fully contained instances
                        ratio_inspoint[instance] = ratio
                        valid_choices.extend(np.where(pts_instance_mask == instance)[0])

                        # Calculate the vote_label for the instance
                        ind = np.where(original_pts_instance_mask == instance)[0]
                        if len(ind) > 0:
                            pos = points_tensor[ind, :3]
                            max_pos = pos.max(0)
                            min_pos = pos.min(0)
                            center = 0.5 * (min_pos + max_pos)

                            # Find the points in the cylinder that belong to this instance
                            cylinder_ind = np.where(pts_instance_mask == instance)[0]
                            vote_label[cylinder_ind, :] = center - points_tensor[choices[cylinder_ind], :3]

            # Add background points to valid choices
            background_choices = np.where(pts_instance_mask == -1)[0]
            valid_choices = np.array(valid_choices + background_choices.tolist())

            # Filter instance_mask and create new continuous instance labels
            filtered_instance_mask = pts_instance_mask[valid_choices]
            valid_instance_ids = np.unique(filtered_instance_mask)

            # Create a mapping from original instance IDs to new continuous instance labels, keeping -1 as background
            instance_mapping = {-1: -1}
            non_background_instances = [inst for inst in valid_instance_ids if inst != -1]
            instance_mapping.update({old_id: new_id for new_id, old_id in enumerate(non_background_instances)})

            # Apply the new mapping to filtered_instance_mask
            new_instance_mask = np.vectorize(instance_mapping.get)(filtered_instance_mask)

            # Update ratio_inspoint with new continuous instance labels
            new_ratio_inspoint = {instance_mapping[inst]: ratio_inspoint[inst] for inst in valid_instance_ids if inst in ratio_inspoint}

            # Update input_dict with valid choices
            input_dict['points'] = input_dict['points'][valid_choices]
            input_dict['pts_instance_mask'] = new_instance_mask  
            input_dict['vote_label'] = torch.tensor(vote_label[valid_choices], dtype=torch.float32)
            input_dict['instance_mask'] = instance_mask[choices][valid_choices]
            input_dict['ratio_inspoint'] = new_ratio_inspoint 

        # Update semantic mask if present
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[choices][valid_choices]

        # Update superpoint mask if present
        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices][valid_choices]
            sp_pts_mask = np.unique(sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        # Update evaluation annotation info if present
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = input_dict['pts_instance_mask']
            input_dict['eval_ann_info']['pts_semantic_mask'] = input_dict['pts_semantic_mask']
            input_dict['eval_ann_info']['instance_mask'] = input_dict['instance_mask']

        return input_dict



@TRANSFORMS.register_module()
class GridSample(BaseTransform):
    def __init__(self, grid_size=0.2, mode="train", hash_type="fnv"):
        self.grid_size = grid_size
        self.mode = mode
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec

    def transform(self, input_dict):
        assert "points" in input_dict.keys()
        points = input_dict["points"]
        
        scaled_points = points.tensor / self.grid_size
        grid_points = torch.floor(scaled_points).int()
        min_points = torch.min(grid_points, dim=0).values
        grid_points -= min_points
        scaled_points -= min_points
        min_points = min_points * self.grid_size

        key = self.hash(grid_points)
        idx_sort = torch.argsort(key)
        key_sort = key[idx_sort]
        unique_results = torch.unique(key_sort, return_inverse=True, return_counts=True)
        _, inverse, count = unique_results

        if self.mode == "train":  # train mode
            idx_select = (
                torch.cumsum(torch.cat((torch.tensor([0]), count[:-1])), dim=0)
                + torch.randint(0, count.max(), count.size()) % count
            )
            choices = idx_sort[idx_select]
        else:
            raise NotImplementedError("Only train mode is implemented in this example")

        # Subsampled data
        input_dict["points"] = points[choices]

        #print(input_dict["points"].shape)
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        vote_label = input_dict.get('vote_label', None)
        instance_mask = input_dict.get('instance_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            
            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]

            input_dict['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if vote_label is not None:
            vote_label = vote_label[choices]
            input_dict['vote_label'] = vote_label

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict['instance_mask'] = instance_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
            input_dict['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            input_dict['eval_ann_info']['instance_mask'] = instance_mask

        return input_dict

    def fnv_hash_vec(self, vec):
        # Use smaller values to avoid overflow issues
        FNV_prime = torch.tensor(16777619, dtype=torch.int64)
        offset_basis = torch.tensor(2166136261, dtype=torch.int64)
        hash = torch.full((vec.shape[0],), offset_basis, dtype=torch.int64)
        for i in range(vec.shape[1]):
            hash = hash ^ vec[:, i].to(torch.int64)
            hash = hash * FNV_prime
        return hash

    def ravel_hash_vec(self, vec):
        # Implement the ravel hash function for vectors
        vec_max = torch.max(vec, dim=0).values + 1
        hash = torch.ravel_multi_index(vec.t(), vec_max)
        return hash
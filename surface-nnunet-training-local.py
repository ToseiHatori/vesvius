#!/usr/bin/env python3
"""
Vesuvius nnUNet Training & Inference - Local Version

This script runs nnUNet training and inference locally or in Docker.
Based on surface-nnunet-training-inference-with-2xt4.ipynb.

Implements the Host Baseline solution:
- Custom Trainer: nnUNetTrainerMedialSurfaceRecall (Skeleton Recall Loss)
- Post-processing: threshold 0.75 + Frangi surfaceness filter
- Achieves 0.543 LB raw, 0.562 with post-processing

Usage:
    # Host Baseline (recommended)
    python surface-nnunet-training-local.py --host-baseline --epochs 1000

    # Debug mode (fast verification with small data)
    python surface-nnunet-training-local.py --debug --train

    # Training only
    python surface-nnunet-training-local.py --train --epochs 50

    # Inference only (using existing model)
    python surface-nnunet-training-local.py --inference --model-path /path/to/checkpoint.pth

    # Full pipeline with post-processing
    python surface-nnunet-training-local.py --train --inference --epochs 100 --postprocess

    # With custom configuration
    python surface-nnunet-training-local.py --train --config 3d_fullres --fold all --gpus 2
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple, List, Literal, Union

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

Epochs = Literal[1, 5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 4000, 8000]

# =============================================================================
# PATH CONFIGURATION - LOCAL
# =============================================================================

PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
DATASET_DIR = DATA_DIR / "vesuvius-challenge-surface-detection"

# nnUNet directories
NNUNET_OUTPUT = PROJECT_DIR / "nnunet_output"
NNUNET_WORK = PROJECT_DIR / "nnunet_work"

NNUNET_RAW = NNUNET_WORK / "nnUNet_data" / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_OUTPUT / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_OUTPUT / "nnUNet_results"

# Pre-prepared preprocessed data (if available)
PREPARED_PREPROCESSED_PATH = NNUNET_PREPROCESSED

# Dataset configuration
DATASET_ID = 100
DATASET_NAME = f"Dataset{DATASET_ID:03d}_VesuviusSurface"

# Host Baseline defaults
HOST_BASELINE_TRAINER = "nnUNetTrainerMedialSurfaceRecall"
HOST_BASELINE_EPOCHS = 1200  # Host used ~1200 epochs
HOST_BASELINE_THRESHOLD = 0.75
HOST_BASELINE_LR = 0.01

# Debug mode defaults
DEBUG_EPOCHS = 1
DEBUG_MAX_CASES = 1
DEBUG_NUM_VAL_BATCHES = 1  # Reduce validation batches
DEBUG_PATCH_SIZE = [64, 64, 64]  # Smaller patch size for OOM prevention
DEBUG_BATCH_SIZE = 1

# Host Baseline 3d_fullres plan configuration (from host_solution.txt)
# This is the exact configuration used by the competition host
HOST_BASELINE_PLAN_CONFIG = {
    "data_identifier": "nnUNetPlans_3d_fullres",
    "preprocessor_name": "DefaultPreprocessor",
    "batch_size": 1,  # Reduced from 2 for single GPU (host used 2 with larger VRAM)
    "patch_size": [192, 192, 192],
    "spacing": [1.0, 1.0, 1.0],
    "normalization_schemes": ["ZScoreNormalization"],
    "use_mask_for_norm": [False],
    "resampling_fn_data": "resample_data_or_seg_to_shape",
    "resampling_fn_seg": "resample_data_or_seg_to_shape",
    "resampling_fn_data_kwargs": {
        "is_seg": False,
        "order": 3,
        "order_z": 0,
        "force_separate_z": None
    },
    "resampling_fn_seg_kwargs": {
        "is_seg": True,
        "order": 1,
        "order_z": 0,
        "force_separate_z": None
    },
    "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
    "resampling_fn_probabilities_kwargs": {
        "is_seg": False,
        "order": 1,
        "order_z": 0,
        "force_separate_z": None
    },
    "architecture": {
        "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        "arch_kwargs": {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True}
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
    },
    "batch_dice": False
}


def get_num_workers() -> int:
    """Get number of CPU workers."""
    return os.cpu_count() or 4


def get_gpu_count() -> int:
    """Get number of available CUDA GPUs."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True
            )
            if result.returncode == 0:
                return len([l for l in result.stdout.strip().split('\n') if l.startswith('GPU')])
        except FileNotFoundError:
            pass
        return 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_environment():
    """Set up nnUNet environment variables and directories."""
    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    # Only set nnUNet_compile if not already set (e.g., by debug mode)
    if "nnUNet_compile" not in os.environ:
        os.environ["nnUNet_compile"] = "true"
    os.environ["nnUNet_USE_BLOSC2"] = "1"

    print(f"nnUNet_raw: {NNUNET_RAW}")
    print(f"nnUNet_preprocessed: {NNUNET_PREPROCESSED}")
    print(f"nnUNet_results: {NNUNET_RESULTS}")
    print(f"nnUNet_compile: {os.environ.get('nnUNet_compile')}")
    print(f"NUM_WORKERS: {get_num_workers()}")
    print(f"NUM_GPUS: {get_gpu_count()}")


def get_trainer_name(epochs: Optional[int], trainer: str = "nnUNetTrainer") -> str:
    """Get trainer class name based on epochs.

    For custom trainers (non-standard), returns the trainer name as-is.
    For standard nnUNetTrainer, appends epoch suffix if needed.
    """
    # Custom trainers don't use epoch suffix
    if trainer != "nnUNetTrainer":
        return trainer
    if epochs is None or epochs == 1000:
        return "nnUNetTrainer"
    elif epochs == 1:
        return "nnUNetTrainer_1epoch"
    else:
        return f"nnUNetTrainer_{epochs}epochs"


def get_training_output_dir(
    epochs: Optional[int] = None,
    plans: str = "nnUNetResEncUNetMPlans",
    config: str = "3d_fullres",
    fold: Union[int, str] = "all",
    trainer: str = "nnUNetTrainer"
) -> Path:
    """Get the training output directory path."""
    trainer_name = get_trainer_name(epochs, trainer)
    return NNUNET_RESULTS / DATASET_NAME / f"{trainer_name}__{plans}__{config}" / f"fold_{fold}"


def run_command(
    cmd: str,
    name: str = "Command",
    timeout: Optional[int] = None
) -> bool:
    """Execute shell command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {cmd}")
    print('='*60)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            timeout=timeout,
            env={**os.environ}
        )
        success = result.returncode == 0
        if success:
            print(f"\n{name} completed successfully!")
        else:
            print(f"\n{name} FAILED with exit code {result.returncode}")
        return success
    except subprocess.TimeoutExpired:
        print(f"\n{name} TIMEOUT after {timeout}s!")
        return False
    except Exception as e:
        print(f"\n{name} ERROR: {e}")
        return False


# =============================================================================
# CUSTOM TRAINER FOR HOST BASELINE
# =============================================================================

def install_custom_trainer() -> Path:
    """
    Install custom trainer files into nnunetv2 package directory.

    This function writes the necessary Python files for the Host Baseline
    custom trainer directly into the nnunetv2 installation directory.

    Components installed:
    1. skeleton_recall.py - SoftSkeletonRecallLoss
    2. skeleton_losses.py - DC_SkelREC_and_CE_loss
    3. medial_surface.py - MedialSurfaceTransform
    4. nnUNetTrainerMedialSurfaceRecall.py - Custom trainer

    Returns:
        Path to nnunetv2 package directory
    """
    import nnunetv2
    nnunet_path = Path(nnunetv2.__file__).parent

    print(f"Installing custom trainer to: {nnunet_path}")

    # ==========================================================================
    # 1. Create loss directory and write skeleton_recall.py
    # ==========================================================================
    loss_dir = nnunet_path / "training" / "loss"
    loss_dir.mkdir(parents=True, exist_ok=True)

    skeleton_recall_code = '''"""
Soft Skeleton Recall Loss for topology-preserving segmentation.

Based on MIC-DKFZ/Skeleton-Recall implementation.
The skeleton is pre-computed and passed directly to the loss function.

Key difference from clDice: skeleton is NOT computed inside the loss function.
Instead, it's computed upstream (in data augmentation or training step) and
passed as the target (y) argument.
"""

import torch
import torch.nn as nn
from typing import Callable

try:
    from nnunetv2.utilities.ddp_allgather import AllGatherGrad
except ImportError:
    AllGatherGrad = None


class SoftSkeletonRecallLoss(nn.Module):
    """
    Soft Skeleton Recall Loss (MIC-DKFZ implementation).

    Computes recall of ground truth skeleton in predicted segmentation:
    recall = sum(pred * skel_gt) / sum(skel_gt)

    IMPORTANT: The skeleton (y) must be pre-computed and passed directly.
    This loss does NOT compute skeleton internally.

    Args:
        apply_nonlin: Nonlinearity to apply to predictions (e.g., softmax)
        batch_dice: Whether to compute batch-level recall (sum across batch)
        do_bg: Whether to include background class (must be False)
        smooth: Smoothing factor
        ddp: Whether using distributed training
    """

    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1.0,
        ddp: bool = True
    ):
        super().__init__()
        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute skeleton recall loss.

        Args:
            x: Network predictions (B, C, D, H, W)
            y: Pre-computed skeleton target (B, 1, D, H, W) with class labels
               OR one-hot skeleton (B, C, D, H, W)
            loss_mask: Optional mask for valid regions

        Returns:
            Negative skeleton recall (for minimization)
        """
        shp_x, shp_y = x.shape, y.shape

        # Apply nonlinearity (softmax) to predictions
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Remove background channel from predictions
        x = x[:, 1:]

        # Spatial axes for summation
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            # Handle shape mismatch
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            # Convert skeleton to one-hot if needed
            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # Already matches prediction shape (minus background removed above)
                # This shouldn't happen normally, but handle it
                y_onehot = y
            elif y.shape[1] == 1:
                # Skeleton has class labels (B, 1, D, H, W) - convert to one-hot
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)
                # Remove background channel
                y_onehot = y_onehot[:, 1:]
            else:
                # Already one-hot with all channels
                y_onehot = y[:, 1:]

        # Compute skeleton recall
        if loss_mask is None:
            sum_gt = y_onehot.sum(axes)
            inter_rec = (x * y_onehot).sum(axes)
        else:
            sum_gt = (y_onehot * loss_mask).sum(axes)
            inter_rec = (x * y_onehot * loss_mask).sum(axes)

        # Handle distributed training
        if self.ddp and self.batch_dice and AllGatherGrad is not None:
            inter_rec = AllGatherGrad.apply(inter_rec).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        # Batch-level recall
        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        # Compute recall: (intersection + smooth) / (skeleton_sum + smooth)
        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt + self.smooth, 1e-8))
        rec = rec.mean()

        # Return negative recall for minimization
        return -rec
'''

    skeleton_recall_path = loss_dir / "skeleton_recall.py"
    skeleton_recall_path.write_text(skeleton_recall_code)
    print(f"  Created: {skeleton_recall_path}")

    # ==========================================================================
    # 2. Write skeleton_losses.py (DC_SkelREC_and_CE_loss)
    # ==========================================================================
    compound_losses_code = '''"""
Compound loss combining Dice, Cross-Entropy, and Skeleton Recall.

Based on MIC-DKFZ/Skeleton-Recall implementation.
This loss is designed for topology-preserving segmentation.
"""

import torch
import torch.nn as nn

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.skeleton_recall import SoftSkeletonRecallLoss


def softmax_helper_dim1(x):
    """Apply softmax along channel dimension."""
    return torch.softmax(x, dim=1)


class DC_SkelREC_and_CE_loss(nn.Module):
    """
    Combined Dice + Skeleton Recall + Cross-Entropy Loss.

    Based on MIC-DKFZ implementation:
    Total loss = weight_dice * DiceLoss + weight_ce * CELoss + weight_srec * SkelRecLoss

    IMPORTANT: The skeleton (skel) must be pre-computed and passed to forward().
    Skeleton is passed to SoftSkeletonRecallLoss as the target (y).

    Args:
        soft_dice_kwargs: Arguments for SoftDiceLoss
        soft_skelrec_kwargs: Arguments for SoftSkeletonRecallLoss
        ce_kwargs: Arguments for RobustCrossEntropyLoss
        weight_ce: Weight for CE loss (default 1.0)
        weight_dice: Weight for Dice loss (default 1.0)
        weight_srec: Weight for Skeleton Recall loss (default 1.0)
        ignore_label: Label to ignore in loss computation
        dice_class: Dice loss class to use
    """

    def __init__(
        self,
        soft_dice_kwargs: dict = None,
        soft_skelrec_kwargs: dict = None,
        ce_kwargs: dict = None,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_srec: float = 1.0,
        ignore_label: int = None,
        dice_class: type = MemoryEfficientSoftDiceLoss
    ):
        super().__init__()

        # Handle ignore label
        if ignore_label is not None:
            if ce_kwargs is None:
                ce_kwargs = {}
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        # Dice loss
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {'batch_dice': False, 'do_bg': False, 'smooth': 1e-5, 'ddp': False}
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        # Cross-entropy loss
        if ce_kwargs is None:
            ce_kwargs = {}
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        # Skeleton recall loss
        if soft_skelrec_kwargs is None:
            soft_skelrec_kwargs = {'batch_dice': False, 'do_bg': False, 'smooth': 1e-5, 'ddp': False}
        self.srec = SoftSkeletonRecallLoss(apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs)

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        skel: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            net_output: Network predictions (B, C, D, H, W)
            target: Ground truth labels (B, 1, D, H, W)
            skel: Pre-computed skeleton (B, 1, D, H, W) with class labels

        Returns:
            Combined loss (scalar)
        """
        # Handle ignore label
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label not implemented for one-hot targets'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            target_skel = torch.where(mask, skel, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_skel = skel
            mask = None

        # Dice loss
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \\
            if self.weight_dice != 0 else 0

        # Skeleton recall loss - pass skeleton as target
        srec_loss = self.srec(net_output, target_skel, loss_mask=mask) \\
            if self.weight_srec != 0 else 0

        # CE loss
        ce_loss = self.ce(net_output, target[:, 0]) \\
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss
        return result
'''

    compound_losses_path = loss_dir / "skeleton_losses.py"
    compound_losses_path.write_text(compound_losses_code)
    print(f"  Created: {compound_losses_path}")

    # ==========================================================================
    # 3. Create transforms directory and write medial_surface.py
    # ==========================================================================
    transforms_dir = nnunet_path / "training" / "data_augmentation" / "custom_transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py
    (transforms_dir / "__init__.py").write_text("")

    medial_surface_code = '''"""
Medial Surface Transform for computing 2D slice-based skeletonization.

Based on villa implementation (ScrollPrize/villa):
https://github.com/ScrollPrize/villa/blob/main/segmentation/models/arch/nnunet/nnunetv2/training/data_augmentation/custom_transforms/skeletonization.py

NOTE: Despite host_solution.txt mentioning "3 different axes", the actual
villa implementation uses Z-axis ONLY for skeletonization.

The skeleton is computed by:
1. For each Z-slice, compute 2D skeleton
2. Optionally dilate (dilation twice per MIC-DKFZ)
3. Multiply by original labels to retain class information
"""

import numpy as np
from typing import Tuple
import torch

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from scipy.ndimage import binary_dilation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_medial_surface(
    segmentation: np.ndarray,
    do_tube: bool = True,
    ignore_label: int = None
) -> np.ndarray:
    """
    Compute medial surface skeleton using Z-axis only 2D skeletonization.

    This matches the villa implementation exactly.

    Args:
        segmentation: Segmentation mask (C, D, H, W) or (D, H, W) with class labels
        do_tube: Whether to dilate skeleton (dilation x 2 per villa)
        ignore_label: Label to treat as background

    Returns:
        Skeleton with same shape as input, containing class labels
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required: pip install scikit-image")
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required: pip install scipy")

    # Handle channel dimension
    squeeze = False
    if segmentation.ndim == 4:
        seg = segmentation[0]
        squeeze = True
    else:
        seg = segmentation

    # Handle ignore label
    if ignore_label is not None:
        seg = np.where(seg == ignore_label, 0, seg)

    binary_seg = (seg > 0)

    if not binary_seg.any():
        result = np.zeros_like(seg, dtype=np.int16)
        if squeeze:
            result = result[np.newaxis, ...]
        return result

    D, H, W = seg.shape
    skeleton = np.zeros_like(binary_seg, dtype=bool)

    # Z-axis only (per villa implementation)
    for z in range(D):
        slice_2d = binary_seg[z]
        if slice_2d.any():
            skeleton[z] |= skeletonize(slice_2d)

    # Convert to int16 and apply dilation if needed
    skeleton = skeleton.astype(np.int16)

    if do_tube and skeleton.any():
        skeleton = binary_dilation(skeleton).astype(np.int16)
        skeleton = binary_dilation(skeleton).astype(np.int16)

    # Multiply by original labels (per villa: skel *= seg_all[0].astype(np.int16))
    skeleton = skeleton * seg.astype(np.int16)

    if squeeze:
        skeleton = skeleton[np.newaxis, ...]

    return skeleton


class MedialSurfaceTransform:
    """
    Batchgenerators-compatible transform for computing medial surface skeletons.

    Matches villa implementation for use in nnUNet data augmentation pipeline.
    Adds a 'skel' key to the data dict containing the skeleton.

    Args:
        do_tube: Whether to dilate skeleton (training=False, validation=True)
        ignore_label: Label to treat as background
    """

    def __init__(
        self,
        do_tube: bool = True,
        ignore_label: int = None
    ):
        self.do_tube = do_tube
        self.ignore_label = ignore_label

    def apply(self, data_dict, **params) -> dict:
        """
        Apply transform to data dict.

        Expected keys:
        - 'segmentation': Segmentation tensor (C, D, H, W)

        Added keys:
        - 'skel': Computed skeleton tensor
        """
        seg = data_dict.get('segmentation')
        if seg is None:
            return data_dict

        # Convert to numpy if tensor
        if isinstance(seg, torch.Tensor):
            seg_np = seg.numpy()
        else:
            seg_np = seg

        # Compute skeleton
        skeleton = compute_medial_surface(
            seg_np,
            do_tube=self.do_tube,
            ignore_label=self.ignore_label
        )

        # Convert back to tensor
        data_dict['skel'] = torch.from_numpy(skeleton)
        return data_dict

    def __call__(self, **data_dict) -> dict:
        """Alias for apply method."""
        return self.apply(data_dict)
'''

    medial_surface_path = transforms_dir / "medial_surface.py"
    medial_surface_path.write_text(medial_surface_code)
    print(f"  Created: {medial_surface_path}")

    # ==========================================================================
    # 4. Create trainer variant directory and write nnUNetTrainerMedialSurfaceRecall.py
    # ==========================================================================
    trainer_variants_dir = nnunet_path / "training" / "nnUNetTrainer" / "variants" / "loss"
    trainer_variants_dir.mkdir(parents=True, exist_ok=True)

    # Ensure __init__.py exists
    init_path = trainer_variants_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")

    trainer_code = '''"""
nnUNet Trainer with Medial Surface Recall Loss for Host Baseline.

This trainer extends nnUNetTrainer to use the combined Dice + CE + Skeleton Recall loss.
Uses standard nnUNet data augmentation for API compatibility.

Based on Host Baseline description:
- Uses MedialSurfaceRecall loss (modified SkeletonRecall)
- Helps fighting the creation of holes in predictions
- Makes output more sheet-like
- Initial learning rate: 0.01

Key modifications:
1. Uses DC_SkelREC_and_CE_loss (Dice + CE + Skeleton Recall)
2. Computes medial surface skeleton during training (Z-axis only, per villa implementation)
3. Full deep supervision support with skeleton at each scale
4. Learning rate set to 0.01 as per Host Baseline
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, List, Dict

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.skeleton_losses import DC_SkelREC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class DeepSupervisionWrapperWithSkeleton(torch.nn.Module):
    """
    Deep supervision wrapper that supports skeleton recall loss.

    This wrapper handles multiple output scales and computes skeleton loss
    at each scale by downsampling the full-resolution skeleton.

    Uses average pooling with binarization for skeleton downsampling
    to preserve thin structures better than max pooling.
    """

    def __init__(self, loss_fn, weights: List[float]):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, outputs: List[torch.Tensor], targets: List[torch.Tensor],
                gt_skeleton: torch.Tensor = None) -> torch.Tensor:
        """
        Compute weighted loss across all scales.

        Args:
            outputs: List of network outputs at different scales (index 0 = full res)
            targets: List of targets at different scales (index 0 = full res)
            gt_skeleton: Full-resolution skeleton (will be downsampled for each scale)

        Returns:
            Weighted sum of losses
        """
        total_loss = 0.0

        # Store original skeleton for downsampling at each scale
        original_skeleton = gt_skeleton

        for i, (output, target, weight) in enumerate(zip(outputs, targets, self.weights)):
            # Skip zero-weighted outputs
            if weight == 0:
                continue

            # Ensure output and target have matching spatial dimensions
            output_shape = output.shape[2:]
            target_shape = target.shape[2:]

            # If shapes don't match, resize target to match output
            if output_shape != target_shape:
                target = F.interpolate(
                    target.float(),
                    size=output_shape,
                    mode='nearest'
                ).long() if target.dtype in (torch.long, torch.int) else F.interpolate(
                    target.float(),
                    size=output_shape,
                    mode='trilinear',
                    align_corners=False
                )

            if original_skeleton is not None:
                # Downsample skeleton to match current output scale
                skel_shape = original_skeleton.shape[2:]

                if output_shape == skel_shape:
                    skel = original_skeleton
                else:
                    # Use adaptive average pooling then threshold
                    skel = F.adaptive_avg_pool3d(original_skeleton, output_shape)
                    # Re-binarize: any overlap > 0.1 is considered skeleton
                    skel = (skel > 0.1).float()

                loss = self.loss_fn(output, target, skel)
            else:
                loss = self.loss_fn(output, target)

            total_loss += weight * loss

        return total_loss


def compute_skeleton_batch(segmentation: np.ndarray, do_tube: bool = False) -> np.ndarray:
    """
    Compute skeleton for a batch of segmentations.

    Uses Z-axis only 2D skeletonization per villa implementation.
    Skeleton pixels retain the original label values (not just binary).

    Args:
        segmentation: (B, C, D, H, W) or (B, 1, D, H, W) array with class labels
        do_tube: Whether to dilate skeleton (training=False, validation=True per villa)

    Returns:
        Skeleton array with same shape as input, containing class labels
    """
    from skimage.morphology import skeletonize
    from scipy.ndimage import binary_dilation

    # Handle different input shapes
    if segmentation.ndim == 5:
        batch_size = segmentation.shape[0]
        # Get the label channel (assumes channel 0 contains class labels)
        seg_4d = segmentation[:, 0]
    elif segmentation.ndim == 4:
        batch_size = segmentation.shape[0]
        seg_4d = segmentation
    else:
        raise ValueError(f"Expected 4D or 5D input, got {segmentation.ndim}D")

    skeletons = []
    for b in range(batch_size):
        seg_3d = seg_4d[b]
        binary_seg = (seg_3d > 0).astype(bool)

        if not binary_seg.any():
            skeletons.append(np.zeros_like(seg_3d, dtype=np.int16))
            continue

        D, H, W = binary_seg.shape
        skeleton = np.zeros_like(binary_seg, dtype=bool)

        # Z-axis only (per villa implementation)
        for z in range(D):
            slice_2d = binary_seg[z]
            if slice_2d.any():
                skeleton[z] |= skeletonize(slice_2d)

        # Convert to int16 for label storage
        skeleton = skeleton.astype(np.int16)

        # Optional dilation (training=False, validation=True per villa)
        if do_tube and skeleton.any():
            skeleton = binary_dilation(skeleton).astype(np.int16)
            skeleton = binary_dilation(skeleton).astype(np.int16)

        # Multiply by original labels to retain class information (per villa)
        # This ensures skeleton pixels have the correct class label, not just 1
        skeleton = skeleton * seg_3d.astype(np.int16)

        skeletons.append(skeleton)

    result = np.stack(skeletons, axis=0)
    # Add channel dimension if needed
    if segmentation.ndim == 5:
        result = result[:, np.newaxis, ...]

    return result


class nnUNetTrainerMedialSurfaceRecall(nnUNetTrainer):
    """
    Custom nnUNet trainer for Host Baseline with Skeleton Recall Loss.

    Modifications from base nnUNetTrainer:
    1. Uses DC_SkelREC_and_CE_loss (Dice + CE + Skeleton Recall)
    2. Computes medial surface skeleton during training (Z-axis only per villa)
    3. Full deep supervision support with skeleton at each scale
    4. Learning rate set to 0.01 as per Host Baseline

    Uses standard nnUNet data augmentation for API compatibility.

    Loss weights:
    - weight_dice: 1.0
    - weight_ce: 1.0
    - weight_srec: 1.0
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Loss weights for skeleton recall
        self.weight_dice = 1.0
        self.weight_ce = 1.0
        self.weight_srec = 1.0

        # Host Baseline uses 0.01 initial learning rate
        self.initial_lr = 0.01

        # Read epochs from environment variable (set by surface-nnunet-training-local.py)
        import os
        epochs_env = os.environ.get("NNUNET_EPOCHS")
        if epochs_env is not None:
            self.num_epochs = int(epochs_env)
            print(f"Using {self.num_epochs} epochs from NNUNET_EPOCHS environment variable")

        # Seed fixing for reproducibility
        seed_env = os.environ.get("NNUNET_SEED")
        if seed_env is not None:
            seed = int(seed_env)
            self._set_seed(seed)
            print(f"Using seed {seed} from NNUNET_SEED environment variable")

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For full determinism (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed} for reproducibility")

    def _build_loss(self):
        """
        Build the combined Dice + CE + Skeleton Recall loss function with Deep Supervision.

        Returns:
            Loss function wrapped with DeepSupervisionWrapperWithSkeleton if deep supervision is enabled
        """
        # Get ignore label if set
        ignore_label = self.label_manager.ignore_label if hasattr(self.label_manager, 'ignore_label') else None

        # Configure loss components (do_bg=False per villa/MIC-DKFZ implementation)
        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }

        # Skeleton recall uses same kwargs as Dice (per MIC-DKFZ compound_losses.py)
        soft_skelrec_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }

        ce_kwargs = {}
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        loss = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            soft_skelrec_kwargs=soft_skelrec_kwargs,
            ce_kwargs=ce_kwargs,
            weight_dice=self.weight_dice,
            weight_ce=self.weight_ce,
            weight_srec=self.weight_srec,
            ignore_label=ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        # Wrap with Deep Supervision if enabled (per villa implementation)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # Exponential decay weights: [1, 0.5, 0.25, 0.125, ...]
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            # Set last weight to 0 (per villa)
            if len(weights) > 1:
                weights[-1] = 0
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            self.ds_loss_weights = weights.tolist()
            loss = DeepSupervisionWrapperWithSkeleton(loss, self.ds_loss_weights)

        return loss

    def _get_deep_supervision_scales(self) -> List[List[int]]:
        """Get deep supervision scales from configuration."""
        pool_op_kernel_sizes = self.configuration_manager.pool_op_kernel_sizes
        return list(list(i) for i in pool_op_kernel_sizes)

    def train_step(self, batch: dict) -> dict:
        """
        Perform a single training step with skeleton recall loss and deep supervision.

        Computes skeleton from target segmentation on CPU, then transfers to GPU.
        Uses do_tube=False for training per villa implementation.

        With deep supervision enabled, loss is computed at all scales with exponential
        decay weights (per villa implementation).
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.autocast('cpu', enabled=False):
            output = self.network(data)

            # Get full resolution target for skeleton computation
            if isinstance(target, list):
                target_fullres = target[0]
            else:
                target_fullres = target

            # Compute skeleton from full resolution target (do_tube=False for training per villa)
            target_np = target_fullres.detach().cpu().numpy()
            skeleton_np = compute_skeleton_batch(target_np, do_tube=False)
            gt_skeleton = torch.from_numpy(skeleton_np).float().to(self.device, non_blocking=True)

            # Compute loss with deep supervision if enabled
            if self.enable_deep_supervision and isinstance(output, (list, tuple)):
                # Pass outputs, targets, and skeleton to DeepSupervisionWrapperWithSkeleton
                l = self.loss(output, target, gt_skeleton)
            else:
                # Single output mode (no deep supervision)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                if isinstance(target, list):
                    target = target[0]
                l = self.loss(output, target, gt_skeleton)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """
        Perform validation step with skeleton recall loss and deep supervision.

        Uses do_tube=True for validation per villa implementation.
        Returns tp_hard, fp_hard, fn_hard for online Dice evaluation.

        With deep supervision enabled, loss is computed at all scales.
        Dice metrics are computed only at full resolution.
        """
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.autocast('cpu', enabled=False):
                output = self.network(data)

                # Get full resolution output and target for metrics
                if isinstance(output, (list, tuple)):
                    output_fullres = output[0]
                else:
                    output_fullres = output

                if isinstance(target, list):
                    target_fullres = target[0]
                else:
                    target_fullres = target

                # Compute skeleton from full resolution target (do_tube=True for validation per villa)
                target_np = target_fullres.detach().cpu().numpy()
                skeleton_np = compute_skeleton_batch(target_np, do_tube=True)
                gt_skeleton = torch.from_numpy(skeleton_np).float().to(self.device, non_blocking=True)

                # Compute loss with deep supervision if enabled
                if self.enable_deep_supervision and isinstance(output, (list, tuple)):
                    # Pass outputs, targets, and skeleton to DeepSupervisionWrapperWithSkeleton
                    l = self.loss(output, target, gt_skeleton)
                else:
                    # Single output mode (no deep supervision)
                    l = self.loss(output_fullres, target_fullres, gt_skeleton)

                # Compute tp/fp/fn for online Dice evaluation (full resolution only)
                # axes excludes batch (0) and channel (1) dimensions
                axes = [0] + list(range(2, output_fullres.ndim))

                # Region-based prediction using sigmoid threshold
                if self.label_manager.has_regions:
                    predicted_segmentation_onehot = (torch.sigmoid(output_fullres) > 0.5).long()
                else:
                    # Standard softmax-based segmentation
                    output_seg = output_fullres.argmax(1)[:, None]
                    predicted_segmentation_onehot = torch.zeros(output_fullres.shape, device=output_fullres.device, dtype=torch.float16)
                    predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                    del output_seg

                # Handle ignore label if present
                target_for_metrics = target_fullres
                if self.label_manager.has_ignore_label:
                    if not self.label_manager.has_regions:
                        mask = (target_for_metrics != self.label_manager.ignore_label).float()
                        target_for_metrics = target_for_metrics.clone()
                        target_for_metrics[target_for_metrics == self.label_manager.ignore_label] = 0
                    else:
                        if target_for_metrics.dtype == torch.bool:
                            mask = ~target_for_metrics[:, -1:]
                        else:
                            mask = 1 - target_for_metrics[:, -1:]
                        target_for_metrics = target_for_metrics[:, :-1]
                else:
                    mask = None

                tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_for_metrics, axes=axes, mask=mask)

                tp_hard = tp.detach().cpu().numpy()
                fp_hard = fp.detach().cpu().numpy()
                fn_hard = fn.detach().cpu().numpy()

                # Remove background class for non-region training
                if not self.label_manager.has_regions:
                    tp_hard = tp_hard[1:]
                    fp_hard = fp_hard[1:]
                    fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
'''

    trainer_path = trainer_variants_dir / "nnUNetTrainerMedialSurfaceRecall.py"
    trainer_path.write_text(trainer_code)
    print(f"  Created: {trainer_path}")

    print("\nCustom trainer installation complete!")
    print("\nUsage:")
    print("  python surface-nnunet-training-local.py --host-baseline --epochs 1000")

    return nnunet_path


def verify_custom_trainer() -> bool:
    """Verify that custom trainer can be imported."""
    try:
        from nnunetv2.training.loss.skeleton_recall import SoftSkeletonRecallLoss
        from nnunetv2.training.loss.skeleton_losses import DC_SkelREC_and_CE_loss
        from nnunetv2.training.data_augmentation.custom_transforms.medial_surface import MedialSurfaceTransform
        from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerMedialSurfaceRecall import nnUNetTrainerMedialSurfaceRecall
        print("All custom components imported successfully!")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        print("Run install_custom_trainer() first.")
        return False


def create_host_baseline_plans(debug_mode: bool = False) -> Path:
    """
    Create nnUNet plans file with Host Baseline configuration.

    This generates a plans.json file that matches the exact configuration
    used by the competition host (from host_solution.txt).

    Key differences from default nnUNetResEncUNetMPlans:
    - patch_size: [192, 192, 192] (host's setting)
    - batch_size: 2
    - n_blocks_per_stage: [1, 3, 4, 6, 6, 6]
    - features_per_stage: [32, 64, 128, 256, 320, 320]

    Args:
        debug_mode: If True, use smaller patch_size and batch_size to prevent OOM

    Returns:
        Path to the generated plans file
    """
    plans_dir = NNUNET_PREPROCESSED / DATASET_NAME
    plans_dir.mkdir(parents=True, exist_ok=True)

    # Load existing plans if available, otherwise create new
    existing_plans_path = plans_dir / "nnUNetResEncUNetMPlans.json"
    if existing_plans_path.exists():
        with open(existing_plans_path, "r") as f:
            plans = json.load(f)
        print(f"Loaded existing plans from: {existing_plans_path}")
    else:
        # Create minimal plans structure
        plans = {
            "dataset_name": DATASET_NAME,
            "plans_name": "nnUNetHostBaselinePlans",
            "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
            "original_median_shape_after_transp": [192, 192, 192],
            "image_reader_writer": "SimpleTiffIO",
            "transpose_forward": [0, 1, 2],
            "transpose_backward": [0, 1, 2],
            "configurations": {},
            "experiment_planner_used": "nnUNetPlannerResEncM",
            "label_manager": "LabelManager",
            "foreground_intensity_properties_per_channel": {
                "0": {"max": 255.0, "mean": 128.0, "median": 128.0, "min": 0.0,
                      "percentile_00_5": 0.0, "percentile_99_5": 255.0, "std": 50.0}
            }
        }
        print("Creating new plans structure")

    # Override 3d_fullres configuration with Host Baseline settings
    config = HOST_BASELINE_PLAN_CONFIG.copy()

    # Debug mode: use smaller settings to prevent OOM
    if debug_mode:
        config["patch_size"] = DEBUG_PATCH_SIZE
        config["batch_size"] = DEBUG_BATCH_SIZE
        print(f"\nDEBUG MODE: Using reduced settings")
        print(f"  - patch_size: {DEBUG_PATCH_SIZE}")
        print(f"  - batch_size: {DEBUG_BATCH_SIZE}")

    plans["configurations"]["3d_fullres"] = config
    plans["plans_name"] = "nnUNetHostBaselinePlans"

    # Save as nnUNetHostBaselinePlans.json
    plans_path = plans_dir / "nnUNetHostBaselinePlans.json"
    with open(plans_path, "w") as f:
        json.dump(plans, f, indent=4)

    print(f"\nHost Baseline plans created: {plans_path}")
    if not debug_mode:
        print("  - patch_size: [192, 192, 192]")
        print("  - batch_size: 2")
    print("  - n_stages: 6")
    print("  - features_per_stage: [32, 64, 128, 256, 320, 320]")
    print("  - n_blocks_per_stage: [1, 3, 4, 6, 6, 6]")

    return plans_path


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def run_training(
    dataset_id: int = DATASET_ID,
    config: str = "3d_fullres",
    fold: Union[int, str] = "all",
    plans: str = "nnUNetResEncUNetMPlans",
    epochs: Optional[int] = None,
    trainer: str = "nnUNetTrainer",
    pretrained_weights: Optional[Path] = None,
    continue_training: bool = False,
    num_gpus: int = 1,
    timeout: Optional[int] = None,
    seed: Optional[int] = None
) -> bool:
    """Run nnUNet training."""
    trainer_name = get_trainer_name(epochs, trainer)

    # Set epochs via environment variable for custom trainers
    epochs_to_use = epochs if epochs else 1000
    os.environ["NNUNET_EPOCHS"] = str(epochs_to_use)

    # Set seed via environment variable for reproducibility (default: 42)
    seed_to_use = seed if seed is not None else 42
    os.environ["NNUNET_SEED"] = str(seed_to_use)

    cmd = f"nnUNetv2_train {dataset_id:03d} {config} {fold} -p {plans} -tr {trainer_name}"

    if pretrained_weights:
        cmd += f" -pretrained_weights {pretrained_weights}"
    if continue_training:
        cmd += " --c"
    if num_gpus > 1:
        cmd += f" -num_gpus {num_gpus}"

    return run_command(cmd, f"Training ({epochs_to_use} epochs, {num_gpus} GPUs)", timeout=timeout)


def run_inference(
    input_dir: Path,
    output_dir: Path,
    dataset_id: int = DATASET_ID,
    config: str = "3d_fullres",
    fold: Union[int, str] = "all",
    plans: str = "nnUNetResEncUNetMPlans",
    epochs: Optional[int] = None,
    trainer: str = "nnUNetTrainer",
    save_probabilities: bool = True,
    timeout: Optional[int] = None
) -> bool:
    """Run inference with trained model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_name = get_trainer_name(epochs, trainer)

    cmd = f"nnUNetv2_predict -d {dataset_id:03d} -c {config} -f {fold}"
    cmd += f" -i {input_dir} -o {output_dir} -p {plans} -tr {trainer_name}"
    cmd += " -npp 2 -nps 2 --verbose"

    if save_probabilities:
        cmd += " --save_probabilities"

    return run_command(cmd, "Inference", timeout=timeout)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def create_spacing_json(output_path: Path, shape: tuple, spacing: tuple = (1.0, 1.0, 1.0)):
    """Create JSON sidecar with spacing info for TIFF files."""
    json_data = {"spacing": list(spacing)}
    with open(output_path, "w") as f:
        json.dump(json_data, f)


def create_dataset_json(output_dir: Path, num_training: int, file_ending: str = ".tif") -> dict:
    """Create dataset.json with ignore label support."""
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "surface": 1, "ignore": 2},
        "numTraining": num_training,
        "file_ending": file_ending,
        "overwrite_image_reader_writer": "SimpleTiffIO"
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Created {json_path}")
    return dataset_json


def prepare_single_case(args) -> bool:
    """Prepare a single TIFF file for nnUNet."""
    import tifffile

    src_path, dest_path, json_path, use_symlinks = args
    try:
        with tifffile.TiffFile(src_path) as tif:
            shape = tif.pages[0].shape if len(tif.pages) == 1 else (len(tif.pages), *tif.pages[0].shape)

        if use_symlinks:
            if not dest_path.exists():
                dest_path.symlink_to(src_path.resolve())
        else:
            shutil.copy2(src_path, dest_path)

        create_spacing_json(json_path, shape)
        return True
    except Exception as e:
        print(f"Error processing {src_path.name}: {e}")
        return False


def prepare_raw_dataset(input_dir: Path, max_cases: Optional[int] = None, use_symlinks: bool = True):
    """Convert competition data to nnUNet raw format."""
    from tqdm import tqdm

    dataset_dir = NNUNET_RAW / DATASET_NAME
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    train_images_dir = input_dir / "train_images"
    train_labels_dir = input_dir / "train_labels"

    if not train_images_dir.exists():
        print(f"ERROR: {train_images_dir} not found!")
        return None

    image_files = sorted(train_images_dir.glob("*.tif"))
    if max_cases:
        image_files = image_files[:max_cases]

    print(f"Found {len(image_files)} training cases")
    print(f"Using {'symlinks' if use_symlinks else 'copy'}")

    # Prepare arguments for parallel processing
    tasks = []
    for img_path in image_files:
        case_id = img_path.stem
        label_path = train_labels_dir / img_path.name

        if not label_path.exists():
            continue

        tasks.append((
            img_path,
            images_dir / f"{case_id}_0000.tif",
            images_dir / f"{case_id}_0000.json",
            use_symlinks
        ))
        tasks.append((
            label_path,
            labels_dir / f"{case_id}.tif",
            labels_dir / f"{case_id}.json",
            use_symlinks
        ))

    # Process in parallel
    num_workers = get_num_workers()
    print(f"Processing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(prepare_single_case, tasks),
            total=len(tasks),
            desc="Preparing dataset"
        ))

    num_converted = sum(results) // 2  # Each case has image + label
    create_dataset_json(dataset_dir, num_converted, file_ending=".tif")

    print(f"\nDataset prepared: {num_converted} cases")
    print(f"Location: {dataset_dir}")
    return dataset_dir


def prepare_test_data(input_dir: Path, output_dir: Path, use_symlinks: bool = True) -> Path:
    """Prepare test TIFF images for inference."""
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir = input_dir / "test_images"

    if not test_images_dir.exists():
        print(f"ERROR: {test_images_dir} not found!")
        return output_dir

    test_files = sorted(test_images_dir.glob("*.tif"))
    print(f"Found {len(test_files)} test cases")

    tasks = []
    for img_path in test_files:
        case_id = img_path.stem
        tasks.append((
            img_path,
            output_dir / f"{case_id}_0000.tif",
            output_dir / f"{case_id}_0000.json",
            use_symlinks
        ))

    with Pool(get_num_workers()) as pool:
        list(tqdm(
            pool.imap(prepare_single_case, tasks),
            total=len(tasks),
            desc="Preparing test data"
        ))

    return output_dir


# =============================================================================
# POST-PROCESSING FOR HOST BASELINE
# =============================================================================
# Based on host description:
# "thresholded the softmax output at 0.75, and run on it a
# modified Frangi filter that enhances surfaceness rather than vesselness"

def compute_hessian_eigenvalues_vectorized(volume, sigma):
    """
    Compute Hessian matrix eigenvalues for each voxel (vectorized implementation).

    This is a fully vectorized implementation that avoids per-voxel loops,
    providing significant speedup over the naive implementation.

    Args:
        volume: 3D volume
        sigma: Gaussian smoothing scale

    Returns:
        Tuple of (lambda1, lambda2, lambda3) sorted by absolute value
    """
    from scipy.ndimage import gaussian_filter
    import numpy as np

    # Smooth the volume with Gaussian
    smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma, mode='nearest')

    # Compute second derivatives (Hessian components) using np.gradient
    # First derivatives
    grad_z = np.gradient(smoothed, axis=0)
    grad_y = np.gradient(smoothed, axis=1)
    grad_x = np.gradient(smoothed, axis=2)

    # Second derivatives (Hessian matrix components)
    Hzz = np.gradient(grad_z, axis=0)
    Hyy = np.gradient(grad_y, axis=1)
    Hxx = np.gradient(grad_x, axis=2)
    Hzy = np.gradient(grad_z, axis=1)
    Hzx = np.gradient(grad_z, axis=2)
    Hxy = np.gradient(grad_x, axis=1)  # = Hyx by symmetry

    # Scale normalization (sigma^2 for scale invariance)
    scale = sigma ** 2
    Hzz *= scale
    Hyy *= scale
    Hxx *= scale
    Hzy *= scale
    Hzx *= scale
    Hxy *= scale

    # For 3x3 symmetric matrix, compute eigenvalues using analytical formula
    # This avoids the slow per-voxel np.linalg.eigvalsh calls
    #
    # The characteristic polynomial for a 3x3 symmetric matrix is:
    # λ³ - tr(H)λ² + (sum of 2x2 minors)λ - det(H) = 0
    #
    # We use Cardano's formula for cubic equations

    # Trace
    trace = Hzz + Hyy + Hxx

    # Sum of 2x2 principal minors (coefficient of λ)
    # M = Hzz*Hyy + Hyy*Hxx + Hzz*Hxx - Hzy² - Hzx² - Hxy²
    minor_sum = (Hzz * Hyy + Hyy * Hxx + Hzz * Hxx
                 - Hzy**2 - Hzx**2 - Hxy**2)

    # Determinant using rule of Sarrus for symmetric matrix
    det = (Hzz * (Hyy * Hxx - Hxy**2)
           - Hzy * (Hzy * Hxx - Hxy * Hzx)
           + Hzx * (Hzy * Hxy - Hyy * Hzx))

    # Convert to depressed cubic: t³ + pt + q = 0
    # where λ = t + trace/3
    p = minor_sum - trace**2 / 3
    q = 2 * trace**3 / 27 - trace * minor_sum / 3 + det

    # Discriminant
    discriminant = (q / 2)**2 + (p / 3)**3

    # For symmetric matrices, we always have 3 real eigenvalues
    # Use trigonometric solution
    # Handle edge cases where p is near zero
    eps = 1e-30
    p_safe = np.where(np.abs(p) < eps, -eps, p)

    # r = sqrt(-p/3), phi = arccos(-q/(2r³))
    r = np.sqrt(np.maximum(-p_safe / 3, 0))
    r_cubed = r**3
    r_cubed_safe = np.where(np.abs(r_cubed) < eps, eps, r_cubed)

    cos_arg = np.clip(-q / (2 * r_cubed_safe), -1, 1)
    phi = np.arccos(cos_arg)

    # Three roots
    t1 = 2 * r * np.cos(phi / 3)
    t2 = 2 * r * np.cos((phi + 2 * np.pi) / 3)
    t3 = 2 * r * np.cos((phi + 4 * np.pi) / 3)

    # Convert back to λ
    offset = trace / 3
    lambda_a = t1 + offset
    lambda_b = t2 + offset
    lambda_c = t3 + offset

    # Stack and sort by absolute value
    eigenvalues = np.stack([lambda_a, lambda_b, lambda_c], axis=0)

    # Sort by absolute value along first axis
    abs_eigenvalues = np.abs(eigenvalues)
    sort_indices = np.argsort(abs_eigenvalues, axis=0)

    # Use advanced indexing to sort
    shape = volume.shape
    i_grid, j_grid, k_grid = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing='ij'
    )

    lambda1 = eigenvalues[sort_indices[0], i_grid, j_grid, k_grid]
    lambda2 = eigenvalues[sort_indices[1], i_grid, j_grid, k_grid]
    lambda3 = eigenvalues[sort_indices[2], i_grid, j_grid, k_grid]

    return lambda1, lambda2, lambda3


def surfaceness_filter(volume, sigmas=range(1, 4), alpha=0.5, beta=0.5, gamma=None):
    """
    Compute surfaceness (plate-like structure) filter.

    This is a modified Frangi filter that enhances sheet-like structures
    instead of tubular structures (vessels).

    For surfaces/sheets: |λ1| ≈ |λ2| ≈ 0, |λ3| >> 0
    (Two small eigenvalues, one large eigenvalue)

    This differs from vesselness where: |λ1| ≈ 0, |λ2| ≈ |λ3| >> 0
    (One small eigenvalue, two large eigenvalues)

    This implementation uses vectorized eigenvalue computation for performance.

    Args:
        volume: 3D probability volume
        sigmas: Range of scales for multi-scale analysis
        alpha: Sensitivity to plate-like vs line-like structures
        beta: Sensitivity to blob-like structures
        gamma: Sensitivity to background noise (auto-computed if None)

    Returns:
        Surface-enhanced volume
    """
    import numpy as np

    volume = np.asarray(volume, dtype=np.float64)

    if gamma is None:
        # Auto-compute gamma based on half the Frobenius norm
        gamma = 0.5 * np.sqrt(np.sum(volume ** 2))
        if gamma == 0:
            gamma = 1.0

    result = np.zeros_like(volume)

    for sigma in sigmas:
        # Compute Hessian eigenvalues (vectorized)
        lambda1, lambda2, lambda3 = compute_hessian_eigenvalues_vectorized(volume, sigma)

        # Avoid division by zero
        eps = 1e-10

        # For surfaceness, we want:
        # - λ1 and λ2 to be small (the two directions along the surface)
        # - λ3 to be large and negative (the normal direction, for bright sheets)

        # Ra: Ratio for distinguishing plate from line
        # For plates: |λ1| ≈ |λ2| << |λ3|, so Ra = |λ2|/|λ3| should be small
        # For lines: |λ1| << |λ2| ≈ |λ3|, so this ratio is ~1
        Ra = np.abs(lambda2) / (np.abs(lambda3) + eps)

        # Rb: Ratio for distinguishing plate from blob
        # For plates: |λ1| ≈ |λ2| << |λ3|
        # For blobs: |λ1| ≈ |λ2| ≈ |λ3|
        # Rb = |λ1| / sqrt(|λ2 * λ3|)
        Rb = np.abs(lambda1) / (np.sqrt(np.abs(lambda2 * lambda3)) + eps)

        # S: "Second order structureness" - Frobenius norm
        S = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)

        # Surfaceness response
        # High when Ra is small (plate, not line)
        # High when Rb is small (plate, not blob)
        # High when S is significant (not background)
        #
        # Key difference from vesselness:
        # - Vesselness uses (1 - exp(-Ra²/...)) which is high when Ra ≈ 1 (tube-like)
        # - Surfaceness uses exp(-Ra²/...) which is high when Ra << 1 (plate-like)
        response = (
            np.exp(-(Ra ** 2) / (2 * alpha ** 2)) *         # Plate vs line (Ra small = plate)
            np.exp(-(Rb ** 2) / (2 * beta ** 2)) *          # Plate vs blob (Rb small = not blob)
            (1 - np.exp(-(S ** 2) / (2 * gamma ** 2)))      # Structureness (S large = structure)
        )

        # For bright sheets on dark background, λ3 should be negative
        # (Hessian is negative in the direction of maximum curvature for bright structures)
        response = np.where(lambda3 > 0, 0, response)

        # Take maximum across scales
        result = np.maximum(result, response)

    return result


def surfaceness_frangi(volume, sigmas=range(1, 4)):
    """
    Apply surfaceness filter (modified Frangi for sheet-like structures).

    This is the Host Baseline post-processing filter that enhances
    "surfaceness" rather than "vesselness".

    Uses vectorized eigenvalue computation for ~100x speedup over naive implementation.

    Args:
        volume: 3D probability volume
        sigmas: Range of scales for multi-scale analysis

    Returns:
        Surface-enhanced volume
    """
    return surfaceness_filter(volume, sigmas=sigmas)


def postprocess_volume(
    probability_map,
    threshold: float = 0.75,
    apply_frangi: bool = True,
    frangi_threshold: float = 0.5,
    min_component_size: int = 100
):
    """
    Apply Host Baseline post-processing to nnUNet probability output.

    This implements the host's post-processing that improved LB from 0.543 to 0.562.

    Args:
        probability_map: Softmax probabilities, shape (C, D, H, W) or (D, H, W)
        threshold: Probability threshold (default 0.75 per host description)
        apply_frangi: Whether to apply Frangi surfaceness filter
        frangi_threshold: Threshold after Frangi filtering
        min_component_size: Minimum connected component size to keep

    Returns:
        Binary segmentation mask (uint8)
    """
    import numpy as np
    from scipy import ndimage

    # Extract surface class probability if multi-channel
    if probability_map.ndim == 4:
        # Shape is (C, D, H, W) - class 1 is surface
        if probability_map.shape[0] <= 3:
            surface_prob = probability_map[1]  # Class 1 = surface
        else:
            # Assume (D, H, W, C) format
            surface_prob = probability_map[..., 1]
    else:
        surface_prob = probability_map

    # Step 1: Threshold at 0.75
    thresholded = surface_prob.copy()
    thresholded[thresholded < threshold] = 0

    if not apply_frangi:
        return (thresholded > 0).astype(np.uint8)

    # Step 2: Apply Frangi surfaceness filter
    enhanced = surfaceness_frangi(thresholded)

    # Normalize and threshold
    if enhanced.max() > 0:
        enhanced = enhanced / enhanced.max()

    result = (enhanced > frangi_threshold).astype(np.uint8)

    # Step 3: Remove small connected components
    labeled, num_features = ndimage.label(result)
    if num_features > 0:
        sizes = ndimage.sum(result, labeled, range(1, num_features + 1))
        mask = np.isin(labeled, np.where(np.array(sizes) >= min_component_size)[0] + 1)
        result = result * mask

    return result.astype(np.uint8)


def load_probabilities(npz_path: Path):
    """Load probability maps from nnUNet inference."""
    import numpy as np
    data = np.load(npz_path)
    return data['probabilities']


def postprocess_predictions(
    pred_dir: Path,
    output_dir: Path,
    threshold: float = 0.75,
    apply_frangi: bool = True,
    frangi_threshold: float = 0.5,
    min_component_size: int = 100
) -> Path:
    """
    Apply Host Baseline post-processing to all predictions in a directory.

    Processes .npz files containing probability maps from nnUNet inference
    (requires save_probabilities=True during inference).

    Args:
        pred_dir: Directory containing nnUNet predictions (.npz files)
        output_dir: Directory to save post-processed results
        threshold: Probability threshold (default 0.75)
        apply_frangi: Whether to apply Frangi filter
        frangi_threshold: Threshold after Frangi filtering
        min_component_size: Minimum component size to keep

    Returns:
        Path to output directory
    """
    import tifffile
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find probability files
    npz_files = list(pred_dir.glob("*.npz"))

    if not npz_files:
        print(f"No .npz probability files found in {pred_dir}")
        print("Ensure inference was run with save_probabilities=True")
        return output_dir

    print(f"Post-processing {len(npz_files)} predictions...")
    print(f"  Threshold: {threshold}")
    print(f"  Frangi filter: {apply_frangi}")

    for npz_path in tqdm(npz_files, desc="Post-processing"):
        # Load probability map
        probs = load_probabilities(npz_path)

        # Apply post-processing
        result = postprocess_volume(
            probs,
            threshold=threshold,
            apply_frangi=apply_frangi,
            frangi_threshold=frangi_threshold,
            min_component_size=min_component_size
        )

        # Save result
        output_path = output_dir / f"{npz_path.stem}.tif"
        tifffile.imwrite(str(output_path), result, compression='zlib')

    print(f"Post-processed predictions saved to: {output_dir}")
    return output_dir


def predictions_to_tiff(pred_dir: Path, output_dir: Path):
    """Convert nnUNet predictions to TIFF files (no post-processing)."""
    import numpy as np
    import tifffile
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = list(pred_dir.glob("*.npz"))
    tif_files = list(pred_dir.glob("*.tif"))

    if npz_files:
        print(f"Converting {len(npz_files)} NPZ files to TIFF...")
        for npz_path in tqdm(npz_files, desc="Converting"):
            case_id = npz_path.stem
            data = np.load(npz_path)
            probs = data['probabilities']
            pred = np.argmax(probs, axis=0).astype(np.uint8)
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    elif tif_files:
        print(f"Copying {len(tif_files)} TIFF files...")
        for tif_path in tqdm(tif_files, desc="Copying"):
            case_id = tif_path.stem
            pred = tifffile.imread(str(tif_path)).astype(np.uint8)
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    else:
        print(f"No prediction files found in {pred_dir}")


def predictions_to_tiff_with_postprocess(
    pred_dir: Path,
    output_dir: Path,
    apply_postprocess: bool = True,
    threshold: float = 0.75,
    apply_frangi: bool = True
) -> Path:
    """
    Convert nnUNet predictions to TIFF with optional post-processing.

    This is a convenience function that combines predictions_to_tiff
    and postprocess_predictions.

    Args:
        pred_dir: Directory containing nnUNet predictions
        output_dir: Directory to save TIFF files
        apply_postprocess: Whether to apply Host Baseline post-processing
        threshold: Probability threshold for post-processing
        apply_frangi: Whether to apply Frangi filter

    Returns:
        Path to output directory
    """
    if apply_postprocess:
        # Check for NPZ files (probability maps required for post-processing)
        npz_files = list(pred_dir.glob("*.npz"))
        if npz_files:
            return postprocess_predictions(
                pred_dir, output_dir,
                threshold=threshold,
                apply_frangi=apply_frangi
            )
        else:
            print("WARNING: No .npz files found, falling back to standard conversion")
            print("For post-processing, run inference with save_probabilities=True")

    # Standard conversion without post-processing
    predictions_to_tiff(pred_dir, output_dir)
    return output_dir


def generate_submission(
    predictions_dir: Path,
    output_zip: Path,
    delete_after_zip: bool = False
) -> Optional[Path]:
    """Create submission ZIP from TIFF predictions."""
    import zipfile
    from tqdm import tqdm

    if not predictions_dir.exists():
        print(f"ERROR: Predictions directory not found: {predictions_dir}")
        return None

    tiff_files = sorted(predictions_dir.glob("*.tif"))
    if not tiff_files:
        print(f"No TIFF files found in {predictions_dir}")
        return None

    print(f"Creating submission ZIP with {len(tiff_files)} files...")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for tiff_path in tqdm(tiff_files, desc="Zipping"):
            zipf.write(tiff_path, tiff_path.name)
            if delete_after_zip:
                tiff_path.unlink()

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Submission saved: {output_zip} ({zip_size_mb:.1f} MB)")
    return output_zip


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def full_pipeline(
    do_prepare_raw: bool = True,
    do_train: bool = True,
    do_inference: bool = True,
    config: str = "3d_fullres",
    fold: Union[int, str] = "all",
    planner: str = "nnUNetPlannerResEncM",
    plans: str = "nnUNetResEncUNetMPlans",
    epochs: Optional[int] = None,
    trainer: str = "nnUNetTrainer",
    continue_training: bool = False,
    pretrained_weights: Optional[Path] = None,
    num_gpus: Optional[int] = None,
    save_probabilities: bool = True,
    # Post-processing options (Host Baseline)
    apply_postprocess: bool = False,
    postprocess_threshold: float = 0.75,
    apply_frangi: bool = True,
    timeout: Optional[int] = None,
    max_cases: Optional[int] = None,
    seed: Optional[int] = None,
) -> bool:
    """
    Run complete training/inference pipeline.

    Args:
        do_prepare_raw: Prepare raw dataset
        do_train: Run training
        do_inference: Run inference
        config: nnUNet configuration (3d_fullres, 2d, 3d_lowres)
        fold: Fold number (0-4) or "all"
        planner: Planner class name
        plans: Plans name
        epochs: Number of training epochs
        trainer: Trainer class name
        continue_training: Continue from last checkpoint
        num_gpus: Number of GPUs
        save_probabilities: Save probability maps during inference
        apply_postprocess: Apply Host Baseline post-processing
        postprocess_threshold: Probability threshold (default 0.75)
        apply_frangi: Apply Frangi surfaceness filter
        timeout: Command timeout in seconds
        max_cases: Limit number of training cases

    Returns:
        True if pipeline completed successfully
    """

    if num_gpus is None:
        num_gpus = get_gpu_count() or 1

    print("=" * 60)
    print("Vesuvius nnUNet Training & Inference - Local")
    print("=" * 60)
    print(f"Config: {config}")
    print(f"Fold: {fold}")
    print(f"Epochs: {epochs or 1000}")
    print(f"GPUs: {num_gpus}")
    print(f"Plans: {plans}")
    print(f"Trainer: {trainer}")
    if apply_postprocess:
        print(f"Post-processing: threshold={postprocess_threshold}, frangi={apply_frangi}")
    print("=" * 60)

    # Setup
    print("\n[1/5] Setting up environment...")
    setup_environment()

    # Install custom trainer if using non-standard trainer
    if trainer != "nnUNetTrainer":
        print(f"\nInstalling custom trainer: {trainer}")
        install_custom_trainer()

    # Check if preprocessed data exists
    preprocessed_dataset = NNUNET_PREPROCESSED / DATASET_NAME
    if not preprocessed_dataset.exists():
        print(f"\nERROR: Preprocessed data not found at {preprocessed_dataset}")
        print("Run preprocessing first:")
        print("  ./run-preprocessing-nohup.sh")
        return False

    print(f"Using preprocessed data: {preprocessed_dataset}")

    # Prepare raw data (symlinks only - fast)
    if do_prepare_raw:
        print("\n[2/5] Preparing raw dataset (symlinks)...")
        raw_dataset = NNUNET_RAW / DATASET_NAME
        if not raw_dataset.exists():
            prepare_raw_dataset(DATASET_DIR, max_cases=max_cases)
        else:
            print(f"Raw dataset exists: {raw_dataset}")
    else:
        print("\n[2/5] Skipping raw data preparation...")

    # Training
    if do_train:
        print("\n[3/5] Training...")
        success = run_training(
            config=config,
            fold=fold,
            plans=plans,
            epochs=epochs,
            trainer=trainer,
            continue_training=continue_training,
            pretrained_weights=pretrained_weights,
            num_gpus=num_gpus,
            timeout=timeout,
            seed=seed
        )
        if not success:
            print("Training failed!")
            return False
    else:
        print("\n[3/5] Skipping training...")

    # Inference
    if do_inference:
        print("\n[4/5] Running inference...")

        # Prepare test data
        test_input_dir = NNUNET_WORK / "test_input"
        prepare_test_data(DATASET_DIR, test_input_dir)

        # Run inference
        predictions_dir = NNUNET_WORK / "predictions"
        success = run_inference(
            test_input_dir,
            predictions_dir,
            config=config,
            fold=fold,
            plans=plans,
            epochs=epochs,
            trainer=trainer,
            save_probabilities=save_probabilities,
            timeout=timeout
        )
        if not success:
            print("Inference failed!")
            return False

        # Convert to TIFF (with optional post-processing)
        print("\n[5/5] Converting predictions...")
        tiff_output_dir = NNUNET_OUTPUT / "predictions_tiff"

        if apply_postprocess:
            print("Applying Host Baseline post-processing...")
            predictions_to_tiff_with_postprocess(
                predictions_dir, tiff_output_dir,
                apply_postprocess=True,
                threshold=postprocess_threshold,
                apply_frangi=apply_frangi
            )
        else:
            predictions_to_tiff(predictions_dir, tiff_output_dir)

        print(f"Predictions saved to: {tiff_output_dir}")

        # Generate submission
        submission_zip = NNUNET_OUTPUT / "submission.zip"
        generate_submission(tiff_output_dir, submission_zip)
    else:
        print("\n[4/5] Skipping inference...")
        print("\n[5/5] Skipping post-processing...")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    # Show output location
    if do_train:
        model_dir = get_training_output_dir(epochs, plans, config, fold, trainer)
        print(f"\nModel saved to: {model_dir}")
        progress_img = model_dir / "progress.png"
        if progress_img.exists():
            print(f"Training progress: {progress_img}")

    return True


def run_host_baseline(
    epochs: Optional[int] = None,
    apply_postprocess: bool = True,
    do_train: bool = True,
    do_inference: bool = True,
    use_host_plans: bool = True,
    debug_mode: bool = False,
    **kwargs
) -> bool:
    """
    Convenience function to run the Host Baseline configuration.

    This sets up the correct trainer and post-processing for the
    Host Baseline (0.543 LB raw, 0.562 w/ post-processing).

    Host Baseline components:
    - Trainer: nnUNetTrainerMedialSurfaceRecall (Skeleton Recall Loss)
    - Plans: nnUNetHostBaselinePlans (exact host configuration)
      - patch_size: [192, 192, 192]
      - batch_size: 2
      - n_blocks_per_stage: [1, 3, 4, 6, 6, 6]
      - features_per_stage: [32, 64, 128, 256, 320, 320]
    - Config: 3d_fullres
    - Epochs: ~1200 (host used), default to 1000 for nnUNet compatibility
    - LR: 0.01
    - Post-processing: threshold 0.75 + Frangi surfaceness filter

    Args:
        epochs: Training epochs (None = 1000 default)
        apply_postprocess: Apply threshold + Frangi post-processing
        do_train: Run training
        do_inference: Run inference
        use_host_plans: Use exact host plans configuration (recommended)
        debug_mode: Use smaller patch_size/batch_size to prevent OOM
        **kwargs: Additional arguments passed to full_pipeline()

    Returns:
        True if pipeline completed successfully
    """
    print("=" * 60)
    if debug_mode:
        print("HOST BASELINE MODE (DEBUG)")
    else:
        print("HOST BASELINE MODE")
    print("=" * 60)
    print("Using Host Baseline configuration:")
    print("  - Trainer: nnUNetTrainerMedialSurfaceRecall")
    print("  - Loss: Dice + CE + Skeleton Recall (3-axis medial surface)")
    print("  - Config: 3d_fullres")
    if use_host_plans:
        print("  - Plans: nnUNetHostBaselinePlans (exact host config)")
        print("    - patch_size: [192, 192, 192]")
        print("    - batch_size: 2")
        print("    - n_blocks_per_stage: [1, 3, 4, 6, 6, 6]")
    else:
        print("  - Plans: nnUNetResEncUNetMPlans (auto-configured)")
    print(f"  - Epochs: {epochs or 1000}")
    print(f"  - Learning rate: {HOST_BASELINE_LR}")
    print(f"  - Post-processing: {'enabled' if apply_postprocess else 'disabled'}")
    if apply_postprocess:
        print(f"    - Threshold: {HOST_BASELINE_THRESHOLD}")
        print("    - Frangi surfaceness filter: enabled")
    print("=" * 60)

    # Determine which plans to use
    if use_host_plans:
        plans_name = "nnUNetHostBaselinePlans"
        # Create host baseline plans file
        print("\nCreating Host Baseline plans...")
        create_host_baseline_plans(debug_mode=debug_mode)
    else:
        plans_name = "nnUNetResEncUNetMPlans"

    return full_pipeline(
        trainer=HOST_BASELINE_TRAINER,
        plans=plans_name,
        config="3d_fullres",
        epochs=epochs,
        do_train=do_train,
        do_inference=do_inference,
        apply_postprocess=apply_postprocess,
        postprocess_threshold=HOST_BASELINE_THRESHOLD,
        apply_frangi=True,
        save_probabilities=True,  # Required for post-processing
        **kwargs
    )


def main():
    parser = argparse.ArgumentParser(
        description="Vesuvius nnUNet Training & Inference - Local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug mode (fast verification)
  python surface-nnunet-training-local.py --debug --train
  python surface-nnunet-training-local.py --debug --host-baseline --train

  # Host Baseline (recommended for best results)
  python surface-nnunet-training-local.py --host-baseline --epochs 1000

  # Host Baseline with custom epochs
  python surface-nnunet-training-local.py --host-baseline --epochs 100

  # Training only (50 epochs)
  python surface-nnunet-training-local.py --train --epochs 50

  # Inference only
  python surface-nnunet-training-local.py --inference

  # Full pipeline with post-processing
  python surface-nnunet-training-local.py --train --inference --postprocess

  # Full pipeline with 2 GPUs
  python surface-nnunet-training-local.py --train --inference --gpus 2

  # Continue training from checkpoint (same trainer, same epoch limit)
  python surface-nnunet-training-local.py --train --continue-training

  # Initialize from pretrained weights (new training with reset LR scheduler)
  python surface-nnunet-training-local.py --train --epochs 4000 --fold 0 \\
    --pretrained-weights /path/to/checkpoint_final.pth

Debug Mode:
  The --debug flag enables fast verification with minimal data:
  - Uses only 1 training case (configurable with --debug-cases)
  - Runs only 1 epoch
  - Can be combined with --host-baseline for debugging custom trainer

Host Baseline:
  The --host-baseline flag enables the competition host's solution:
  - Custom Trainer: nnUNetTrainerMedialSurfaceRecall (Skeleton Recall Loss)
  - Plans: nnUNetHostBaselinePlans (exact host configuration)
    - patch_size: [192, 192, 192]
    - batch_size: 2
    - n_blocks_per_stage: [1, 3, 4, 6, 6, 6]
  - Loss: Dice + CE + Skeleton Recall (3-axis medial surface)
  - Learning rate: 0.01
  - Post-processing: threshold 0.75 + Frangi surfaceness filter
  - Achieves 0.543 LB raw, 0.562 with post-processing

  Use --no-host-plans to use auto-configured plans instead of exact host plans.
"""
    )

    # Special modes
    parser.add_argument("--host-baseline", action="store_true",
                        help="Run Host Baseline configuration (recommended)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: fast verification with small data (1 case, 1 epoch)")
    parser.add_argument("--debug-cases", type=int, default=DEBUG_MAX_CASES,
                        help=f"Number of cases for debug mode (default: {DEBUG_MAX_CASES})")

    # Pipeline stages
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument("--skip-raw-prep", action="store_true", help="Skip raw data preparation")

    # Training configuration
    parser.add_argument("--config", default="3d_fullres",
                        choices=["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"],
                        help="nnUNet configuration (default: 3d_fullres)")
    parser.add_argument("--fold", default="all",
                        help="Fold number (0-4) or 'all' (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (1,5,10,20,50,100,250,500,750,1000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs (default: auto-detect)")
    parser.add_argument("--continue-training", action="store_true",
                        help="Continue from last checkpoint")
    parser.add_argument("--pretrained-weights", type=str, default=None,
                        help="Path to pretrained weights (.pth) to initialize from")

    # Post-processing options (Host Baseline)
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply Host Baseline post-processing (threshold + Frangi)")
    parser.add_argument("--postprocess-threshold", type=float, default=0.75,
                        help="Probability threshold for post-processing (default: 0.75)")
    parser.add_argument("--no-frangi", action="store_true",
                        help="Disable Frangi filter in post-processing")

    # Advanced options
    parser.add_argument("--trainer", default="nnUNetTrainer",
                        help="Trainer class (default: nnUNetTrainer)")
    parser.add_argument("--plans", default="nnUNetResEncUNetMPlans",
                        help="Plans name (default: nnUNetResEncUNetMPlans)")
    parser.add_argument("--no-host-plans", action="store_true",
                        help="Use auto-configured plans instead of exact host plans (with --host-baseline)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Command timeout in seconds")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of training cases")

    args = parser.parse_args()

    # Debug mode: override settings for fast verification
    if args.debug:
        print("=" * 60)
        print("DEBUG MODE")
        print("=" * 60)
        print(f"  - Epochs: {DEBUG_EPOCHS}")
        print(f"  - Max cases: {args.debug_cases}")
        print("=" * 60)
        # Override settings
        if args.epochs is None:
            args.epochs = DEBUG_EPOCHS
        if args.max_cases is None:
            args.max_cases = args.debug_cases
        if args.gpus is None:
            args.gpus = 1  # Use single GPU to avoid DDP batch size issues
        # Set environment for faster validation
        os.environ["nnUNet_n_proc_DA"] = "2"
        os.environ["nnUNet_compile"] = "false"  # Disable torch.compile for faster startup

    # Host Baseline mode
    if args.host_baseline:
        # Convert fold to int if numeric
        fold = args.fold
        if fold.isdigit():
            fold = int(fold)

        # Convert pretrained_weights to Path if provided
        pretrained_weights = Path(args.pretrained_weights) if args.pretrained_weights else None

        success = run_host_baseline(
            epochs=args.epochs,
            apply_postprocess=True,
            do_train=args.train or (not args.train and not args.inference),  # Default to train
            do_inference=args.inference or (not args.train and not args.inference),  # Default to both
            use_host_plans=not args.no_host_plans,
            debug_mode=args.debug,
            do_prepare_raw=not args.skip_raw_prep,
            fold=fold,
            num_gpus=args.gpus,
            continue_training=args.continue_training,
            pretrained_weights=pretrained_weights,
            timeout=args.timeout,
            max_cases=args.max_cases,
            seed=args.seed,
        )
        sys.exit(0 if success else 1)

    # Default mode
    if not args.train and not args.inference:
        print("No action specified. Use --train and/or --inference, or use --host-baseline")
        parser.print_help()
        sys.exit(1)

    # Convert fold to int if numeric
    fold = args.fold
    if fold.isdigit():
        fold = int(fold)

    # Convert pretrained_weights to Path if provided
    pretrained_weights = Path(args.pretrained_weights) if args.pretrained_weights else None

    success = full_pipeline(
        do_prepare_raw=not args.skip_raw_prep,
        do_train=args.train,
        do_inference=args.inference,
        config=args.config,
        fold=fold,
        plans=args.plans,
        epochs=args.epochs,
        trainer=args.trainer,
        continue_training=args.continue_training,
        pretrained_weights=pretrained_weights,
        num_gpus=args.gpus,
        apply_postprocess=args.postprocess,
        postprocess_threshold=args.postprocess_threshold,
        apply_frangi=not args.no_frangi,
        timeout=args.timeout,
        max_cases=args.max_cases,
        seed=args.seed,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

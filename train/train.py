#!/usr/bin/env python3
"""
Vesuvius Surface 3D Detection Training Script (Keras Multi-GPU)

Based on train-vesuvius-surface-3d-detection-on-tpu_v3.ipynb
Uses Keras high-level API with torch backend for multi-GPU training.

Usage:
    python train.py --config configs/exp001_baseline.yaml
"""

import os
import sys
import argparse
import glob
import warnings
import logging
from datetime import datetime

import yaml

warnings.filterwarnings('ignore')

# Set Keras backend BEFORE importing keras
os.environ["KERAS_BACKEND"] = "torch"

# Disable TensorFlow GPU (only used for TFRecord loading)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
import torch

import keras
from keras import ops
from keras.optimizers import AdamW
from keras.optimizers.schedules import CosineDecay
from keras.callbacks import Callback

import medicai
from medicai.transforms import (
    Compose,
    NormalizeIntensity,
    RandShiftIntensity,
    RandRotate90,
    RandRotate,
    RandFlip,
    RandCutOut,
    RandSpatialCrop
)
from medicai.models import TransUNet, SegFormer
from medicai.losses import SparseDiceCELoss, SparseCenterlineDiceLoss
from medicai.metrics import SparseDiceMetric
from medicai.callbacks import SlidingWindowInferenceCallback
from medicai.utils import SlidingWindowInference

# Import topometrics for competition evaluation
sys.path.insert(0, '/workspace/topological-metrics-kaggle')
try:
    from topometrics import compute_leaderboard_score
    HAS_TOPOMETRICS = True
except ImportError:
    HAS_TOPOMETRICS = False
    print("Warning: topometrics not available. Competition metrics will be skipped.")


# =============================================================================
# Config Loading
# =============================================================================
def load_config(config_path):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(exp_name, output_dir):
    """Setup logging to file and console."""
    log_path = os.path.join(output_dir, f"{exp_name}.log")

    # Create logger
    logger = logging.getLogger('vesuvius')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# Data Loading (TFRecord → tf.data)
# =============================================================================
def parse_tfrecord_fn(example):
    """Parse TFRecord format."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
        "image_shape": tf.io.FixedLenFeature([3], tf.int64),
        "label_shape": tf.io.FixedLenFeature([3], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_raw(parsed_example["image"], tf.uint8)
    label = tf.io.decode_raw(parsed_example["label"], tf.uint8)
    image_shape = tf.cast(parsed_example["image_shape"], tf.int64)
    label_shape = tf.cast(parsed_example["label_shape"], tf.int64)
    image = tf.reshape(image, image_shape)
    label = tf.reshape(label, label_shape)
    return image, label


def prepare_inputs(image, label):
    """Add channel dimension and convert to float32."""
    image = image[..., None]
    label = label[..., None]
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def get_train_transform(input_shape, cldice_iters):
    """Get training augmentation pipeline."""
    return Compose([
        RandSpatialCrop(
            keys=["image", "label"],
            roi_size=input_shape,
            random_center=True,
            random_size=False,
            invalid_label=2,
            min_valid_ratio=0.5,
            max_attempts=10
        ),
        RandFlip(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlip(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlip(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandRotate90(
            keys=["image", "label"],
            prob=0.4,
            max_k=3,
            spatial_axes=(0, 1)
        ),
        RandRotate(
            keys=["image", "label"],
            factor=0.2,
            prob=0.7,
            fill_mode="crop",
        ),
        NormalizeIntensity(
            keys=["image"],
            nonzero=True,
            channel_wise=False
        ),
        RandShiftIntensity(
            keys=["image"],
            offsets=0.10,
            prob=0.5
        ),
        RandCutOut(
            keys=["image", "label"],
            invalid_label=2,
            mask_size=[input_shape[1] // 4, input_shape[2] // 4],
            fill_mode="constant",
            cutout_mode='volume',
            prob=0.8,
            num_cuts=5,
        ),
    ])


def get_val_transform():
    """Get validation transform (normalization only)."""
    return Compose([
        NormalizeIntensity(
            keys=["image"],
            nonzero=True,
            channel_wise=False
        ),
    ])


def train_transformation(image, label, transform):
    data = {"image": image, "label": label}
    result = transform(data)
    return result["image"], result["label"]


def val_transformation(image, label, transform):
    data = {"image": image, "label": label}
    result = transform(data)
    return result["image"], result["label"]


def create_dataset(tfrecord_files, input_shape, batch_size, is_train=True, cldice_iters=10):
    """Create tf.data dataset from TFRecord files."""
    dataset = tf.data.TFRecordDataset(tfrecord_files)

    if is_train:
        dataset = dataset.shuffle(buffer_size=100)

    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(prepare_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        transform = get_train_transform(input_shape, cldice_iters)
        dataset = dataset.map(
            lambda x, y: train_transformation(x, y, transform),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        transform = get_val_transform()
        dataset = dataset.map(
            lambda x, y: val_transformation(x, y, transform),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =============================================================================
# Epoch Logging Callback
# =============================================================================
class EpochLoggingCallback(Callback):
    """Callback to log metrics at the end of each epoch and track history."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self.epoch_history = []  # Store per-epoch metrics

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Store epoch metrics
        epoch_data = {
            'epoch': epoch + 1,
            **{k: float(v) for k, v in logs.items()}
        }
        self.epoch_history.append(epoch_data)

        # Log to console/file
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self._log(f"Epoch {epoch + 1} - {metrics_str}")


# =============================================================================
# Competition Metric Evaluation Callback
# =============================================================================
class CompetitionMetricCallback(Callback):
    """Callback to evaluate with official competition metrics."""

    def __init__(self, model, val_ds, input_shape, num_classes,
                 eval_interval=100, surface_tolerance=2.0, overlap=0.5,
                 voi_connectivity=26, voi_transform='one_over_one_plus',
                 voi_alpha=0.3, combine_weights=(0.3, 0.35, 0.35),
                 ignore_label=2, output_dir='.', logger=None):
        super().__init__()
        self.eval_model = model
        self.val_ds = val_ds
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.eval_interval = eval_interval
        self.surface_tolerance = surface_tolerance
        self.overlap = overlap
        self.voi_connectivity = voi_connectivity
        self.voi_transform = voi_transform
        self.voi_alpha = voi_alpha
        self.combine_weights = combine_weights
        self.ignore_label = ignore_label
        self.output_dir = output_dir
        self.logger = logger
        self.best_score = 0.0
        self.results_history = []

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def evaluate_competition_metrics(self, epoch):
        """Run competition metric evaluation."""
        if not HAS_TOPOMETRICS:
            self._log("Skipping competition metrics (topometrics not available)")
            return None

        self._log(f"Running competition metric evaluation at epoch {epoch}...")

        # Create sliding window inferencer
        swi = SlidingWindowInference(
            self.eval_model,
            num_classes=self.num_classes,
            roi_size=self.input_shape,
            mode='gaussian',
            sw_batch_size=1,
            overlap=self.overlap,
        )

        all_scores = []
        all_topo = []
        all_surface_dice = []
        all_voi = []

        for i, (x, y) in enumerate(self.val_ds):
            # Run inference
            y_pred = swi(x)

            # Convert to numpy and get segmentation
            pred_seg = y_pred.argmax(-1).astype(np.uint8).squeeze()
            gt_seg = y.numpy().astype(np.uint8).squeeze()

            # Compute competition score
            try:
                report = compute_leaderboard_score(
                    predictions=pred_seg,
                    labels=gt_seg,
                    dims=(0, 1, 2),
                    spacing=(1.0, 1.0, 1.0),
                    surface_tolerance=self.surface_tolerance,
                    voi_connectivity=self.voi_connectivity,
                    voi_transform=self.voi_transform,
                    voi_alpha=self.voi_alpha,
                    combine_weights=self.combine_weights,
                    fg_threshold=None,
                    ignore_label=self.ignore_label,
                    ignore_mask=None,
                )

                all_scores.append(report.score)
                all_topo.append(report.topo.toposcore)
                all_surface_dice.append(report.surface_dice)
                all_voi.append(report.voi.voi_score)
            except Exception as e:
                self._log(f"  Sample {i+1}: Error - {e}")
                continue

        if not all_scores:
            return None

        results = {
            'epoch': epoch,
            'mean_score': float(np.mean(all_scores)),
            'mean_topo': float(np.mean(all_topo)),
            'mean_surface_dice': float(np.mean(all_surface_dice)),
            'mean_voi': float(np.mean(all_voi)),
            'std_score': float(np.std(all_scores)),
            'n_samples': len(all_scores),
        }

        self._log(f"Competition Metrics at epoch {epoch}:")
        self._log(f"  Topo Score:     {results['mean_topo']:.4f}")
        self._log(f"  Surface Dice:   {results['mean_surface_dice']:.4f}")
        self._log(f"  VOI Score:      {results['mean_voi']:.4f}")
        self._log(f"  Combined Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")

        self.results_history.append(results)

        # Save if best
        if results['mean_score'] > self.best_score:
            self.best_score = results['mean_score']
            self._log(f"  New best competition score!")

        return results

    def on_epoch_end(self, epoch, logs=None):
        # Evaluate every eval_interval epochs (1-indexed for user display)
        if (epoch + 1) % self.eval_interval == 0:
            self.evaluate_competition_metrics(epoch + 1)

    def on_train_end(self, logs=None):
        # Results are now saved in comprehensive history.json
        pass


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Vesuvius Training Script')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--tfrecords', '-t', type=str, default='/workspace/tfrecords',
                        help='Path to TFRecords directory')
    parser.add_argument('--output', '-o', type=str, default='/workspace/experiments',
                        help='Base output directory')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: 1 epoch, 1%% of data, all evaluations enabled')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    exp_name = config['experiment']['name']

    # Create experiment directory
    exp_dir = os.path.join(args.output, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(exp_name, exp_dir)
    logger.info(f"{'=' * 60}")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Description: {config['experiment'].get('description', 'N/A')}")
    logger.info(f"{'=' * 60}")

    # Save config copy
    import shutil
    config_copy = os.path.join(exp_dir, f"{exp_name}.yaml")
    shutil.copy(args.config, config_copy)
    logger.info(f"Config saved to: {config_copy}")

    # Extract config values
    model_arch = config['model']['architecture']
    encoder = config['model']['encoder']

    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    warmup_ratio = config['training']['warmup_ratio']

    input_size = config['data']['input_size']
    val_fold = config['data']['val_fold']
    num_classes = config['data']['num_classes']

    cldice_iters = config['loss']['cldice_iters']
    cldice_weight = config['loss']['cldice_weight']

    val_interval = config['evaluation']['val_interval']
    eval_interval = config['evaluation']['eval_interval']
    overlap = config['evaluation']['overlap']
    surface_tolerance = config['evaluation']['surface_tolerance']

    # Competition metric settings (with defaults for backward compatibility)
    comp_config = config.get('competition', {})
    voi_connectivity = comp_config.get('voi_connectivity', 26)
    voi_transform = comp_config.get('voi_transform', 'one_over_one_plus')
    voi_alpha = comp_config.get('voi_alpha', 0.3)
    combine_weights_dict = comp_config.get('combine_weights', {'topo': 0.3, 'surface_dice': 0.35, 'voi': 0.35})
    combine_weights = (combine_weights_dict['topo'], combine_weights_dict['surface_dice'], combine_weights_dict['voi'])
    ignore_label = comp_config.get('ignore_label', 2)

    input_shape = (input_size, input_size, input_size)

    # ==========================================================================
    # Debug Mode Overrides
    # ==========================================================================
    debug_mode = args.debug
    if debug_mode:
        logger.info("=" * 60)
        logger.info("DEBUG MODE ENABLED")
        logger.info("=" * 60)
        epochs = 1
        val_interval = 1      # Validate every epoch
        eval_interval = 1     # Competition eval every epoch

    # ==========================================================================
    # Setup GPU
    # ==========================================================================
    num_devices = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"PyTorch devices: {num_devices} GPU(s)")
    if num_devices > 0:
        for i in range(num_devices):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Note: Keras with torch backend currently uses only 1 GPU
    # TODO: Implement DataParallel for multi-GPU support
    active_devices = 1
    total_batch_size = batch_size * active_devices

    logger.info(f"Keras version: {keras.version()}")
    logger.info(f"Backend: {keras.config.backend()}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Batch size per GPU: {batch_size}")
    logger.info(f"Total batch size: {total_batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Model: {model_arch} ({encoder})")
    logger.info(f"clDice iters: {cldice_iters}, weight: {cldice_weight}")
    logger.info(f"Competition weights: topo={combine_weights[0]}, surf_dice={combine_weights[1]}, voi={combine_weights[2]}")

    # ==========================================================================
    # Load Data
    # ==========================================================================
    all_tfrec = sorted(
        glob.glob(os.path.join(args.tfrecords, "*.tfrec")),
        key=lambda x: int(x.split("_")[-1].replace(".tfrec", ""))
    )

    if len(all_tfrec) == 0:
        raise FileNotFoundError(f"No .tfrec files found in {args.tfrecords}")

    # Split train/val
    if val_fold == -1:
        val_files = [all_tfrec[-1]]
        train_files = all_tfrec[:-1]
    else:
        val_files = [all_tfrec[val_fold]]
        train_files = [f for i, f in enumerate(all_tfrec) if i != val_fold]

    # Debug mode: use only ~1% of training data
    if debug_mode:
        num_debug_files = max(1, len(train_files) // 100)
        train_files = train_files[:num_debug_files]
        logger.info(f"DEBUG: Using {num_debug_files} training file(s) (~1%)")

    logger.info(f"Training files: {len(train_files)}")
    logger.info(f"Validation file: {os.path.basename(val_files[0])}")

    num_train_samples = len(train_files) * 6
    logger.info(f"Training samples: ~{num_train_samples}")

    train_ds = create_dataset(
        train_files, input_shape, total_batch_size, is_train=True, cldice_iters=cldice_iters
    )
    val_ds = create_dataset(
        val_files, input_shape, batch_size=1, is_train=False
    )

    # Debug mode: limit validation samples
    if debug_mode:
        val_ds = val_ds.take(1)  # Only 1 validation sample
        logger.info("DEBUG: Using 1 validation sample")

    # ==========================================================================
    # Create Model
    # ==========================================================================
    if model_arch == 'transunet':
        model = TransUNet(
            encoder_name=encoder,
            input_shape=input_shape + (1,),
            num_classes=num_classes,
            classifier_activation='softmax'
        )
    elif model_arch == 'segformer':
        model = SegFormer(
            input_shape=input_shape + (1,),
            encoder_name=encoder,
            classifier_activation='softmax',
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    logger.info(f"Model parameters: {model.count_params() / 1e6:.2f}M")

    # Build model
    logger.info("Building model...")
    dummy_input = torch.zeros(1, *input_shape, 1).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    del dummy_input
    torch.cuda.empty_cache()
    logger.info(f"GPU memory after build: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ==========================================================================
    # Setup Training
    # ==========================================================================
    steps_per_epoch = num_train_samples // total_batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = max(1, total_steps - warmup_steps)

    lr_schedule = CosineDecay(
        initial_learning_rate=1e-6,
        decay_steps=decay_steps,
        warmup_target=min(lr, 1e-4 * (total_batch_size / 2)),
        warmup_steps=warmup_steps,
        alpha=0.1,
    )

    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
    )

    # Loss functions
    dice_ce_loss = SparseDiceCELoss(
        from_logits=False,
        num_classes=num_classes,
        ignore_class_ids=2,
    )
    cldice_loss = SparseCenterlineDiceLoss(
        from_logits=False,
        num_classes=num_classes,
        target_class_ids=1,
        ignore_class_ids=2,
        iters=cldice_iters
    )

    def combined_loss(y_true, y_pred):
        return dice_ce_loss(y_true, y_pred) + cldice_weight * cldice_loss(y_true, y_pred)

    # Metrics
    dice_metric = SparseDiceMetric(
        from_logits=False,
        num_classes=num_classes,
        ignore_class_ids=2,
        name='dice'
    )

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[dice_metric],
    )

    # ==========================================================================
    # Callbacks
    # ==========================================================================
    # Sliding Window Inference callback for validation
    swi_callback_metric = SparseDiceMetric(
        from_logits=False,
        ignore_class_ids=2,
        num_classes=num_classes,
        name='val_dice',
    )

    swi_callback = SlidingWindowInferenceCallback(
        model,
        dataset=val_ds,
        metrics=swi_callback_metric,
        num_classes=num_classes,
        interval=val_interval,
        overlap=overlap,
        mode='gaussian',
        roi_size=input_shape,
        sw_batch_size=1,
        save_path=os.path.join(exp_dir, "model.weights.h5")
    )

    # Competition metric callback
    comp_metric_callback = CompetitionMetricCallback(
        model=model,
        val_ds=val_ds,
        input_shape=input_shape,
        num_classes=num_classes,
        eval_interval=eval_interval,
        surface_tolerance=surface_tolerance,
        overlap=overlap,
        voi_connectivity=voi_connectivity,
        voi_transform=voi_transform,
        voi_alpha=voi_alpha,
        combine_weights=combine_weights,
        ignore_label=ignore_label,
        output_dir=exp_dir,
        logger=logger
    )

    # Epoch logging callback
    epoch_logging_callback = EpochLoggingCallback(logger=logger)

    callbacks = [epoch_logging_callback, swi_callback, comp_metric_callback]

    # ==========================================================================
    # Train
    # ==========================================================================
    logger.info("\nStarting training...")

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # ==========================================================================
    # Final Evaluation with Best Model
    # ==========================================================================
    best_weights_path = os.path.join(exp_dir, "model.weights.h5")

    # Save model if not saved yet (e.g., epochs < val_interval)
    if not os.path.exists(best_weights_path):
        logger.info("No model saved during training. Saving current model...")
        model.save_weights(best_weights_path)

    # Run final competition metric evaluation
    logger.info("\nRunning final evaluation with best model...")
    model.load_weights(best_weights_path)
    final_results = comp_metric_callback.evaluate_competition_metrics(epoch='final')

    # ==========================================================================
    # Summary
    # ==========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Training Complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Best competition score: {comp_metric_callback.best_score:.4f}")
    logger.info(f"Model saved to: {best_weights_path}")
    logger.info(f"Log saved to: {os.path.join(exp_dir, f'{exp_name}.log')}")

    # Save comprehensive history
    import json
    history_path = os.path.join(exp_dir, 'history.json')

    # Combine all history data
    comprehensive_history = {
        # Per-epoch training metrics (loss, dice)
        'training': {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        },
        # Per-epoch detailed logs
        'epochs': epoch_logging_callback.epoch_history,
        # Competition metrics (eval_interval epochs + final)
        'competition_metrics': comp_metric_callback.results_history,
        # Best scores
        'best': {
            'competition_score': comp_metric_callback.best_score,
        },
        # Config summary
        'config': {
            'experiment_name': exp_name,
            'model': model_arch,
            'encoder': encoder,
            'input_size': input_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'debug_mode': debug_mode,
        }
    }

    with open(history_path, 'w') as f:
        json.dump(comprehensive_history, f, indent=2)
    logger.info(f"History saved to: {history_path}")


if __name__ == '__main__':
    main()

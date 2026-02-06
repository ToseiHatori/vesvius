#!/usr/bin/env python3
"""
Vesuvius Competition Metric Evaluation Script

Evaluates a trained model using the OFFICIAL competition metric (topometrics):
- Topological Score (Betti matching) - 30%
- Surface Dice - 35%
- VOI (Variation of Information) - 35%

Usage:
    python evaluate.py --weights model.weights.h5 --tfrecords /path/to/tfrecords

The validation set is the last TFRecord file (same as training notebook).
"""

import argparse
import glob
import os
import sys

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import tensorflow as tf
import keras
from keras import ops

from medicai.transforms import Compose, NormalizeIntensity
from medicai.models import TransUNet
from medicai.utils import SlidingWindowInference

# Official topometrics library
from topometrics import compute_leaderboard_score


# Model configuration (same as training v3)
INPUT_SHAPE = (160, 160, 160)
NUM_CLASSES = 3


def parse_tfrecord_fn(example):
    """Parse TFRecord format (same as training notebook)."""
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
    """Prepare inputs (same as training notebook)."""
    image = image[..., None]  # (D, H, W, 1)
    label = label[..., None]  # (D, H, W, 1)
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def val_transformation(image, label):
    """Validation transformation (same as training notebook)."""
    data = {"image": image, "label": label}
    pipeline = Compose([
        NormalizeIntensity(
            keys=["image"],
            nonzero=True,
            channel_wise=False
        ),
    ])
    result = pipeline(data)
    return result["image"], result["label"]


def load_validation_dataset(tfrecord_dir: str, batch_size: int = 1):
    """
    Load validation dataset from TFRecords.

    Uses the last TFRecord file as validation (same as training notebook).
    """
    all_tfrec = sorted(
        glob.glob(os.path.join(tfrecord_dir, "*.tfrec")),
        key=lambda x: int(x.split("_")[-1].replace(".tfrec", ""))
    )

    if len(all_tfrec) == 0:
        raise FileNotFoundError(f"No .tfrec files found in {tfrecord_dir}")

    # Use last TFRecord as validation (same as training)
    val_patterns = [all_tfrec[-1]]
    print(f"Using validation file: {val_patterns[0]}")

    dataset = tf.data.TFRecordDataset(val_patterns)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(prepare_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(val_transformation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_model(weights_path: str):
    """Load TransUNet model with weights."""
    print(f"Loading model from: {weights_path}")

    model = TransUNet(
        encoder_name='seresnext50',
        input_shape=INPUT_SHAPE + (1,),
        num_classes=NUM_CLASSES,
        classifier_activation='softmax'
    )

    model.load_weights(weights_path)
    print(f"Model loaded: {model.count_params() / 1e6:.2f}M parameters")

    return model


def evaluate(
    model,
    val_ds,
    surface_tolerance: float = 2.0,
    overlap: float = 0.5,
):
    """
    Evaluate model using OFFICIAL competition metrics (topometrics).

    Args:
        model: Loaded Keras model
        val_ds: Validation dataset
        surface_tolerance: Tolerance for surface dice (default: 2.0)
        overlap: Overlap for sliding window inference (default: 0.5)

    Returns:
        Dictionary with evaluation results
    """
    # Setup sliding window inference
    swi = SlidingWindowInference(
        model,
        num_classes=NUM_CLASSES,
        roi_size=INPUT_SHAPE,
        mode='gaussian',
        sw_batch_size=1,
        overlap=overlap,
    )

    all_scores = []
    all_topo = []
    all_surface_dice = []
    all_voi = []

    print("\n" + "=" * 60)
    print("Evaluating with OFFICIAL competition metrics (topometrics)")
    print("=" * 60)

    for i, (x, y) in enumerate(val_ds):
        print(f"\nSample {i + 1}:")
        print(f"  Input shape: {x.shape}")

        # Run inference
        y_pred = swi(x)

        # Convert to numpy and get segmentation
        pred_seg = y_pred.argmax(-1).astype(np.uint8).squeeze()
        gt_seg = y.numpy().astype(np.uint8).squeeze()

        print(f"  Pred unique values: {np.unique(pred_seg)}")
        print(f"  GT unique values: {np.unique(gt_seg)}")

        # Compute OFFICIAL competition score
        report = compute_leaderboard_score(
            predictions=pred_seg,
            labels=gt_seg,
            dims=(0, 1, 2),
            spacing=(1.0, 1.0, 1.0),
            surface_tolerance=surface_tolerance,
            voi_connectivity=26,
            voi_transform='one_over_one_plus',
            voi_alpha=0.3,
            combine_weights=(0.3, 0.35, 0.35),
            fg_threshold=None,
            ignore_label=2,
            ignore_mask=None,
        )

        all_scores.append(report.score)
        all_topo.append(report.topo.toposcore)
        all_surface_dice.append(report.surface_dice)
        all_voi.append(report.voi.voi_score)

        print(f"  Topo Score:     {report.topo.toposcore:.4f}")
        print(f"    TopoF1 dim0:  {report.topo.topoF1_by_dim.get(0, float('nan')):.4f}")
        print(f"    TopoF1 dim1:  {report.topo.topoF1_by_dim.get(1, float('nan')):.4f}")
        print(f"    TopoF1 dim2:  {report.topo.topoF1_by_dim.get(2, float('nan')):.4f}")
        print(f"  Surface Dice:   {report.surface_dice:.4f}")
        print(f"  VOI Score:      {report.voi.voi_score:.4f}")
        print(f"    VOI split:    {report.voi.voi_split:.4f}")
        print(f"    VOI merge:    {report.voi.voi_merge:.4f}")
        print(f"  Combined Score: {report.score:.4f}")

    # Aggregate results
    results = {
        'mean_score': float(np.mean(all_scores)),
        'mean_topo': float(np.mean(all_topo)),
        'mean_surface_dice': float(np.mean(all_surface_dice)),
        'mean_voi': float(np.mean(all_voi)),
        'std_score': float(np.std(all_scores)),
        'n_samples': len(all_scores),
        'per_sample_scores': all_scores,
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS (Official Competition Metric)")
    print("=" * 60)
    print(f"  Samples evaluated: {results['n_samples']}")
    print(f"  Mean Topo Score:     {results['mean_topo']:.4f}")
    print(f"  Mean Surface Dice:   {results['mean_surface_dice']:.4f}")
    print(f"  Mean VOI Score:      {results['mean_voi']:.4f}")
    print(f"  Mean Combined Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Vesuvius model with OFFICIAL competition metrics'
    )
    parser.add_argument(
        '--weights', '-w',
        type=str,
        required=True,
        help='Path to model weights (.h5 file)'
    )
    parser.add_argument(
        '--tfrecords', '-t',
        type=str,
        required=True,
        help='Path to directory containing TFRecord files'
    )
    parser.add_argument(
        '--surface-tolerance',
        type=float,
        default=2.0,
        help='Surface tolerance for Surface Dice (default: 2.0)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='Overlap for sliding window inference (default: 0.5)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file for results (JSON format)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)

    if not os.path.isdir(args.tfrecords):
        print(f"Error: TFRecords directory not found: {args.tfrecords}")
        sys.exit(1)

    # Load model
    model = load_model(args.weights)

    # Load validation dataset
    val_ds = load_validation_dataset(args.tfrecords)

    # Evaluate
    results = evaluate(
        model,
        val_ds,
        surface_tolerance=args.surface_tolerance,
        overlap=args.overlap,
    )

    # Save results if output specified
    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == '__main__':
    main()

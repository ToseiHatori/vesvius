#!/usr/bin/env python3
"""
Validation Error Analysis Script

予測確率値とGround Truthを並べてエラー解析を行うスクリプト。
画像はファイルに保存されます。

Usage:
    # コンテナ内で実行
    docker-compose exec nnunet python /workspace/visualize_validation.py --case 1004283650

    # 複数ケースをまとめて可視化
    docker-compose exec nnunet python /workspace/visualize_validation.py --case 1004283650 102536988

    # 全スライスを保存（gifアニメーション用）
    docker-compose exec nnunet python /workspace/visualize_validation.py --case 1004283650 --all-slices

    # 閾値を変更
    docker-compose exec nnunet python /workspace/visualize_validation.py --case 1004283650 --threshold 0.75

    # 利用可能なケース一覧
    docker-compose exec nnunet python /workspace/visualize_validation.py --list
"""

import argparse
import numpy as np
import tifffile
import blosc2
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI不要
import matplotlib.pyplot as plt


# ディレクトリ設定（Docker内パス）
PROJECT_ROOT = Path('/workspace')
NPZ_DIR = PROJECT_ROOT / 'nnunet_work' / 'val_output_npz'
GT_DIR = PROJECT_ROOT / 'nnunet_output' / 'nnUNet_preprocessed' / 'Dataset100_VesuviusSurface' / 'gt_segmentations'
IMAGE_DIR = PROJECT_ROOT / 'nnunet_output' / 'nnUNet_preprocessed' / 'Dataset100_VesuviusSurface' / 'nnUNetPlans_3d_fullres'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'validation_analysis'


def get_available_cases():
    """利用可能なケースIDを取得"""
    npz_files = sorted(NPZ_DIR.glob('*.npz'))
    return [f.stem for f in npz_files if (GT_DIR / f'{f.stem}.tif').exists()]


def load_case(case_id: str):
    """
    ケースを読み込む

    Returns:
        image: (320, 320, 320) original image (padded to match)
        probs: (320, 320, 320) surface class probability
        gt: (320, 320, 320) ground truth
    """
    npz_path = NPZ_DIR / f'{case_id}.npz'
    with np.load(npz_path) as data:
        probs = data['probabilities'][1]  # class 1 = surface

    gt_path = GT_DIR / f'{case_id}.tif'
    gt = tifffile.imread(gt_path)

    # 元画像を読み込み（サイズが異なる場合はパディング）
    image_path = IMAGE_DIR / f'{case_id}.b2nd'
    if image_path.exists():
        arr = blosc2.open(str(image_path))
        img = arr[0]  # (1, D, H, W) -> (D, H, W)

        # 320x320x320にパディング
        target_shape = probs.shape
        if img.shape != target_shape:
            padded = np.zeros(target_shape, dtype=img.dtype)
            d, h, w = img.shape
            td, th, tw = target_shape
            # 中央に配置
            d_off = (td - d) // 2
            h_off = (th - h) // 2
            w_off = (tw - w) // 2
            padded[d_off:d_off+d, h_off:h_off+h, w_off:w_off+w] = img
            image = padded
        else:
            image = img
    else:
        image = None

    return image, probs, gt


def compute_stats(probs, gt, threshold=0.5):
    """統計情報を計算"""
    valid_mask = gt != 2
    gt_valid = gt[valid_mask]
    probs_valid = probs[valid_mask]

    pred_binary = (probs >= threshold).astype(np.uint8)
    pred_valid = pred_binary[valid_mask]

    tp = ((pred_valid == 1) & (gt_valid == 1)).sum()
    fp = ((pred_valid == 1) & (gt_valid == 0)).sum()
    fn = ((pred_valid == 0) & (gt_valid == 1)).sum()
    tn = ((pred_valid == 0) & (gt_valid == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'dice': dice,
        'gt_positive_ratio': gt_valid.mean(),
        'pred_positive_ratio': pred_valid.mean(),
        'ignore_ratio': (gt == 2).mean()
    }


def visualize_slice(image, probs, gt, case_id, axis=0, slice_idx=160, threshold=0.5, output_path=None):
    """スライスを可視化して保存（5枚: 元画像, 確率, 予測, GT, エラーマップ）"""
    # スライス取得
    if axis == 0:
        img_slice = image[slice_idx, :, :] if image is not None else None
        prob_slice = probs[slice_idx, :, :]
        gt_slice = gt[slice_idx, :, :]
        axis_name = 'Z'
    elif axis == 1:
        img_slice = image[:, slice_idx, :] if image is not None else None
        prob_slice = probs[:, slice_idx, :]
        gt_slice = gt[:, slice_idx, :]
        axis_name = 'Y'
    else:
        img_slice = image[:, :, slice_idx] if image is not None else None
        prob_slice = probs[:, :, slice_idx]
        gt_slice = gt[:, :, slice_idx]
        axis_name = 'X'

    pred_slice = (prob_slice >= threshold).astype(np.uint8)

    # ignore領域のマスク
    ignore_mask = gt_slice == 2
    gt_display = np.where(ignore_mask, 0.5, gt_slice.astype(float))

    # エラーマップ作成
    error_rgb = np.zeros((*prob_slice.shape, 3))
    valid = ~ignore_mask

    fp_mask = valid & (pred_slice == 1) & (gt_slice == 0)
    fn_mask = valid & (pred_slice == 0) & (gt_slice == 1)
    tp_mask = valid & (pred_slice == 1) & (gt_slice == 1)
    tn_mask = valid & (pred_slice == 0) & (gt_slice == 0)

    error_rgb[fp_mask] = [1, 0.3, 0.3]   # 赤 (FP)
    error_rgb[fn_mask] = [0.3, 0.3, 1]   # 青 (FN)
    error_rgb[tp_mask] = [0.3, 1, 0.3]   # 緑 (TP)
    error_rgb[tn_mask] = [0.2, 0.2, 0.2] # 暗いグレー (TN)
    error_rgb[ignore_mask] = [0.5, 0.5, 0]  # 黄色 (ignore)

    # プロット（5枚）
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # 元画像
    if img_slice is not None:
        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title('Original')
    else:
        axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Original (N/A)')

    im1 = axes[1].imshow(prob_slice, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Probability')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].imshow(pred_slice, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Prediction (th={threshold:.2f})')

    axes[3].imshow(gt_display, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Ground Truth (gray=ignore)')

    axes[4].imshow(error_rgb)
    axes[4].set_title('Error Map (R:FP, B:FN, G:TP)')

    for ax in axes:
        ax.axis('off')

    # スライス統計
    slice_tp = tp_mask.sum()
    slice_fp = fp_mask.sum()
    slice_fn = fn_mask.sum()

    fig.suptitle(f'Case: {case_id} | {axis_name}={slice_idx} | TP={slice_tp} FP={slice_fp} FN={slice_fn}', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return slice_tp, slice_fp, slice_fn


def visualize_3axis(image, probs, gt, case_id, z=160, y=160, x=160, threshold=0.5, output_path=None):
    """3軸同時表示（5列: 元画像, 確率, 予測, GT, エラーマップ）"""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    slices = [
        (image[z, :, :] if image is not None else None, probs[z, :, :], gt[z, :, :], f'Z={z}'),
        (image[:, y, :] if image is not None else None, probs[:, y, :], gt[:, y, :], f'Y={y}'),
        (image[:, :, x] if image is not None else None, probs[:, :, x], gt[:, :, x], f'X={x}')
    ]

    for row, (img_slice, prob_slice, gt_slice, title) in enumerate(slices):
        pred_slice = (prob_slice >= threshold).astype(np.uint8)
        ignore_mask = gt_slice == 2
        gt_display = np.where(ignore_mask, 0.5, gt_slice.astype(float))

        error_rgb = np.zeros((*prob_slice.shape, 3))
        valid = ~ignore_mask
        error_rgb[valid & (pred_slice == 1) & (gt_slice == 0)] = [1, 0.3, 0.3]
        error_rgb[valid & (pred_slice == 0) & (gt_slice == 1)] = [0.3, 0.3, 1]
        error_rgb[valid & (pred_slice == 1) & (gt_slice == 1)] = [0.3, 1, 0.3]
        error_rgb[valid & (pred_slice == 0) & (gt_slice == 0)] = [0.2, 0.2, 0.2]
        error_rgb[ignore_mask] = [0.5, 0.5, 0]

        # 元画像
        if img_slice is not None:
            axes[row, 0].imshow(img_slice, cmap='gray')
        else:
            axes[row, 0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[row, 0].transAxes)
        axes[row, 0].set_ylabel(title, fontsize=12)

        axes[row, 1].imshow(prob_slice, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].imshow(pred_slice, cmap='gray', vmin=0, vmax=1)
        axes[row, 3].imshow(gt_display, cmap='gray', vmin=0, vmax=1)
        axes[row, 4].imshow(error_rgb)

    axes[0, 0].set_title('Original')
    axes[0, 1].set_title('Probability')
    axes[0, 2].set_title('Prediction')
    axes[0, 3].set_title('Ground Truth')
    axes[0, 4].set_title('Error Map')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Case: {case_id} | Threshold: {threshold}', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def probability_histogram(probs, gt, case_id, output_path=None):
    """確率分布をGTクラス別に可視化"""
    valid_mask = gt != 2
    bg_probs = probs[valid_mask & (gt == 0)]
    fg_probs = probs[valid_mask & (gt == 1)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(bg_probs.ravel(), bins=50, alpha=0.7, label='Background (GT=0)', density=True)
    axes[0].hist(fg_probs.ravel(), bins=50, alpha=0.7, label='Surface (GT=1)', density=True)
    axes[0].set_xlabel('Probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Probability Distribution by GT Class')
    axes[0].legend()
    axes[0].axvline(0.5, color='r', linestyle='--', alpha=0.5)

    axes[1].axis('off')
    stats_text = f"""Background (GT=0):
  Count: {len(bg_probs):,}
  Mean:  {bg_probs.mean():.4f}
  Std:   {bg_probs.std():.4f}
  >0.5:  {(bg_probs > 0.5).mean()*100:.2f}%

Surface (GT=1):
  Count: {len(fg_probs):,}
  Mean:  {fg_probs.mean():.4f}
  Std:   {fg_probs.std():.4f}
  <0.5:  {(fg_probs < 0.5).mean()*100:.2f}%"""
    axes[1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                 verticalalignment='center', transform=axes[1].transAxes)
    axes[1].set_title('Statistics')

    fig.suptitle(f'Case: {case_id}', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def threshold_analysis(probs, gt, case_id, output_path=None):
    """閾値ごとのメトリクス変化"""
    thresholds = np.arange(0.1, 1.0, 0.05)

    valid_mask = gt != 2
    gt_valid = gt[valid_mask]
    probs_valid = probs[valid_mask]

    results = []
    for th in thresholds:
        pred = (probs_valid >= th).astype(np.uint8)

        tp = ((pred == 1) & (gt_valid == 1)).sum()
        fp = ((pred == 1) & (gt_valid == 0)).sum()
        fn = ((pred == 0) & (gt_valid == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        results.append({'threshold': th, 'precision': precision, 'recall': recall, 'dice': dice})

    fig, ax = plt.subplots(figsize=(10, 5))
    ths = [r['threshold'] for r in results]
    ax.plot(ths, [r['precision'] for r in results], 'b-o', label='Precision', markersize=4)
    ax.plot(ths, [r['recall'] for r in results], 'r-o', label='Recall', markersize=4)
    ax.plot(ths, [r['dice'] for r in results], 'g-o', label='Dice', markersize=4)

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(f'Threshold Analysis - Case: {case_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def find_error_hotspots(probs, gt, threshold=0.5, top_k=5):
    """エラーが集中しているスライスを検出"""
    pred = (probs >= threshold).astype(np.uint8)
    valid = gt != 2

    hotspots = {}
    for axis, axis_name in [(0, 'Z'), (1, 'Y'), (2, 'X')]:
        fp_counts = []
        fn_counts = []

        for i in range(320):
            if axis == 0:
                s_pred, s_gt, s_valid = pred[i], gt[i], valid[i]
            elif axis == 1:
                s_pred, s_gt, s_valid = pred[:, i], gt[:, i], valid[:, i]
            else:
                s_pred, s_gt, s_valid = pred[:, :, i], gt[:, :, i], valid[:, :, i]

            fp = (s_valid & (s_pred == 1) & (s_gt == 0)).sum()
            fn = (s_valid & (s_pred == 0) & (s_gt == 1)).sum()
            fp_counts.append(fp)
            fn_counts.append(fn)

        fp_top = np.argsort(fp_counts)[-top_k:][::-1]
        fn_top = np.argsort(fn_counts)[-top_k:][::-1]

        hotspots[axis_name] = {
            'fp_top': [(int(i), int(fp_counts[i])) for i in fp_top],
            'fn_top': [(int(i), int(fn_counts[i])) for i in fn_top]
        }

    return hotspots


def process_case(case_id, threshold=0.5, all_slices=False, output_dir=None):
    """1ケースを処理"""
    print(f"\n{'='*60}")
    print(f"Processing case: {case_id}")
    print(f"{'='*60}")

    # 出力ディレクトリ
    if output_dir is None:
        output_dir = OUTPUT_DIR / case_id
    else:
        output_dir = Path(output_dir) / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("Loading data...")
    image, probs, gt = load_case(case_id)

    # 統計表示
    stats = compute_stats(probs, gt, threshold)
    print(f"\nStatistics (threshold={threshold}):")
    print(f"  TP: {stats['tp']:,}  FP: {stats['fp']:,}")
    print(f"  FN: {stats['fn']:,}  TN: {stats['tn']:,}")
    print(f"  Precision: {stats['precision']:.4f}")
    print(f"  Recall:    {stats['recall']:.4f}")
    print(f"  Dice:      {stats['dice']:.4f}")

    # エラーホットスポット
    print("\nError hotspots:")
    hotspots = find_error_hotspots(probs, gt, threshold)
    for axis_name, data in hotspots.items():
        print(f"  {axis_name}-axis FP top: {data['fp_top'][:3]}")
        print(f"  {axis_name}-axis FN top: {data['fn_top'][:3]}")

    # 可視化
    print("\nGenerating visualizations...")

    # 3軸表示
    visualize_3axis(image, probs, gt, case_id, threshold=threshold,
                    output_path=output_dir / 'overview_3axis.png')
    print(f"  Saved: overview_3axis.png")

    # 確率分布
    probability_histogram(probs, gt, case_id,
                          output_path=output_dir / 'probability_histogram.png')
    print(f"  Saved: probability_histogram.png")

    # 閾値分析
    threshold_analysis(probs, gt, case_id,
                       output_path=output_dir / 'threshold_analysis.png')
    print(f"  Saved: threshold_analysis.png")

    # 代表スライス（中央 + エラーホットスポット）
    slices_to_save = [
        (0, 160, 'z160_center'),
        (1, 160, 'y160_center'),
        (2, 160, 'x160_center'),
    ]
    # ホットスポットも追加
    for axis, axis_name in [(0, 'Z'), (1, 'Y'), (2, 'X')]:
        top_fp_idx = hotspots[axis_name]['fp_top'][0][0]
        top_fn_idx = hotspots[axis_name]['fn_top'][0][0]
        slices_to_save.append((axis, top_fp_idx, f'{axis_name.lower()}{top_fp_idx}_top_fp'))
        slices_to_save.append((axis, top_fn_idx, f'{axis_name.lower()}{top_fn_idx}_top_fn'))

    for axis, idx, name in slices_to_save:
        visualize_slice(image, probs, gt, case_id, axis=axis, slice_idx=idx, threshold=threshold,
                        output_path=output_dir / f'slice_{name}.png')
    print(f"  Saved: {len(slices_to_save)} slice images")

    # 全スライス（オプション）
    if all_slices:
        all_slices_dir = output_dir / 'all_slices'
        all_slices_dir.mkdir(exist_ok=True)
        print(f"  Saving all Z-slices to {all_slices_dir}...")
        for z in range(0, 320, 4):  # 4スライスごと
            visualize_slice(image, probs, gt, case_id, axis=0, slice_idx=z, threshold=threshold,
                            output_path=all_slices_dir / f'z_{z:03d}.png')
        print(f"  Saved: {320//4} slice images")

    print(f"\nOutput directory: {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Validation Error Analysis')
    parser.add_argument('--case', nargs='+', help='Case ID(s) to process')
    parser.add_argument('--list', action='store_true', help='List available cases')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold (default: 0.5)')
    parser.add_argument('--all-slices', action='store_true', help='Save all Z-slices')
    parser.add_argument('--output-dir', type=str, help='Output directory')

    args = parser.parse_args()

    if args.list:
        cases = get_available_cases()
        print(f"Available cases: {len(cases)}")
        for i, c in enumerate(cases):
            print(f"  {c}", end='')
            if (i + 1) % 5 == 0:
                print()
        print()
        return

    if not args.case:
        # デフォルトで最初のケースを処理
        cases = get_available_cases()
        if cases:
            args.case = [cases[0]]
            print(f"No case specified, using first available: {args.case[0]}")
        else:
            print("No cases available!")
            return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for case_id in args.case:
        process_case(case_id, threshold=args.threshold, all_slices=args.all_slices,
                     output_dir=args.output_dir)


if __name__ == '__main__':
    main()

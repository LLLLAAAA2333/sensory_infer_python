import argparse
import os
import re
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.comm_utils.prints import print_info_message
import src.plot_result.vis_trajectory as vis


def natural_sort_key(text):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"([0-9]+)", text)]


def load_volume_paths(volume_dir):
    if not os.path.isdir(volume_dir):
        raise FileNotFoundError(f"Volume directory not found: {volume_dir}")

    direct_paths = [
        os.path.join(volume_dir, file_name)
        for file_name in os.listdir(volume_dir)
        if file_name.endswith(".npy")
    ]
    if direct_paths:
        direct_paths.sort(key=natural_sort_key)
        return direct_paths

    paths = []
    subdirs = [
        os.path.join(volume_dir, file_name)
        for file_name in os.listdir(volume_dir)
        if os.path.isdir(os.path.join(volume_dir, file_name))
    ]
    subdirs.sort(key=natural_sort_key)
    for subdir in subdirs:
        sub_paths = [
            os.path.join(subdir, file_name)
            for file_name in os.listdir(subdir)
            if file_name.endswith(".npy")
        ]
        sub_paths.sort(key=natural_sort_key)
        paths.extend(sub_paths)

    if not paths:
        raise FileNotFoundError(f"No .npy volumes found in: {volume_dir}")
    paths.sort(key=natural_sort_key)
    return paths


def apply_zrange(volume_np, zrange=None):
    if zrange is None:
        return volume_np
    z_start, z_end = zrange
    z_start = max(int(z_start), 0)
    if z_end == -1 or z_end > volume_np.shape[2]:
        z_end = volume_np.shape[2]
    if z_start == 0 and z_end == volume_np.shape[2]:
        return volume_np
    return volume_np[:, :, z_start:z_end]


def parse_zrange(zrange_text):
    if not zrange_text:
        return None
    parts = [int(part.strip()) for part in zrange_text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected zrange as 'start,end', got: {zrange_text}")
    return tuple(parts)


def expand_span_with_min_size(start, end, limit, min_size):
    start = int(max(0, start))
    end = int(min(limit, end))
    if end <= start:
        return 0, int(limit)

    span = end - start
    if span >= min_size:
        return start, end

    center = (start + end) // 2
    half = int(np.ceil(min_size / 2.0))
    start = center - half
    end = start + min_size
    if start < 0:
        start = 0
        end = min(limit, min_size)
    if end > limit:
        end = limit
        start = max(0, end - min_size)
    return int(start), int(end)


def compute_global_roi_bounds_xyz(neuron_pt_tuple, volume_shape_zyx, padding_xyz=(64, 64, 6), min_size_xyz=(256, 256, 16)):
    pts = np.asarray(neuron_pt_tuple)
    if pts.ndim != 3 or pts.shape[-1] < 3:
        z_lim, y_lim, x_lim = volume_shape_zyx
        return {"x0": 0, "x1": int(x_lim), "y0": 0, "y1": int(y_lim), "z0": 0, "z1": int(z_lim)}

    valid = ~np.isnan(pts[..., :3]).any(axis=-1)
    if not np.any(valid):
        z_lim, y_lim, x_lim = volume_shape_zyx
        return {"x0": 0, "x1": int(x_lim), "y0": 0, "y1": int(y_lim), "z0": 0, "z1": int(z_lim)}

    xyz = pts[..., :3][valid]
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    if pts.shape[-1] >= 6:
        whd = np.maximum(pts[..., 3:6][valid], 0.0)
        hx = 0.5 * whd[:, 0]
        hy = 0.5 * whd[:, 1]
        hz = 0.5 * whd[:, 2]
    else:
        hx = np.zeros_like(x)
        hy = np.zeros_like(y)
        hz = np.zeros_like(z)

    pad_x, pad_y, pad_z = padding_xyz
    z_lim, y_lim, x_lim = volume_shape_zyx

    x0 = int(np.floor(np.min(x - hx) - pad_x))
    x1 = int(np.ceil(np.max(x + hx) + pad_x))
    y0 = int(np.floor(np.min(y - hy) - pad_y))
    y1 = int(np.ceil(np.max(y + hy) + pad_y))
    z0 = int(np.floor(np.min(z - hz) - pad_z))
    z1 = int(np.ceil(np.max(z + hz) + pad_z))

    x0, x1 = expand_span_with_min_size(x0, x1, x_lim, int(min_size_xyz[0]))
    y0, y1 = expand_span_with_min_size(y0, y1, y_lim, int(min_size_xyz[1]))
    z0, z1 = expand_span_with_min_size(z0, z1, z_lim, int(min_size_xyz[2]))

    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": z0, "z1": z1}


def build_roi_video(
    volume_paths,
    neuron_pt_tuple,
    output_dir,
    fps=1.0,
    z_ratio=5.0,
    source_min=102,
    source_max=200,
    default_bbox_size=(6.0, 6.0, 6.0),
    red_pseudo_color=False,
    zrange=None,
    padding_xyz=(64, 64, 6),
    min_size_xyz=(256, 256, 16),
):
    if source_max <= source_min:
        raise ValueError(f"source_max must be greater than source_min, got {source_min} and {source_max}")

    neuron_pt_tuple = np.asarray(neuron_pt_tuple)
    if neuron_pt_tuple.ndim != 3:
        raise ValueError(f"Expected neuron_pt_tuple as (T, N, F), got shape {neuron_pt_tuple.shape}")

    total_frames = min(len(volume_paths), neuron_pt_tuple.shape[0])
    if total_frames <= 0:
        raise ValueError("No frames available for rendering.")

    volume_paths = list(volume_paths[:total_frames])
    neuron_pt_tuple = neuron_pt_tuple[:total_frames]

    first_volume = apply_zrange(np.load(volume_paths[0]), zrange)
    if first_volume.ndim != 3:
        raise ValueError(f"Expected volume shape (Y, X, Z), got {first_volume.shape}")

    volume_shape_zyx = (first_volume.shape[2], first_volume.shape[0], first_volume.shape[1])
    roi = compute_global_roi_bounds_xyz(
        neuron_pt_tuple,
        volume_shape_zyx=volume_shape_zyx,
        padding_xyz=padding_xyz,
        min_size_xyz=min_size_xyz,
    )
    print_info_message(f"Using global ROI: {roi}")

    def load_frame(idx):
        return apply_zrange(np.load(volume_paths[idx]), zrange)

    def mip_fetcher(idx):
        raw_volume = load_frame(idx)
        roi_volume = raw_volume[roi["y0"]:roi["y1"], roi["x0"]:roi["x1"], roi["z0"]:roi["z1"]]
        volume_zyx = np.transpose(roi_volume, (2, 0, 1))
        volume_u8 = np.clip(volume_zyx, source_min, source_max)
        volume_u8 = ((volume_u8 - source_min) / (source_max - source_min) * 255).astype(np.uint8)
        return vis.get_mip_from_uint8_gray_volume(
            volume_u8,
            x_ratio=1.0,
            y_ratio=1.0,
            z_ratio=z_ratio,
            source_min=source_min,
            source_max=source_max,
            red_pseudo_color=red_pseudo_color,
        )

    def neuron_fetcher(idx):
        frame_pts = np.asarray(neuron_pt_tuple[idx])
        if frame_pts.ndim != 2 or frame_pts.shape[1] < 3:
            raise ValueError(f"Expected neuron frame shape (N, F>=3), got {frame_pts.shape}")

        valid_mask = ~np.isnan(frame_pts[:, :3]).any(axis=1)
        frame_pts = frame_pts[valid_mask]
        neuron_ids = np.where(valid_mask)[0].astype(np.int32)

        if frame_pts.size == 0:
            return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int32)

        bbox = np.zeros((frame_pts.shape[0], 6), dtype=np.float32)
        bbox[:, :3] = frame_pts[:, :3]
        if frame_pts.shape[1] >= 6:
            bbox[:, 3:6] = frame_pts[:, 3:6]
        else:
            bbox[:, 3] = default_bbox_size[0]
            bbox[:, 4] = default_bbox_size[1]
            bbox[:, 5] = default_bbox_size[2]

        bbox[:, 0] -= roi["x0"]
        bbox[:, 1] -= roi["y0"]
        bbox[:, 2] -= roi["z0"]
        return bbox, neuron_ids

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "roi_bounds_xyz.npy"), np.array([
        roi["x0"], roi["x1"], roi["y0"], roi["y1"], roi["z0"], roi["z1"]
    ], dtype=np.int32))

    all_ids = np.arange(neuron_pt_tuple.shape[1], dtype=np.int32)
    viewer = vis.Plot3DResult(
        mip_fetcher,
        neuron_fetcher,
        split_number=1,
        bbox_thickness=1,
        trace_length=0,
        all_ids=all_ids,
    )
    subset_path = os.path.join(output_dir, "neuron_trace", "subset_00.avi")
    print_info_message("Saving single ROI tri-view video in one pass...")
    viewer.save_views(subset_path, total_frames, 0, fps)
    print_info_message(f"Video saved to {subset_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a tri-view neuron video from a global ROI computed from neuron coordinates.")
    parser.add_argument("--volume-dir", required=True, help="Directory containing volume .npy files.")
    parser.add_argument("--neuron-path", required=True, help="Path to ex_neuron_pt_tuple.npy.")
    parser.add_argument("--output-dir", required=True, help="Directory for output video files.")
    parser.add_argument("--fps", type=float, default=1.0, help="Output video FPS.")
    parser.add_argument("--z-ratio", type=float, default=5.0, help="Display scaling ratio for Z.")
    parser.add_argument("--source-min", type=float, default=102.0, help="Display lower bound.")
    parser.add_argument("--source-max", type=float, default=200.0, help="Display upper bound.")
    parser.add_argument("--padding-xy", type=int, default=64, help="ROI padding applied on X and Y.")
    parser.add_argument("--padding-z", type=int, default=6, help="ROI padding applied on Z.")
    parser.add_argument("--min-size-xy", type=int, default=256, help="Minimum ROI size on X and Y.")
    parser.add_argument("--min-size-z", type=int, default=16, help="Minimum ROI size on Z.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick tests.")
    parser.add_argument("--zrange", type=str, default=None, help="Optional zrange as 'start,end'.")
    parser.add_argument("--red-pseudo-color", action="store_true", help="Use red pseudo-color instead of gray.")
    parser.add_argument("--default-bbox-size", type=float, nargs=3, default=(6.0, 6.0, 6.0), metavar=("W", "H", "D"),
                        help="Fallback neuron box size when width/height/depth are absent.")
    args = parser.parse_args()

    volume_paths = load_volume_paths(args.volume_dir)
    neuron_pt_tuple = np.load(args.neuron_path)
    if args.max_frames is not None:
        frame_count = min(args.max_frames, len(volume_paths), neuron_pt_tuple.shape[0])
        volume_paths = volume_paths[:frame_count]
        neuron_pt_tuple = neuron_pt_tuple[:frame_count]

    build_roi_video(
        volume_paths=volume_paths,
        neuron_pt_tuple=neuron_pt_tuple,
        output_dir=args.output_dir,
        fps=args.fps,
        z_ratio=args.z_ratio,
        source_min=args.source_min,
        source_max=args.source_max,
        default_bbox_size=tuple(args.default_bbox_size),
        red_pseudo_color=args.red_pseudo_color,
        zrange=parse_zrange(args.zrange),
        padding_xyz=(args.padding_xy, args.padding_xy, args.padding_z),
        min_size_xyz=(args.min_size_xy, args.min_size_xy, args.min_size_z),
    )


if __name__ == "__main__":
    main()

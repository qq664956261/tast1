#!/usr/bin/env python3
"""
Convert a ROS1 .bag with stereo images into a 4Seasons-like layout:

<out_dir>/
  times.txt                                 # timestamp_ns timestamp_s exposure_s
  undistorted_images/
    cma0/<timestamp_ns>.png                 # left images
    cma1/<timestamp_ns>.png                 # right images

Notes:
- Pairs left/right frames by nearest timestamps within a tolerance.
- The first column (timestamp_ns) is used as the filename stem for both images.
- The third column (exposure_s) is constant unless you provide an input CSV/constant.

Requirements:
  pip install opencv-python
  ROS1 Python: rosbag, rospy, cv_bridge (usually available inside a ROS1 environment)

Example:
  python rosbag_to_4seasons.py \
    --bag 1027images_2025-10-27-09-50-44.bag \
    --left /imsee/image/rectified/left \
    --right /imsee/image/rectified/right \
    --out 4seasons_export \
    --exposure-const 0.4783219993 \
    --tolerance-ms 8 \
    --image-format png
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Lazy imports for ROS & OpenCV so we can show a helpful error if unavailable
try:
    import rosbag
except Exception as e:
    rosbag = None
try:
    import rospy
except Exception:
    rospy = None
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None
try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None

@dataclass
class Frame:
    stamp_ns: int            # from msg.header.stamp in nanoseconds
    stamp_s: float           # seconds (float)
    msg: object              # sensor_msgs/Image


def time_to_ns(secs: int, nsecs: int) -> int:
    return int(secs) * 1_000_000_000 + int(nsecs)


def load_topic_frames(bag: "rosbag.Bag", topic: str) -> List[Frame]:
    frames: List[Frame] = []
    for _, msg, _ in bag.read_messages(topics=[topic]):
        # Prefer header stamp
        if hasattr(msg, 'header') and msg.header and hasattr(msg.header, 'stamp'):
            secs = int(msg.header.stamp.secs)
            nsecs = int(msg.header.stamp.nsecs)
        else:
            # Fallback to bag time if header is missing
            # Note: read_messages gives (topic, msg, t). We used '_' above, so reopen.
            # We'll re-read to capture t without double parsing bag.
            pass
        stamp_ns = time_to_ns(secs, nsecs)
        stamp_s = secs + nsecs * 1e-9
        frames.append(Frame(stamp_ns=stamp_ns, stamp_s=stamp_s, msg=msg))
    frames.sort(key=lambda f: f.stamp_ns)
    return frames


def pair_frames(left: List[Frame], right: List[Frame], tol_ns: int) -> List[Tuple[Frame, Frame]]:
    pairs: List[Tuple[Frame, Frame]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        dl = left[i].stamp_ns
        dr = right[j].stamp_ns
        diff = dr - dl
        if abs(diff) <= tol_ns:
            pairs.append((left[i], right[j]))
            i += 1
            j += 1
        elif diff < 0:
            j += 1
        else:
            i += 1
    return pairs


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def msg_to_cv2(bridge: CvBridge, msg, prefer_bgr: bool = True):
    """Convert sensor_msgs/Image to a cv2 ndarray without losing bit-depth if possible."""
    # Try to keep original bit depth first
    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # If user wants standard 8-bit BGR output for color, we can convert later if needed
        return img
    except Exception:
        pass
    if prefer_bgr:
        # Fall back to bgr8 conversion
        return bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    # Last resort
    return bridge.imgmsg_to_cv2(msg)


def save_image(path: str, img) -> None:
    # If image is floating-point, normalize to 0..1 -> 0..65535 or 0..255 depending on dynamic range
    if img.dtype == 'float32' or img.dtype == 'float64':
        arr = img
        finite = np.isfinite(arr)
        if not np.any(finite):
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            if mx - mn < 1e-12:
                arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                arr = (arr - mn) / (mx - mn)
                # Choose 8-bit for general compatibility
                arr = (arr * 255.0).clip(0, 255).astype('uint8')
        cv2.imwrite(path, arr)
    else:
        cv2.imwrite(path, img)


def main():
    parser = argparse.ArgumentParser(description='Convert ROS1 bag stereo images to 4Seasons-like layout.')
    parser.add_argument('--bag', required=True, help='Path to .bag file')
    parser.add_argument('--left', required=True, help='Left image topic (e.g., /imsee/image/rectified/left)')
    parser.add_argument('--right', required=True, help='Right image topic (e.g., /imsee/image/rectified/right)')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--image-format', default='png', choices=['png', 'jpg', 'jpeg'], help='Image format to write')
    parser.add_argument('--tolerance-ms', type=float, default=10.0, help='Max time difference (ms) to consider frames a stereo pair')
    parser.add_argument('--exposure-const', type=float, default=0.0, help='Constant exposure time (seconds) for the 3rd column in times.txt')
    args = parser.parse_args()

    if rosbag is None:
        print('ERROR: rosbag Python module not found. Run inside a ROS1 environment.', file=sys.stderr)
        sys.exit(1)
    if cv2 is None or np is None:
        print('ERROR: OpenCV (cv2) and numpy are required. pip install opencv-python numpy', file=sys.stderr)
        sys.exit(1)
    if CvBridge is None:
        print('ERROR: cv_bridge is required (usually available in ROS1).', file=sys.stderr)
        sys.exit(1)

    tol_ns = int(args.tolerance_ms * 1_000_000)  # ms -> ns

    out_dir = os.path.abspath(args.out)
    cma0_dir = os.path.join(out_dir, 'undistorted_images', 'cma0')
    cma1_dir = os.path.join(out_dir, 'undistorted_images', 'cma1')
    ensure_dir(cma0_dir)
    ensure_dir(cma1_dir)

    print(f'Reading bag: {args.bag}')
    with rosbag.Bag(args.bag, 'r') as bag:
        left_frames = load_topic_frames(bag, args.left)
        right_frames = load_topic_frames(bag, args.right)

    print(f'Loaded left frames:  {len(left_frames)}')
    print(f'Loaded right frames: {len(right_frames)}')

    pairs = pair_frames(left_frames, right_frames, tol_ns)
    print(f'Paired frames (within {args.tolerance_ms} ms): {len(pairs)}')

    bridge = CvBridge()
    times_path = os.path.join(out_dir, 'times.txt')

    # Write times as we export images
    written = 0
    with open(times_path, 'w') as f:
        for left, right in pairs:
            # Use left timestamp as canonical time for the pair
            t_ns = left.stamp_ns
            t_s = left.stamp_s
            base = str(t_ns)
            left_path = os.path.join(cma0_dir, f'{base}.{args.image_format}')
            right_path = os.path.join(cma1_dir, f'{base}.{args.image_format}')

            # Convert images
            try:
                imgL = msg_to_cv2(bridge, left.msg, prefer_bgr=True)
                imgR = msg_to_cv2(bridge, right.msg, prefer_bgr=True)
            except Exception as e:
                print(f'WARN: cv_bridge conversion failed at {t_ns}: {e}', file=sys.stderr)
                continue

            # Save
            okL = save_image(left_path, imgL)
            okR = save_image(right_path, imgR)

            # OpenCV imwrite returns True/False; our save_image currently returns None, so check existence
            if not (os.path.exists(left_path) and os.path.exists(right_path)):
                print(f'WARN: Failed to write images for {t_ns}', file=sys.stderr)
                continue

            # times.txt line: <timestamp_ns> <timestamp_s> <exposure_s>
            f.write(f"{t_ns} {t_s:.10f} {args.exposure_const:.10f}\n")
            written += 1

    print(f'Exported {written} stereo pairs to:')
    print(f'  {cma0_dir}')
    print(f'  {cma1_dir}')
    print(f'Wrote times: {times_path}')


if __name__ == '__main__':
    main()


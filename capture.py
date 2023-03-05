import argparse
import os

import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs


def start_depth_camera(output_dir):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    while True:
        frames = pipeline.wait_for_frames()
        timestamp = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow("Secondary View", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Secondary View", images)

        cv2.imwrite(f"{output_dir}/frame-{timestamp}.color.png", color_image)
        cv2.imwrite(f"{output_dir}/frame-{timestamp}.depth.png", depth_image)

        # To get key presses
        key = cv2.waitKey(1)

        if key in (27, ord("q")) or cv2.getWindowProperty("Secondary View", cv2.WND_PROP_AUTOSIZE) < 0:
            break

    pipeline.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Intel Lidar L515")
    parser.add_argument("--experiment", default=0, type=int)
    parser.add_argument("--sequence", default=0, type=int)
    args = parser.parse_args()

    output_path = f"data/raw_data/exp_{args.experiment}/{args.sequence:02d}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    start_depth_camera(output_path)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
import cv2  # Added to save images

import argparse  # argparse imported later, but we need it here for annotation


# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    #image_dir = os.path.join(args.scene_dir, "images")
    #image_path_list = sorted([
    #  p for p in glob.glob(os.path.join(image_dir, "*.jpg"))
    #  if os.path.isfile(p)
    #])
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    # === Start: Save VGGT preprocessed network input images ===
    pre_dir = os.path.join(args.scene_dir, "preprocessed")
    os.makedirs(pre_dir, exist_ok=True)
    # 1) Resize images to network input resolution (518×518)
    imgs_resized = F.interpolate(
        images,
        size=(vggt_fixed_resolution, vggt_fixed_resolution),
        mode="bilinear",
        align_corners=False,
    )
    # 2) Directly save the [0,1] normalized tensors as 0-255 PNGs
    for idx_img in range(imgs_resized.shape[0]):
        img_t = imgs_resized[idx_img]                 # (3, H, W), float in [0,1]
        img_np = (img_t * 255).clamp(0, 255).byte()    # (3, H, W), uint8
        img_np = img_np.permute(1, 2, 0).cpu().numpy() # H×W×3, RGB
        img_bgr = img_np[..., ::-1]                   # RGB → BGR for OpenCV
        cv2.imwrite(os.path.join(pre_dir, f"img_{idx_img:04d}.png"), img_bgr)
    print(f"Saved preprocessed images to {pre_dir}")
# === End: Save VGGT preprocessed network input images ===
    print(f"Loaded {len(images)} images from {image_dir}")


    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    
    # <<<< ADDED: Save depth, intrinsics, and pose for SAM3D START >>>>
    ## 1) original images size
    #image_dir = os.path.join(args.scene_dir, "images")
    #save_list = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".jpg"))
    # 
    #sample = cv2.imread(os.path.join(image_dir, save_list[0]), cv2.IMREAD_UNCHANGED)
    #orig_H, orig_W = sample.shape[:2]
    #print(f"Original image size detected: {orig_W}×{orig_H}")

    ## 2) prepare folders
    #out_base   = args.scene_dir      # point to …/images
    #depth_dir  = os.path.join(out_base, "depth")
    #intri_dir  = os.path.join(out_base, "intrinsics")
    #pose_dir   = os.path.join(out_base, "pose")
    #os.makedirs(depth_dir, exist_ok=True)
    #os.makedirs(intri_dir, exist_ok=True)
    #os.makedirs(pose_dir,  exist_ok=True)

    # 3) 
    #depth_resized = []
    #for d in depth_map:
    #    dr = cv2.resize(d, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
    #    depth_resized.append(dr)
    #depth_map = np.stack(depth_resized)

    ## 缩放内参：按照 sx=orig_W/518, sy=orig_H/518
    #sx, sy = orig_W / depth_resized[0].shape[1], orig_H / depth_resized[0].shape[0]
    #intrinsic[:, 0, 0] *= sx
    #intrinsic[:, 1, 1] *= sy
    #intrinsic[:, 0, 2] *= sx
    #intrinsic[:, 1, 2] *= sy
    #print(f"Resized depth & adjusted intrinsic with sx={sx:.3f}, sy={sy:.3f}")

    ## 4) save
    #for i, fname in enumerate(save_list):
    #    base = os.path.splitext(fname)[0]
    #    
    #    d_mm = (depth_map[i] * 1000.0).astype(np.uint16)
    #    cv2.imwrite(os.path.join(depth_dir, f"{base}.png"), d_mm)
    #    
    #    np.savetxt(os.path.join(intri_dir, f"{base}.txt"), intrinsic[i], fmt="%.6f")
    #    
    #    P = extrinsic[i]
    #    P3 = P[:3, :] if P.shape == (4,4) else P
    #    np.savetxt(os.path.join(pose_dir, f"{base}.txt"), P3, fmt="%.6f")

    #print(f"Saved depth → {depth_dir}")
    #print(f"Saved intrinsics → {intri_dir}")
    #print(f"Saved pose       → {pose_dir}")
    ## imgs_resized 是在脚本前面已生成的 (N,3,518,518) Tensor
    #color_dir = os.path.join(out_base, "color")
    #os.makedirs(color_dir, exist_ok=True)
    #for idx, fname in enumerate(save_list):
    #    # 从 imgs_resized 里取第 idx 张
    #    img_t = imgs_resized[idx]                     # Tensor (3,518,518) in [0,1]
    #    img_np = (img_t * 255).clamp(0,255).byte()    # to uint8
    #    img_np = img_np.permute(1,2,0).cpu().numpy()  # H×W×3 RGB
    #    bgr   = img_np[..., ::-1]                     # to BGR
    #    cv2.imwrite(os.path.join(color_dir, fname), bgr)
    #print(f"Saved 518×518 color → {color_dir}")


    out_base  = args.scene_dir
    depth_dir = os.path.join(out_base, "depth")
    intri_dir = os.path.join(out_base, "intrinsics")
    pose_dir  = os.path.join(out_base, "pose")
    color_dir = os.path.join(out_base, "color")
    for d in (depth_dir, intri_dir, pose_dir, color_dir):
        os.makedirs(d, exist_ok=True)

    # save depth_map (518×518) → depth/*.png
    for idx, name in enumerate(base_image_path_list):
        base = os.path.splitext(name)[0]
        d_mm = (depth_map[idx] * 1000.0).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"{base}.png"), d_mm)
        # intrinsics (same for all): intrinsic[idx] shape (3,3)
        np.savetxt(os.path.join(intri_dir, f"{base}.txt"), intrinsic[idx], fmt="%.6f")
        # pose
        P  = extrinsic[idx]
        P3 = P[:3]   # take first 3 rows
        np.savetxt(os.path.join(pose_dir, f"{base}.txt"), P3, fmt="%.6f")

    print(f"Saved 518×518 depth → {depth_dir}")
    print(f"Saved intrinsics → {intri_dir}")
    print(f"Saved pose       → {pose_dir}")

    # save imgs_resized (518×518) → color/*.png
    for idx, name in enumerate(base_image_path_list):
        img_t  = imgs_resized[idx]
        img_np = (img_t * 255).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
        cv2.imwrite(os.path.join(color_dir, name), img_np[...,::-1])
    print(f"Saved 518×518 color → {color_dir}")


    #depth_intrinsic = intrinsic[0]
    # 1) intrinsic_depth.txt
    #depth_intrinsic_path = os.path.join(intri_dir, "intrinsic_depth.txt")
    #np.savetxt(depth_intrinsic_path, depth_intrinsic, fmt="%.6f")
    #print(f"Saved depth intrinsics → {depth_intrinsic_path}")

    # 2) intrinsic_color.txt
    #color_intrinsic_path = os.path.join(intri_dir, "intrinsic_color.txt")
    #np.savetxt(color_intrinsic_path, intrinsic[0], fmt="%.6f")
    #print(f"Saved color intrinsics → {color_intrinsic_path}")

    i3 = intrinsic[0]  # shape (3,3)

    # 拼接成 3×4：最后一列 (tx, ty, tz) 设为 (0,0,0)
    depth_intrinsic_3x4 = np.array([
        [i3[0,0], i3[0,1], i3[0,2], 0.0],  # tx = 0
        [i3[1,0], i3[1,1], i3[1,2], 0.0],  # ty = 0
        [i3[2,0], i3[2,1], i3[2,2], 0.0],
        [0.0, 0.0,  0.0,  1.0],  # tz = 0
    ])

    # 1) 保存给 SAM3D 用的 3×4 深度内参
    depth_intrinsic_path = os.path.join(intri_dir, "intrinsic_depth.txt")
    np.savetxt(depth_intrinsic_path, depth_intrinsic_3x4, fmt="%.6f")
    print(f"Saved 3×4 depth intrinsics → {depth_intrinsic_path}")

    # 2) 保存给彩色图用的 3×3 内参（仍可保留）
    color_intrinsic_path = os.path.join(intri_dir, "intrinsic_color.txt")
    np.savetxt(color_intrinsic_path, i3, fmt="%.6f")
    print(f"Saved 3×3 color intrinsics → {color_intrinsic_path}")
    #<<< added end


    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)
        valid_mask = np.all(np.isfinite(points_3d), axis=1)
        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
""" 
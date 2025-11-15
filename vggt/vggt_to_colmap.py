import os
import numpy as np
from read_write_model import Image, Camera, Point3D, write_model

def vggt_to_colmap(image_paths, extrinsics, intrinsics, pointcloud_xyz,
                   output_path, width=518, height=518):
    os.makedirs(output_path, exist_ok=True)

    cameras = {}
    images = {}
    points3D = {}

    # Use single camera model (shared intrinsics), or per-image
    for i, (img_path, ext, intr) in enumerate(zip(image_paths, extrinsics, intrinsics)):
        camera_id = i + 1
        image_id = i + 1
        img_name = os.path.basename(img_path)

        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        camera = Camera(
            id=camera_id,
            model=CameraModel.PINHOLE,
            width=width,
            height=height,
            params=np.array([fx, fy, cx, cy])
        )
        cameras[camera_id] = camera

        R = ext[:3, :3]
        t = ext[:3, 3]
        qvec = rotmat_to_qvec(R)
        image = Image(
            id=image_id,
            qvec=qvec,
            tvec=t,
            camera_id=camera_id,
            name=img_name,
            xys=np.zeros((0, 2)),  # No 2D-3D matches
            point3D_ids=np.array([], dtype=np.int64)
        )
        images[image_id] = image

    # Sparse point cloud (optional)
    for i, xyz in enumerate(pointcloud_xyz):
        points3D[i + 1] = Point3D(
            id=i + 1,
            xyz=xyz,
            rgb=np.array([255, 255, 255]),
            error=0.0,
            image_ids=[],
            point2D_idxs=[]
        )

    write_model(images, cameras, points3D, output_path, ext=".bin")
    print(f"✅ COLMAP model written to: {output_path}")

# Utility from COLMAP
def rotmat_to_qvec(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion [w, x, y, z]"""
    q = np.empty(4)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0)
        q[0] = 0.5 * s
        s = 0.5 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[1] = 0.5 * s
            s = 0.5 / s
            q[2] = (R[0, 1] + R[1, 0]) * s
            q[3] = (R[0, 2] + R[2, 0]) * s
            q[0] = (R[2, 1] - R[1, 2]) * s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[2] = 0.5 * s
            s = 0.5 / s
            q[1] = (R[0, 1] + R[1, 0]) * s
            q[3] = (R[1, 2] + R[2, 1]) * s
            q[0] = (R[0, 2] - R[2, 0]) * s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = 0.5 * s
            s = 0.5 / s
            q[1] = (R[0, 2] + R[2, 0]) * s
            q[2] = (R[1, 2] + R[2, 1]) * s
            q[0] = (R[1, 0] - R[0, 1]) * s
    return q
import open3d as o3d
import numpy as np
from os.path import join
from PIL import Image
from torchvision import transforms as TF
import cv2

# Load your point cloud (PLY, OBJ, etc.)
data_path = "/content/drive/MyDrive/vggt/vggt/final/plane_detection_data"
rgb_path = "/content/drive/MyDrive/vggt/vggt/final/plane_detection_data"
scene_name = "empty_room_new"

# -------------------------
# 1. Load and downsample
# -------------------------
pcd_data = np.load(join(data_path, scene_name, "point_map.npy"))

num_images = pcd_data.shape[0]
image_path_list = [join(data_path, scene_name, "color", "frame_" + str(i).zfill(4) + ".png") for i in
                   range(1, num_images + 1)]
images = []
to_tensor = TF.ToTensor()
for image_path in image_path_list:
    # Open image
    img = Image.open(image_path)
    img = to_tensor(img).numpy()
    images.append(img)

images = np.transpose(np.array(images), (0, 2, 3, 1))
org_colors = np.reshape(images, [-1, 3])

pts = np.reshape(pcd_data, [-1, 3])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

print(f"number of points: {len(pcd.points)}")
voxel_size = 0.005  # adjust depending on scene scale
pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"Downsampled points: {len(pcd_down.points)}")

# Copy downsampled cloud for plane detection
pcd_remaining = pcd_down
plane_threshold = 0.05  # max distance to plane for inliers (in meters)
corner_threshold = 0.02
num_iterations = 20000  # iterations for RANSAC
min_inliers = 10000  # ignore very small planes
# -------------------------
# 3. Iterative plane detection
# -------------------------
planes = []

while True:
    plane_model, inliers = pcd_remaining.segment_plane(
        distance_threshold=plane_threshold,
        ransac_n=3,
        num_iterations=num_iterations
    )
    if len(inliers) < min_inliers:
        print("No more large planes detected.")
        break

    print(f"Detected plane with {len(inliers)} points: {plane_model}")

    # Extract plane points from downsampled cloud
    plane_pcd_down = pcd_remaining.select_by_index(inliers)
    pcd_remaining = pcd_remaining.select_by_index(inliers, invert=True)

    planes.append(plane_model)  # store plane equation only


def project_to_plane(points, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    normal /= np.linalg.norm(normal)
    distances = (points @ normal) + d
    return points - np.outer(distances, normal), distances


# -------------------------
# 4. Flatten corresponding points in full-resolution cloud
# -------------------------

full_points = np.asarray(pcd.points)

# Step A: First identify planes
proj_points_set = []
distances_set = []
plane_masks = []
for plane_model in planes:
    proj_points, distances = project_to_plane(full_points, plane_model)
    proj_points_set.append(proj_points)
    distances_set.append(distances)
    mask = np.abs(distances) < corner_threshold
    plane_masks.append(mask)

# Step B: Ignore overlapping regions
corner_mask = np.zeros((full_points.shape[0]), dtype=bool)
for i in range(len(planes)):
    for j in range(i + 1, len(planes)):
        # Points close to both planes, so is corner region
        mask = plane_masks[i] & plane_masks[j]
        corner_mask = corner_mask | mask

# Step C: Project to planes
# Find the minimum distances
distances_array = np.array(distances_set)

plane_index = np.argmin(np.abs(distances_array), axis=0)

proj_points_array = np.array(proj_points_set)

full_points_flattened = proj_points_array[plane_index, np.arange(plane_index.shape[0]), :]

full_points[~corner_mask, :] = full_points_flattened[~corner_mask, :]


# Update full-resolution point cloud
pcd.points = o3d.utility.Vector3dVector(full_points)

# -------------------------
# 5. Save flattened full-resolution cloud
# -------------------------
o3d.io.write_point_cloud(join(data_path, scene_name, "flattened_full_points.ply"), pcd)
print("Flattened full-resolution point cloud saved.")


# -------------------------
# 6. Calculate Extrinsics
# -------------------------
points_image = full_points.reshape((num_images, -1, 3))
H = pcd_data.shape[1]
W = pcd_data.shape[2]
# Create a grid of coordinates
ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  # shape (H,W)
# Flatten
xs_flat = xs.flatten()  # x-coordinates (columns)
ys_flat = ys.flatten()  # y-coordinates (rows)
# Stack as Nx2 array
image_points_2d_for_image = np.stack([xs_flat, ys_flat], axis=1)  # shape (H*W, 2)

intrinsic = np.load(join(data_path, scene_name, "intrinsic.npy"))
extrinsic = np.zeros((1, num_images, 3, 4))

for i in range(num_images):
    objectPoints = np.array(points_image[i, :, :], dtype=np.float32)  # Nx3
    imagePoints = np.array(image_points_2d_for_image, dtype=np.float32)      # Nx2
    K = np.array(intrinsic[0, i, :, :], dtype=np.float32)   # from VGGT

    retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, K, None)
    R, _ = cv2.Rodrigues(rvec)
    # Store R and tvec as the new extrinsics for OpenMVS
    extrinsic[0, i, :, :] = np.hstack([R, tvec])


np.save(join(data_path, scene_name, "extrinsic_adjusted.npy"), extrinsic)


# -------------------------
# 7. Optional: visualize
# -------------------------

pcd.colors = o3d.utility.Vector3dVector(org_colors)
# o3d.io.write_point_cloud(join(data_path, scene_name, "flattened_full_points_with_color.ply"), pcd)
o3d.visualization.draw_geometries([pcd])

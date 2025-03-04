import numpy as np
import open3d as o3d
import cv2

def depth_to_point_cloud(depth_map, intrinsic_matrix, camera_pose):
    """Projects a depth map into 3D space using camera intrinsics and transforms to global frame."""
    height, width = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u, v = u.flatten(), v.flatten()
    z = depth_map.flatten()
    
    # Filter out invalid depth values
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]
    
    # Compute normalized camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Create homogeneous coordinates and transform to global frame
    points = np.vstack((x, y, z, np.ones_like(z)))
    transformed_points = (camera_pose @ points)[:3].T
    
    return transformed_points

def visualize_multiple_depth_maps(depth_map_files, intrinsic_matrices, camera_poses):
    """Visualizes multiple depth maps as a combined point cloud in a global frame."""
    all_points = []
    
    for depth_map_file, intrinsic_matrix, camera_pose in zip(depth_map_files, intrinsic_matrices, camera_poses):
        depth_map = cv2.imread(depth_map_file, cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            continue
        points = depth_to_point_cloud(depth_map, intrinsic_matrix, camera_pose)
        all_points.append(points)
    
    if not all_points:
        print("No valid depth maps provided.")
        return
    
    all_points = np.vstack(all_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # Compute point cloud extents for scaling
    min_bound, max_bound = np.min(all_points, axis=0), np.max(all_points, axis=0)
    scale = np.linalg.norm(max_bound - min_bound) * 0.1  # Scale factor
    
    # Create coordinate frame (axes) scaled accordingly
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, axis],
                                       window_name="3D Point Cloud",
                                       zoom=0.8,
                                       front=[0, 0, -1],
                                       lookat=[0, 0, 1],
                                       up=[0, -1, 0])

if __name__ == "__main__":
    depth_map_files = [
        "/Users/allen/sources/pallas/luz-py/barty_side_depth_large.png",
        # "/Users/allen/sources/pallas/luz-py/barty_depth.png",
        # "/Users/allen/sources/pallas/luz-py/cats_depth2.png",
        # "/Users/allen/sources/pallas/luz-py/cats_depth3.png"        
    ]
    intrinsic_matrices = [np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1]])] * len(depth_map_files)
    camera_poses = [np.eye(4)] * len(depth_map_files)
    
    visualize_multiple_depth_maps(depth_map_files, intrinsic_matrices, camera_poses)

import numpy as np
import open3d as o3d

# Generate a simple mock point cloud (e.g., a 3D cube)
points = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],  # Bottom face
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],  # Top face
    ],
    dtype=np.float32,
)

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

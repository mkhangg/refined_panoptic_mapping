#! usr/bin/python3

# Import neccessary libraries
import sys
import open3d as o3d

# Check input arguments
try:
    pcd_path = sys.argv[1]
except:
    sys.exit(">> ERROR USAGE: python3 view_pcd.py <folder/filename>")

# Read point cloud from provided path
pcd = o3d.io.read_point_cloud(f"{pcd_path}")

# Flip point cloud and paint with neutral color for better visualization
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

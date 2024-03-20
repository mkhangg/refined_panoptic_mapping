import copy
import numpy as np
import open3d as o3d
import seaborn as sns
import matplotlib.pyplot as plt

from KDEpy import FFTKDE
from scipy.signal import find_peaks


def remove_outliers(depth_values):
    zs = depth_values.flatten()
    zs = zs[zs > 0]
    xz, yz = FFTKDE(kernel='gaussian', bw='ISJ', norm=1).fit(zs).evaluate()

    peaks, _ = find_peaks(yz, max(yz))
    left_bound = np.where(yz[:peaks[0]] < 1e-6)[0][-1]
    right_bound = np.where(yz[peaks[0]:] < 1e-6)[0][0] + peaks[0]

    near_z, far_z = xz[left_bound], xz[right_bound]
    return xz, yz, near_z, far_z


no_objects = 5
folder = '/mnt/data/workspace/grasp/semantic_scene_perception/cap_data/exp11'
sns_colors = sns.color_palette("tab10", int(no_objects))
cmap_color = 'OrRd'
width = 2
offset = 3000

fig, axs = plt.subplots(5, int(no_objects), figsize=(18, 12))
noised_scene = o3d.geometry.PointCloud()
filtered_scene = o3d.geometry.PointCloud()

for i in range(int(no_objects)):

    pcd = o3d.io.read_point_cloud(filename=f'{folder}/masked_point_cloud_{i}.pcd')
    pcd.paint_uniform_color(sns_colors[i])
    noised_scene += copy.deepcopy(pcd)

    points = np.asarray(pcd.points)

    data = np.loadtxt(f'{folder}/depth_data.txt', delimiter=',')
    mask = np.loadtxt(f'{folder}/mask_{i}.txt', delimiter=',')
    axs[0, i].imshow(mask, interpolation='nearest', cmap='Greys_r')
    axs[0, i].axis('off')

    data[mask == 0] = 0
    axs[1, i].imshow(data, interpolation='nearest', cmap=cmap_color)
    axs[1, i].axis('off')
    
    xz, yz, near_z, far_z = remove_outliers(data)
    axs[2, i].plot(xz, yz, linewidth=width)
    axs[2, i].axvline(x=near_z, color='r', linewidth=width)
    axs[2, i].axvline(x=far_z, color='r', linewidth=width)
    axs[2, i].fill_between(xz, yz, where=(xz >= near_z) & (xz <= far_z), 
                           color='grey', alpha=0.25)
    axs[2, i].grid(True, which='both', axis='both', 
                   linestyle='--', linewidth=width/5)

    data[(data < near_z) | (data > far_z)] = 0
    axs[3, i].imshow(data, interpolation='nearest', cmap=cmap_color)
    axs[3, i].axis('off')

    mask[data == 0] = 0
    axs[4, i].imshow(mask, interpolation='nearest', cmap='Greys_r')
    axs[4, i].axis('off')

    filtered_points = points[(points[:, 2] >= near_z) & (points[:, 2] <= far_z)]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.paint_uniform_color(sns_colors[i])
    filtered_scene += filtered_pcd

plt.tight_layout()
plt.show()

sbs_scene = noised_scene.translate((-offset, 0 ,0)) + filtered_scene.translate((offset, 0 ,0))
sbs_scene.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([sbs_scene])

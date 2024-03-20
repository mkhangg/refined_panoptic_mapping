import numpy as np
import matplotlib.pyplot as plt

from KDEpy import FFTKDE
from scipy.signal import find_peaks


def remove_outliers(depth_values):
    zs = depth_values.flatten()
    zs = zs[zs > 0]
    xz, yz = FFTKDE(kernel='gaussian', bw='ISJ', norm=1).fit(zs).evaluate()

    peaks, _ = find_peaks(yz, max(yz))
    left_bound = np.where(yz[:peaks[0]] < max(yz[:peaks[0]]/20))[0][-1]
    right_bound = np.where(yz[peaks[0]:] < max(yz[:peaks[0]]/20))[0][0] + peaks[0]
    near_z, far_z = xz[left_bound], xz[right_bound]

    return xz, yz, near_z, far_z

no_masks = 5
fig, axs = plt.subplots(5, int(no_masks))
folder = '/mnt/data/workspace/grasp/semantic_scene_perception/cap_data/exp12'
# folder = './test'

for i in range(int(no_masks)):
    data = np.loadtxt(f'{folder}/depth_data.txt', delimiter=',')
    mask = np.loadtxt(f'{folder}/mask_{i}.txt', delimiter=',')
    # data = np.loadtxt(f'{folder}/depth_map.txt', delimiter=',')
    # mask = np.loadtxt(f'{folder}/mask_obj_{i}.txt', delimiter=',')
    axs[0, i].imshow(mask, interpolation='nearest')

    data[mask == 0] = 0
    axs[1, i].imshow(data, interpolation='nearest')

    xz, yz, near_z, far_z = remove_outliers(data)
    axs[2, i].plot(xz, yz)
    axs[2, i].axvline(x=near_z, color='r')
    axs[2, i].axvline(x=far_z, color='r')

    data[(data < near_z) | (data > far_z)] = 0
    axs[3, i].imshow(data, interpolation='nearest')

    mask[data == 0] = 0
    axs[4, i].imshow(mask, interpolation='nearest')


plt.tight_layout()
plt.show()

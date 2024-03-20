import os
import cv2
import math
import copy
import time
import numpy as np
import open3d as o3d

from PIL import Image
# from KDEpy import FFTKDE


class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False] 
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


def count_subfolders(path):
    count = 0
    for _, dirs, _ in os.walk(path):
        count += len(dirs)
    return count


def mask_depth_data_using_target_image(depth_data, target_image):

    target_height, target_width = target_image.shape[0], target_image.shape[1]

    for i in range(target_height):
        for j in range(target_width):
            if int(target_image[i][j]) == 0:
                depth_data[i][j] = 0

    return depth_data


def create_masked_pc(depth_data, target_mask, 
                     CX_RGB, CY_RGB, FX_RGB, FY_RGB,
                     CX_DEPTH, CY_DEPTH, FX_DEPTH, FY_DEPTH,
                     R, T, bColor=True, THRESHOLD=0, MAX_THRESHOLD=255):

    rectified_depth_data = mask_depth_data_using_target_image(depth_data, target_mask)
    # rectified_depth_data = depth_data

    height, width = rectified_depth_data.shape
    n, m = np.tile(range(width), height), np.repeat(range(height), width)
    z = rectified_depth_data.reshape(height*width)
    x, y = (n - CX_DEPTH) * z / FX_DEPTH, (m - CY_DEPTH) * z / FY_DEPTH

    pcd = np.dstack((x, y, z)).reshape((height*width, 3))
    rgbd_pcd = o3d.geometry.PointCloud()
    rgbd_pcd.points = o3d.utility.Vector3dVector(pcd)

    rgb_img = cv2.merge((target_mask, target_mask, target_mask))
    _, rgb_img = cv2.threshold(rgb_img, THRESHOLD, MAX_THRESHOLD, cv2.THRESH_BINARY)

    if bColor == True:
        cam_RGB = np.apply_along_axis(np.linalg.inv(R).dot, 1, pcd) - np.linalg.inv(R).dot(T)
        v_rgb = ((cam_RGB[:, 0] * FX_RGB) / cam_RGB[:, 2] + CX_RGB + width / 2).astype(int).clip(0, width - 1)
        u_rgb = ((cam_RGB[:, 1] * FY_RGB) / cam_RGB[:, 2] + CY_RGB).astype(int).clip(0, height - 1)
        colors = rgb_img[u_rgb, v_rgb]/255
        rgbd_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    return rgbd_pcd, rgb_img


# def create_mask(target_mask, THRESHOLD=0, MAX_THRESHOLD=255):
#     rgb_img = cv2.merge((target_mask, target_mask, target_mask))
#     _, rgb_img = cv2.threshold(rgb_img, THRESHOLD, MAX_THRESHOLD, cv2.THRESH_BINARY)
#     return rgb_img


def partition_cloud(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    return outlier_cloud, inlier_cloud


def remove_outliers(pcd, Z_THRESHOLD_MIN=5):
    removal_indexes = []
    pcd_pts = np.asarray(pcd.points)
    for i in range(len(pcd_pts)):
        if pcd_pts[i][2] <= Z_THRESHOLD_MIN:
            removal_indexes.append(i)

    inlier_cloud, _ = partition_cloud(pcd, removal_indexes)

    return inlier_cloud


def remove_outliers_in_scene(pcd, Z_THRESHOLD_MIN=0, Z_THRESHOLD_MAX=2500):
    removal_indexes = []
    pcd_pts = np.asarray(pcd.points)
    for i in range(len(pcd_pts)):
        if pcd_pts[i][2] >= Z_THRESHOLD_MAX or pcd_pts[i][2] <= Z_THRESHOLD_MIN:
            removal_indexes.append(i)

    inlier_cloud, _ = partition_cloud(pcd, removal_indexes)

    return inlier_cloud


def translate_pcd(pcd, direcs):
    trans_pcd = copy.deepcopy(pcd).translate((direcs[0], \
                                            direcs[1], \
                                            direcs[2]))
    return trans_pcd


def aggregate_mask_for_rgb_image(rgb_image, masks):
    h, w, c = masks[0].shape
    aggregated_mask = np.full((h, w, c), (0, 0, 0), dtype=np.uint8)
    for mask in masks:
        aggregated_mask = cv2.add(aggregated_mask, mask.astype('uint8'))

    aggregated_mask = np.array(cv2.cvtColor(aggregated_mask, cv2.COLOR_BGR2GRAY))
    masked_rgb_img = cv2.bitwise_and(rgb_image, rgb_image, mask=aggregated_mask)

    return masked_rgb_img


def aggregate_mask_for_depth_image(depth_image, masks):
    h, w, c = masks[0].shape
    aggregated_mask = np.full((h, w, c), (0, 0, 0), dtype=np.uint8)
    for mask in masks:
        aggregated_mask = cv2.add(aggregated_mask, mask.astype('uint8'))

    aggregated_mask = np.array(cv2.cvtColor(aggregated_mask, cv2.COLOR_BGR2GRAY))
    masked_depth_img = cv2.bitwise_and(depth_image, depth_image, mask=aggregated_mask)

    return masked_depth_img


def segmap_gen(masks):
    h, w, c = masks[0].shape
    aggregated_mask = np.full((h, w, c), (0, 0, 0), dtype=np.uint8)
    for mask in masks:
        aggregated_mask = cv2.add(aggregated_mask, mask.astype('uint8'))

    aggregated_mask = np.array(cv2.cvtColor(aggregated_mask, cv2.COLOR_BGR2GRAY))

    return aggregated_mask


def find_matching_features(img1, img2, best_n):
    grey_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb, matcher = cv2.ORB_create(), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    first_kps, first_dsps = orb.detectAndCompute(grey_img1, None)
    second_kps, second_dsps = orb.detectAndCompute(grey_img2, None)
    matches = matcher.match(first_dsps, second_dsps)
    matches = sorted(matches, key=lambda x:x.distance)

    # Just take N best matches
    matches = matches[:best_n]

    return matches, first_kps, second_kps


def find_matching_features_coords(img1, img2, best_n):
    grey_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb, matcher = cv2.ORB_create(), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    first_kps, first_dsps = orb.detectAndCompute(grey_img1, None)
    second_kps, second_dsps = orb.detectAndCompute(grey_img2, None)
    matches = matcher.match(first_dsps, second_dsps)
    matches = sorted(matches, key=lambda x:x.distance)

    # Just take N best matches
    matches = matches[:best_n]

    list_kp1, list_kp2 = [], []
    for match in matches:
        list_kp1.append(first_kps[match.queryIdx].pt)
        list_kp2.append(second_kps[match.trainIdx].pt)

    return matches, first_kps, second_kps, list_kp1, list_kp2


def correspondance_to_3dpoints(correspondances_depth_data, correspondances_coordinates, 
                             CX_DEPTH, CY_DEPTH, FX_DEPTH, FY_DEPTH):
    pointset = []
    for j in range(len(correspondances_depth_data)):
        (n, m) = correspondances_coordinates[j]
        z = correspondances_depth_data[j]
        x, y = (n - CX_DEPTH) * z / FX_DEPTH, (m - CY_DEPTH) * z / FY_DEPTH
        pointset.append([x, y, z])

    return np.asmatrix(pointset)


def correspondance2pointcloud(correspondances_depth_data, correspondances_coordinates, 
                             CX_DEPTH, CY_DEPTH, FX_DEPTH, FY_DEPTH):
    pointset = []
    for j in range(len(correspondances_depth_data)):
        (n, m) = correspondances_coordinates[j]
        z = correspondances_depth_data[j]
        x, y = (n - CX_DEPTH) * z / FX_DEPTH, (m - CY_DEPTH) * z / FY_DEPTH
        pointset.append([x, y, z])

    correspondance_pointcloud = o3d.geometry.PointCloud()
    correspondance_pointcloud.points = o3d.utility.Vector3dVector(pointset)
    return correspondance_pointcloud


def get_3D_points_from_2d_correspondences(correspondances_depth_data, correspondances_coordinates, 
                                      CX_DEPTH, CY_DEPTH, FX_DEPTH, FY_DEPTH, topointcloud=False):
    pointset = []
    for j in range(len(correspondances_depth_data)):
        (n, m) = correspondances_coordinates[j]
        z = correspondances_depth_data[j]
        x, y = (n - CX_DEPTH) * z / FX_DEPTH, (m - CY_DEPTH) * z / FY_DEPTH
        pointset.append([x, y, z])

    if topointcloud == True:
        correspondances_3d = o3d.geometry.PointCloud()
        correspondances_3d.points = o3d.utility.Vector3dVector(pointset)
    elif topointcloud == False:
        correspondances_3d = np.asmatrix(pointset)

    return correspondances_3d


def convert2transparent(EXP_IDS, folder):

    for exp_id in EXP_IDS:
        img = Image.open(f"cap_data/exp{exp_id}/masked_rgb_image.jpg")
        img = img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            # if (item[0] > 240 and item[1] > 240 and item[2] > 240) or (item[0] < 10 and item[1] < 10 and item[2] < 10):
            if item[0] < 10 and item[1] < 10 and item[2] < 10:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        img.save(f"{folder}/exp{int(EXP_IDS.index(exp_id))}.png", "PNG")


def extract_xyz_coordinates(object_pcd):
    pts = np.asarray(object_pcd.points)
    x_coords, y_coords, z_coords = pts[:, 0], pts[:, 1], pts[:, 2]
    return x_coords, y_coords, z_coords


def get_probs_by_dimension(coords, pdf_x, pdf_y):
    probs = []
    for i in range(len(coords)):
        distances = []
        for j in range(len(pdf_x)):
            distances.append(abs(coords[i] - pdf_x[j]))
        probs.append(pdf_y[distances.index(min(distances))])
    return np.array(probs)


# def kde_weighing(data, kernel, optim_bw):

#     # pcd_file_path = "exp_feature/first_correspondance_pcd.pcd"
#     # correspondance_pcd = o3d.io.read_point_cloud(filename=pcd_file_path)
#     correspondance_pcd = data

#     xs, ys, zs = extract_xyz_coordinates(correspondance_pcd)

#     start_weighing_time = time.time()
#     xx, yx = FFTKDE(kernel=kernel, bw=optim_bw, norm=1).fit(xs).evaluate()
#     end_weighing_time = time.time()

#     xy, yy = FFTKDE(kernel=kernel, bw=optim_bw, norm=1).fit(ys).evaluate()
#     xz, yz = FFTKDE(kernel=kernel, bw=optim_bw, norm=1).fit(zs).evaluate()
#     print(f">> Weighing time: {1000*(end_weighing_time - start_weighing_time):.4f} ms.")

#     probs_x = get_probs_by_dimension(xs, xx, yx)
#     probs_y = get_probs_by_dimension(ys, xy, yy)
#     probs_z = get_probs_by_dimension(zs, xz, yz)
#     probs = np.array((probs_x * probs_y * probs_z))

#     return probs

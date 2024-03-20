#! usr/bin/python3

# Import neccessary libraries
import os
import cv2
import copy
import numpy as np
import open3d as o3d
import configargparse
import pyrealsense2 as rs

# Import utility functions 
from PIL import Image
from KDEpy import FFTKDE
from scipy.signal import find_peaks

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ConfigParser(configargparse.ArgParser):
    """
    Configuration parser to parse configuration parameters for the pipeline.
    """

    def __init__(self):
        super().__init__(default_config_files=[os.path.join(os.path.dirname(__file__), './config.yml')],
                         conflict_handler='resolve')

        # Pipline parameters
        self.add('--device', type=str, default='cuda', help='Device to run the pipeline.')
        self.add('--engine', type=str, choices=['tensor', 'legacy'], help='.')
        self.add('--multiprocessing', action='store_true', help='Use multiprocessing in operations.')
        
        # Camera parameters
        self.add('--img_width', type=float, help='Image width of RGB-D images.')
        self.add('--img_height', type=float, help='Image height of RGB-D images.')
        self.add('--fps', type=int, help='Frame rate of RGB-D stream for captured device.')
        self.add('--depth_min', type=float, help='Min clipping distance (in meter) for input depth data.')
        self.add('--depth_max', type=float, help='Max clipping distance (in meter) for input depth data.')

        # Scene update parameters
        self.add('--voxel_size', type=float, help='Voxel size in meter for volumetric integration.')
        self.add('--block_resolution', type=int, help='Block resolution for volumetric integration.')
        self.add('--trunc_voxel_multiplier', type=float, help='Truncation distance multiplier for signed distance.')
        self.add('--est_point_count', type=int, help='Estimated point cloud size for surface extraction.')
        self.add('--block_count', type=int, help='Pre-allocated voxel block count for volumetric integration.')
        self.add('--point_weight_thr', type=float, help='Threshold to filter outliers for point cloud reconstruction.')
        self.add('--surface_weight_thr', type=float, help='Threshold to filter outliers for surface reconstruction.')
        self.add('--wait_frames', type=int, help='Frames to wait when no detections or no new objects detected.')
        self.add('--update_mesh_surface', action='store_true', help='Update the scene mesh surface or not.')
        self.add('--do_semantic', action='store_true', help='Perform segmentation or not.')
        self.add('--raycast_box', action='store_true', help='Use raycast TSDF or not.')
        self.add('--recolorize', action='store_true', help='Recolorize the segmented objects or not.')
        self.add('--use_kde', action='store_true', help='Remove outliers or not.')

        # GUI options
        self.add('--win_width', type=int, help='Width of GUI window.')
        self.add('--win_height', type=int, help='Height of GUI window.')
        self.add('--skip_frames', type=int, help='Number of skipping frames to guarantee auto-exposuring.')
        self.add('--trajectory_interval', type=int, help='Frame interval to draw camera trajectory.')

        # Segmentation model configurations
        self.add('--seg_model', type=str, help='Segmentation model path.')
        self.add('--seg_device', type=str, help='device to run segmentation model.')

        # Running option
        self.add('--save_data', action='store_true', help='Save scene and camera trajectory or not.')
        self.add('--save_folder', type=str, help='Name of folder to save to.')
        self.add('--save_mesh', action='store_true', help='Save scene as mesh or not.')
        self.add('--save_traj_log', action='store_true', help='Save camera trajectory as log file or not.')


    def get_config(self):
        """
        Parse the command-line arguments and resolve any conflicts based on the configuration.
        
        Returns:
            config: A parsed configuration object containing the specified parameters.
        """
        
        config = self.parse_args()

        # Resolve conflicts
        if config.engine == 'legacy':
            if config.device.lower().startswith('cuda'):
                print('Legacy engine only supports CPU.', 'Fallback to CPU.')
                config.device = 'CPU:0'

        elif config.engine == 'tensor':
            if config.multiprocessing:
                print('Tensor engine does not support multiprocessing. Disabled.')
                config.multiprocessing = False

            if (config.device.lower().startswith('cuda') and (not o3d.core.cuda.is_available())):
                print('No CUDA support or no CUDA device available. Fallback to CPU.')
                config.device = 'CPU:0'

        return config


def colorstr(*input):
    """
    Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, 
        i.e.  colorstr('bold', 'hello world!').
    """

    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m'}
        
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def parse_trajectory(log_file):
    """
    Parse a trajectory log file and extract matrices from it.

    Args:
        log_file (str): Path to the trajectory log file.

    Returns:
        List of numpy arrays: Matrices extracted from the log file.
    """

    matrices = []
    with open(log_file, 'r') as file:
        lines = file.readlines()
    for i in range(0, len(lines), 5):
        matrix_lines = lines[i+1 : i+5] 
        if len(matrix_lines) < 4:
            break
        matrix = np.array([list(map(float, line.strip().split())) for line in matrix_lines])
        matrices.append(matrix)

    return matrices


def convert_mesh_to_pcd(pcd):
    """
    Convert a mesh to a point cloud.

    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.

    Returns:
        open3d.geometry.PointCloud: New point cloud generated from the input mesh.
    """

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = pcd.points 
    new_pcd.colors = pcd.colors 

    return new_pcd


def sns_color_to_255_scale(sns_colors):
    """
    Convert a seaborn color from the [0, 1] scale to the [0, 255] scale.

    Args:
        sns_colors (tuple): RGB tuple representing the seaborn color.

    Returns:
        tuple: RGB tuple scaled to the [0, 255] scale.
    """

    return (int(sns_colors[0]*255), int(sns_colors[1]*255), int(sns_colors[2]*255))


def count_subfolders(path):
    """
    Count the number of subfolders within a given directory.

    Args:
        path (str): Path to the directory.

    Returns:
        int: Number of subfolders within the directory.
    """

    count = 0
    for _, dirs, _ in os.walk(path):
        count += len(dirs)
    return count


def get_next_frame(pipeline, align, skip=None):
    """
    Get the next frame from the RealSense pipeline and align if needed.

    Args:
        pipeline (rs.pipeline): RealSense pipeline object.
        align (rs.align): RealSense align object for aligning frames.
        skip (bool, optional): Flag to indicate whether to skip or not. Defaults to None.

    Returns:
        tuple: A tuple containing color and depth images (along with intrinsics if skip is True).
    """

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    
    color_frame, depth_frame = aligned_frames.get_color_frame(), aligned_frames.get_depth_frame()
    color_image, depth_image = np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data())

    if skip == True:
        color_intrin = rs.video_stream_profile(color_frame.profile).get_intrinsics()
        CX_RGB, CY_RGB, FX_RGB, FY_RGB = color_intrin.ppx, color_intrin.ppy, color_intrin.fx, color_intrin.fy
        return color_image, depth_image, CX_RGB, CY_RGB, FX_RGB, FY_RGB

    return color_image, depth_image


def colorize_segment_blob(one_channel_array, color_for_one):
    """
    Colorize segmented blobs represented by a single-channel array.

    Args:
        one_channel_array (numpy.ndarray): Single-channel array representing segmented blobs.
        color_for_one (tuple): RGB color tuple to be assigned to the segmented blob.

    Returns:
        numpy.ndarray: RGB image representing the segmented blobs with assigned colors.
    """

    rgb_image = np.full((*one_channel_array.shape, 3), [255, 255, 255], dtype=np.uint8)
    rgb_image[np.where(one_channel_array == 1)] = color_for_one
    
    return rgb_image


def mask_rgb_image(rgb_image, masks):
    """
    Apply a set of masks to an RGB image and return the masked RGB image.

    Args:
        rgb_image (numpy.ndarray): RGB image to be masked.
        masks (list of numpy.ndarray): List of masks to be applied.

    Returns:
        numpy.ndarray: Masked RGB image.
    """

    h, w, c = masks[0].shape
    aggregated_mask = np.full((h, w, c), (0, 0, 0), dtype=np.uint8)
    for mask in masks:
        aggregated_mask = cv2.add(aggregated_mask, mask.astype('uint8'))

    aggregated_mask = np.array(cv2.cvtColor(aggregated_mask, cv2.COLOR_BGR2GRAY))
    masked_rgb_img = cv2.bitwise_and(rgb_image, rgb_image, mask=aggregated_mask)

    return masked_rgb_img


def remove_outliers(depth_values, target_mask):
    """
    Remove outliers from a set of depth values.

    Args:
        depth_values (numpy.ndarray): Array containing depth values.
        target_mask (numpy.ndarray): Array containing binary mask values.

    Returns:
        tuple: Tuple containing xz and yz values from the kernel density, 
               along with the near and far depth values determined as outliers.
    """

    depth_values[target_mask == 0] = 0
    zs = depth_values.flatten()
    zs = zs[zs > 0]
    
    # Normalize depth data for better convergence rate (Ref: https://github.com/tommyod/KDEpy/issues/133)
    zs_min, zs_max = zs.min(), zs.max()
    zs = (zs - zs_min) / (zs_max - zs_min)
    xz, yz = FFTKDE(kernel='gaussian', bw='ISJ', norm=1).fit(zs).evaluate()
    xz = xz * (zs_max - zs_min) + zs_min        # Revert the data backward

    peaks, _ = find_peaks(yz, max(yz))
    left_bound = np.where(yz[:peaks[0]] < max(yz[:peaks[0]]/20))[0][-1]
    right_bound = np.where(yz[peaks[0]:] < max(yz[:peaks[0]]/20))[0][0] + peaks[0]
    near_z, far_z = xz[left_bound], xz[right_bound]

    return xz, yz, near_z, far_z


def mask_refinement(depth_values, mask, lower_z, upper_z):
    """
    Refine a mask based on lower and upper depth thresholds.

    Args:
        depth_values (numpy.ndarray): Array containing depth values.
        mask (numpy.ndarray): Mask to be refined.
        lower_z (float): Lower depth threshold.
        upper_z (float): Upper depth threshold.

    Returns:
        tuple: A tuple containing the refined mask and depth values.
    """

    depth_values[(depth_values < lower_z) | (depth_values > upper_z)] = 0
    mask[depth_values == 0] = 0

    return mask, depth_values


def perceive_scene(device, model, names, color_image, depth_map, colors, b_recolorize, b_remove_outliers):
    """
    Perform scene perception using segmentation for objects of interest.

    Args:
        device (str): Device to run the model on.
        model: Object detection and segmentation model.
        names (dict): Dictionary mapping class indices to class names.
        color_image (numpy.ndarray): RGB color image.
        depth_map (numpy.ndarray): Depth map.
        colors (list): List of colors for recolorization.
        b_recolorize (bool): Flag indicating whether to recolorize the masks or not.
        b_remove_outliers (bool): Flag indicating whether to remove outliers or not.

    Returns:
        tuple: A tuple containing boolean flag indicating detections, array of object IDs, and masked RGB image.
    """

    b_detections = False
    color_image_RGB_format = Image.fromarray(cv2.cvtColor(copy.deepcopy(color_image), cv2.COLOR_BGR2RGB))
    results = model.track(source=color_image_RGB_format, save=False, save_txt=False, save_conf=True, 
                          save_crop=False, verbose=False, show=False, show_labels=True, show_conf=True, 
                          retina_masks=True, show_boxes=False, device=device, tracker="bytetrack.yaml")

    if results[0].boxes.id == None:
        masked_rgb_image = np.full((color_image.shape[0], color_image.shape[1], 3), 255, dtype=np.uint8)
        return b_detections, np.array([]), np.asanyarray(masked_rgb_image)
    else:   # Abnormal behavior fixed (Ref: https://github.com/ultralytics/ultralytics/issues/4315)
        boxes = results[0].boxes.cpu()
        masks = results[0].cpu().masks.data.numpy().transpose(1, 2, 0)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
    
    b_detections = True
    pred_classes = [int(boxes[i].cls.item()) for i in range(len(boxes))]
    pred_names = [names[int(pred_class)] for pred_class in pred_classes]    
    target_obj_ids = [list(names.keys())[list(names.values()).index(target_object)] for target_object in pred_names]
    
    target_masks = []
    traced_pred_classes = copy.deepcopy(pred_classes)
    for target_obj_id in target_obj_ids:
        target_masks.append(masks[:, :, traced_pred_classes.index(target_obj_id)])
        traced_pred_classes[traced_pred_classes.index(target_obj_id)] = -1                      # mark as done
    
    if b_recolorize == True:
        masked_rgb_image = np.zeros((color_image.shape[0], color_image.shape[1], 3), dtype=np.uint8)
        for target_mask in range(len(target_masks)):
            if b_remove_outliers == True:
                _, _, near_z, far_z = remove_outliers(copy.deepcopy(depth_map), 
                                                      copy.deepcopy(target_masks[target_mask]))
                # np.savetxt(f"./test/mask_obj_{target_mask}.txt", 
                #            np.asanyarray(target_masks[target_mask]), delimiter=',', fmt='%d')
                refined_target_mask, _ = mask_refinement(copy.deepcopy(depth_map), 
                                                         copy.deepcopy(target_masks[target_mask]), 
                                                         near_z, far_z)
            elif b_remove_outliers == False:
                refined_target_mask = target_masks[target_mask]
                # np.savetxt(f"./test/mask_obj_{target_mask}.txt", 
                #            np.asanyarray(refined_target_mask), delimiter=',', fmt='%d')
            masked_rgb_image += colorize_segment_blob(refined_target_mask, colors[int(target_obj_ids[target_mask])])
    
    elif b_recolorize == False:
        mask_images = []
        for target_mask in range(len(target_masks)):
            if b_remove_outliers == True:
                _, _, near_z, far_z = remove_outliers(copy.deepcopy(depth_map), 
                                                      copy.deepcopy(target_masks[target_mask]))
                refined_target_mask, _ = mask_refinement(copy.deepcopy(depth_map), 
                                                         copy.deepcopy(target_masks[target_mask]), 
                                                         near_z, far_z)
            elif b_remove_outliers == False:
                refined_target_mask = target_masks[target_mask]
            mask_image = cv2.merge((refined_target_mask, refined_target_mask, refined_target_mask))
            mask_images.append(mask_image)

        masked_rgb_image = np.asanyarray(mask_rgb_image(color_image, mask_images))
        masked_rgb_image[np.all(masked_rgb_image == [0, 0, 0], axis=-1)] = [255, 255, 255]      # mark as white 
    
    # cv2.imwrite(f"./test/masked_rgb_image.jpg", cv2.cvtColor(copy.deepcopy(masked_rgb_image), cv2.COLOR_BGR2RGB))
    # np.savetxt(f'./test/depth_map.txt', np.asanyarray(depth_map), delimiter=',', fmt='%d')

    return b_detections, np.array(ids), np.asanyarray(masked_rgb_image)


def remove_uninterested_pts(pcd):
    """
    Remove points from a point cloud based on color threshold.

    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.

    Returns:
        open3d.geometry.PointCloud: Filtered point cloud.
    """
    
    points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
    kept_indices = np.where(~np.all(colors > [220, 220, 220], axis=1))
    filtered_points, filtered_colors = points[kept_indices], colors[kept_indices]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd


def save_scene(volume, config, dest_folder, b_save_mesh=None):
    """
    Save the reconstructed scene to a file.

    Args:
        volume (open3d.integration.DenseSpatialMapping): Volumetric data structure representing the scene.
        config (ConfigParser): Configuration object.
        dest_folder (str): Destination folder to save the scene.
        b_save_mesh (bool, optional): Flag indicating whether to save the scene as a mesh or point cloud. 
    """

    filename = f"scene.ply" if b_save_mesh == True else f"scene.pcd"
    save_path = f"{dest_folder}/{filename}"

    if config.engine == 'legacy':
        if b_save_mesh == True:
            scene = volume.extract_triangle_mesh(weight_threshold=config.surface_weight_thr)
            scene.compute_vertex_normals()
            scene.compute_triangle_normals()
            o3d.io.write_triangle_mesh(save_path, scene, write_ascii=True)

        elif b_save_mesh == False:
            scene = volume.extract_point_cloud(weight_threshold=config.point_weight_thr)
            filtered_scene = remove_uninterested_pts(scene)
            o3d.io.write_point_cloud(save_path, filtered_scene, write_ascii=True)
            
    elif config.engine == 'tensor':
        if b_save_mesh == True:
            scene = volume.extract_triangle_mesh(weight_threshold=config.surface_weight_thr)
            scene = scene.to_legacy()
            o3d.io.write_triangle_mesh(save_path, scene, write_ascii=True)

        elif b_save_mesh == False:
            scene = volume.extract_point_cloud(weight_threshold=config.point_weight_thr)
            scene = scene.to_legacy()
            filtered_scene = remove_uninterested_pts(scene)
            o3d.io.write_point_cloud(save_path, filtered_scene, write_ascii=True)

    print(f">> {filename} is saved to {colorstr('bold', dest_folder)}/.")


def save_camera_trajectory(dest_folder, poses, b_save_traj_log=None):
    """
    Save the camera trajectory to a file.

    Args:
        dest_folder (str): Destination folder to save the trajectory.
        poses (list): List of camera poses.
        b_save_traj_log (bool, optional): Flag indicating whether to save as a log file or json.
    """
        
    filename = "trajectory.log" if b_save_traj_log == True else "trajectory.json"
    save_path = f"{dest_folder}/{filename}"
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    if b_save_traj_log == True:
        params = []
        trajectory = o3d.camera.PinholeCameraTrajectory()
        for pose in poses:
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic, param.extrinsic = intrinsic, np.linalg.inv(pose)
            params.append(param)
        trajectory.parameters = params
        o3d.io.write_pinhole_camera_trajectory(save_path, trajectory)

    elif b_save_traj_log == False:
        pose_graph = o3d.pipelines.registration.PoseGraph()
        for pose in poses:
            node = o3d.pipelines.registration.PoseGraphNode()
            node.pose = pose
            pose_graph.nodes.append(node)
        o3d.io.write_pose_graph(save_path, pose_graph)

    print(f">> {filename} is saved to {colorstr('bold', dest_folder)}/.")

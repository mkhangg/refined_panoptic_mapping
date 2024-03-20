#! usr/bin/python3

# Import neccessary libraries
import time
import threading
import numpy as np
import open3d as o3d
import seaborn as sns
import open3d.core as o3c
import pyrealsense2 as rs
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Import utility functions 
from pathlib import Path
from ultralytics import YOLO
from utils import (ConfigParser, sns_color_to_255_scale, count_subfolders, get_next_frame, 
                   perceive_scene, save_scene, save_camera_trajectory)


class RGBD_SLAM:

    def __init__(self, config):

        # Configure RealSense camera
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, int(config.img_width), int(config.img_height), 
                                rs.format.z16, config.fps)
        rs_config.enable_stream(rs.stream.color, int(config.img_width), int(config.img_height), 
                                rs.format.rgb8, config.fps)
        profile = self.pipeline.start(rs_config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.align_to_color = rs.align(rs.stream.color)
        config.depth_scale = int(np.ceil(1.0/(depth_sensor.get_depth_scale())))

        # Skip n first frames
        for _ in range(config.skip_frames):
            _, _ = get_next_frame(self.pipeline, self.align_to_color)
        
        self.config = config
        self.window = gui.Application.instance.create_window('', config.win_width, config.win_height, 0, 0)
        self.building_scene_window = gui.SceneWidget()
        self.building_scene_window.scene = rendering.Open3DScene(self.window.renderer)
        self.building_scene_window.scene.set_background([1, 1, 1, 1])
        
        w = self.window
        w.add_child(self.building_scene_window)
        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)        

        # Initialize GUI parameters
        self.is_done = False
        self.is_started = False
        self.is_running = True
        self.is_surface_updated = True
        
        gui.Application.instance.post_to_main_thread(self.window, self._on_start)
        threading.Thread(name='UpdateMain', target=self.update_main).start()


    def _on_layout(self, _):
        """
        Handle layout changes in the GUI.

        Args:
            _: Placeholder for the event argument (not used).
        """

        rect = self.window.content_rect
        self.building_scene_window.frame = gui.Rect(0, rect.y, rect.get_right(), rect.height)


    def _on_start(self):
        """
        Perform initialization when starting the application.
        """

        pcd_placeholder = o3d.t.geometry.PointCloud(o3c.Tensor(np.zeros((config.est_point_count, 3), dtype=np.float32)))
        pcd_placeholder.point.colors = o3c.Tensor(np.zeros((config.est_point_count, 3), dtype=np.float32))
        mat = rendering.MaterialRecord()
        mat.shader, mat.sRGB_color = 'defaultUnlit', True
        self.building_scene_window.scene.scene.add_geometry('points', pcd_placeholder, mat)
        self.model = o3d.t.pipelines.slam.Model(config.voxel_size, int(config.block_resolution), config.block_count, 
                                                o3c.Tensor(np.eye(4)), o3c.Device(config.device))
        self.is_started = True


    def _on_close(self):
        """
        Handle actions to be performed when the application is being closed.
        Sets a flag to indicate that the application is done.and creates an 
        experiment directory and saves captured data, if needed.
        """

        self.is_done = True

        # Create experiment directory and save capture data
        if config.save_data == True:
            exp_id = "" if count_subfolders(config.save_folder) == 0 else count_subfolders(config.save_folder)
            exp_dir = Path(f'./{config.save_folder}/exp{exp_id}')
            exp_dir.mkdir(exist_ok=True)
            dest_exp_folder = f"{config.save_folder}/exp{exp_id}"
            save_scene(self.model.voxel_grid, config, f'{dest_exp_folder}', b_save_mesh=config.save_mesh)
            save_camera_trajectory(f'{dest_exp_folder}', self.poses, b_save_traj_log=config.save_traj_log)

        return True


    def init_render(self):
        """
        Initialize the rendering setup for the scene: sets camera positions, 
        sets up the camera with a bounding box, and defines the initial view of the scene.
        """

        self.camera_positions = []
        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.building_scene_window.setup_camera(60, bbox, [0, 0, 0])
        self.building_scene_window.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])


    def update_camera_trajectory(self):
        """
        Update the camera trajectory visualization.

        Returns:
            o3d.geometry.LineSet or None: Line set representing the camera trajectory.
        """
        
        if len(self.camera_positions) > 1:
            points = np.array(self.camera_positions)
            lines = [[i, i+1] for i in range(len(points)-1)]
            colors = [[1, 0, 0] for _ in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            return line_set
        else:
            return None
    

    def update_render(self, pcd, frustum):
        """
        Update the rendered scene with a new point cloud and frustum.

        Args:
            pcd (open3d.t.geometry.PointCloud): New point cloud data.
            frustum (open3d.t.geometry.TriangleMesh): Frustum geometry.
        """
        
        if self.is_scene_updated:
            if pcd is not None and pcd.point.positions.shape[0] > 0:
                self.building_scene_window.scene.scene.update_geometry('points', pcd, 
                        rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG)
        self.building_scene_window.scene.remove_geometry("frustum")
        mat = rendering.MaterialRecord()
        mat.shader, mat.line_width = "unlitLine", 5.0
        self.building_scene_window.scene.add_geometry("frustum", frustum, mat)


    def update_main(self):
        """
        Update the main process loop. 
        Continuously updates the scene based on incoming frames and detected objects.
        """
        
        # Get camera intrinsic parameters from the next frame
        _, _, CX_RGB, CY_RGB, FX_RGB, FY_RGB = get_next_frame(self.pipeline, self.align_to_color, True)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(int(config.img_width), 
                                                      int(config.img_height), 
                                                      FX_RGB, FY_RGB, 
                                                      CX_RGB, CY_RGB)

        if config.engine == 'tensor':
            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix,o3d.core.Dtype.Float64)
        
        # Wait until a detection is made
        if config.do_semantic == True:
            b_detections = False
            while b_detections == False:
                color_image, depth_image = get_next_frame(self.pipeline, self.align_to_color)
                b_detections, _, color_image = perceive_scene(config.seg_device, seg_model, class_names, 
                                                            color_image, depth_image, colors_255, 
                                                            config.recolorize, config.use_kde)
        elif config.do_semantic == False:
            color_image, depth_image = get_next_frame(self.pipeline, self.align_to_color)

        # Initialize camera reference and input frames
        device = o3d.core.Device(config.device)
        color_ref, depth_ref = o3d.t.geometry.Image(color_image), o3d.t.geometry.Image(depth_image)
        input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns, intrinsic, device)
        raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns, intrinsic, device)
        
        # Set data for input and raycast frames
        input_frame.set_data_from_image('depth', depth_ref)
        input_frame.set_data_from_image('color', color_ref)
        raycast_frame.set_data_from_image('depth', depth_ref)
        raycast_frame.set_data_from_image('color', color_ref)

        gui.Application.instance.post_to_main_thread(self.window, lambda: self.init_render())

        pcd = None
        self.poses = []
        self.idx, frames_skipped = 0, 0
        prev_obj_ids = np.array([])
        trans_mat = o3c.Tensor(np.identity(4))

        while not self.is_done:

            if not self.is_started or not self.is_running:
                continue
            
            # Get the next frame and perform object detection
            color_image, depth_image = get_next_frame(self.pipeline, self.align_to_color)
            if config.do_semantic == True:
                b_detections, obj_ids, color_image = perceive_scene(config.seg_device, seg_model, class_names, 
                                                                    color_image, depth_image, colors_255, 
                                                                    config.recolorize, config.use_kde)
            
                # Check for new objects detected or frames to be skipped
                b_new_element = True if np.setdiff1d(obj_ids, prev_obj_ids).size > 0 else False
                prev_obj_ids = obj_ids

                if b_detections == False:
                    continue            # No objects detected, skip this
                elif b_detections == True and b_new_element == False and frames_skipped < config.wait_frames:
                    frames_skipped += 1
                    continue            # No new objects detected, wait for a bit, skip this
                elif b_detections == True and b_new_element == False and frames_skipped > config.wait_frames:
                    frames_skipped = 0  # No new objects detected, but for a while ago, update once
                else:
                    frames_skipped = 0  # New objects detected, update

            # Set data for input frame
            color, depth = o3d.t.geometry.Image(color_image), o3d.t.geometry.Image(depth_image)
            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color)

            # Track frame to model, integrate, and synthesize model frame
            if self.idx > 0:
                result = self.model.track_frame_to_model(input_frame, raycast_frame, 
                                                         float(config.depth_scale), 
                                                         config.depth_max)
                trans_mat = trans_mat @ result.transformation

            self.poses.append(trans_mat.cpu().numpy())
            self.model.update_frame_pose(self.idx, trans_mat)
            self.model.integrate(input_frame, float(config.depth_scale), 
                                 config.depth_max, config.trunc_voxel_multiplier)
            self.model.synthesize_model_frame(raycast_frame, float(config.depth_scale), 
                                              config.depth_min, config.depth_max, 
                                              config.trunc_voxel_multiplier, config.raycast_box)

            # Update the scene every 10 frames if mesh surface updating is enabled
            if self.idx % 10 == 0 and config.update_mesh_surface:
                pcd = self.model.voxel_grid.extract_point_cloud(config.point_weight_thr, 
                                config.est_point_count).to(o3d.core.Device(config.device))
                self.is_scene_updated = True
            else:
                self.is_scene_updated = False

            frustum = o3d.geometry.LineSet.create_camera_visualization(color.columns, 
                                                                       color.rows, 
                                                                       intrinsic.numpy(), 
                                                                       np.linalg.inv(trans_mat.cpu().numpy()), 
                                                                       0.2)
            frustum.paint_uniform_color([0.961, 0.475, 0.000])

            # Draw camera trajectory per k frames
            if self.idx % config.trajectory_interval == 0:
                self.camera_positions.append(trans_mat[:3, 3].cpu().numpy())
                self.building_scene_window.scene.remove_geometry("movement_lines")
                movement_lines = self.update_camera_trajectory()
                if movement_lines is not None:
                    mat = rendering.MaterialRecord()
                    mat.shader, mat.line_width = "unlitLine", 4.0
                    self.building_scene_window.scene.add_geometry("movement_lines", movement_lines, mat)

            gui.Application.instance.post_to_main_thread(self.window, lambda: self.update_render(pcd, frustum))

            self.idx += 1
            self.is_done = self.is_done
            if self.idx > int(config.skip_frames):
                print(f">> Processed {(self.idx - int(config.skip_frames)):d} frames.")

        time.sleep(1.0)


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add('--config', is_config_file=True)
    config = parser.get_config()
    print(config)

    # Load segmentation model and assign color scheme
    seg_model = YOLO(config.seg_model)
    class_names = seg_model.model.names
    sns_colors = sns.color_palette("dark", len(class_names))
    colors_255 = [sns_color_to_255_scale(sns_color) for sns_color in sns_colors]
    # colors_255 = colors_255[:len(class_names)]

    app = gui.Application.instance
    app.initialize()
    w = RGBD_SLAM(config)
    app.run()

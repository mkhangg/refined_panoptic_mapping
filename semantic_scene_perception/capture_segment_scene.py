#! usr/bin/python3

# Import neccessary libraries
import cv2
import math
import time
import copy
import psutil
import random
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from utils.utils import *
from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.utils import colorstr

# Option selections
index, NO_CAPS = 0, 8
# MODE = "view" 
MODE = "deploy"
PROCESS_NAME = "python"

TARGET_OBJECT = "person"      
CAPTURE_FOLDER = "cap_data"
YOLO_MODEL = "models/yolov8x-seg.pt"

# Init application
state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
# decimate = rs.decimation_filter()
# decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
# hole_filling = rs.hole_filling_filter(0)
colorizer = rs.colorizer()

def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)

if MODE == "view":
    cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(state.WIN_NAME, w, h)
    cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


out = np.empty((h, w, 3), dtype=np.uint8)

while True:
    if not state.paused:
        # if index == NO_CAPS:
        #     for proc in psutil.process_iter():
        #         if proc.name() == PROCESS_NAME:
        #             proc.kill()

        #     # Break this script
        #     break

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Get depth frame and color frame from pipeline
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # depth_frame = decimate.process(depth_frame)
        # depth_frame = hole_filling.process(depth_frame)

        # Align frames
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        aligned_depth_frame = frames.get_depth_frame()

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_intrin = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        color_intrin = rs.video_stream_profile(color_frame.profile).get_intrinsics()
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        CX_DEPTH, CY_DEPTH = depth_intrin.ppx, depth_intrin.ppy
        FX_DEPTH, FY_DEPTH = depth_intrin.fx, depth_intrin.fy
        CX_RGB, CY_RGB = color_intrin.ppx, color_intrin.ppy
        FX_RGB, FY_RGB = color_intrin.fx, color_intrin.fy
        R, T = depth_to_color_extrin.rotation, depth_to_color_extrin.translation
        R, T = np.transpose(np.array(R).reshape(3, 3)), np.array(T).reshape(3, )

        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_dim, color_colormap_dim = depth_colormap.shape, color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        aligned_depth_frame_view = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())    
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, aligned_depth_frame_view))
        else:
            images = np.hstack((color_image, aligned_depth_frame_view))

        if MODE == "view":
            cv2.imshow('RGB and Depth Images', images)

        if index == NO_CAPS - 1:

            # Process segmentation using YOLO
            model = YOLO(YOLO_MODEL)
            color_image_RGB_format = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            results = model.predict(source=color_image_RGB_format, save=True, save_txt=False, save_conf=True, save_crop=False, show_labels=True,
                                    show_conf=True, retina_masks=True, boxes=False, device='cpu', project=CAPTURE_FOLDER, name='exp')

            if len(results) == 0: # If no detection, then break
                break
            
            names, boxes, masks = model.model.names, results[0].boxes, results[0].masks.data.numpy().transpose(1, 2, 0)

            pred_classes = []
            for i in range(len(boxes)):
                pred_classes.append(int(boxes[i].cls.item()))
            
            pred_names = []
            for pred_class in pred_classes:
                pred_names.append(names[int(pred_class)])

            target_obj_ids = []
            for target_object in pred_names:
                target_obj_id = list(names.keys())[list(names.values()).index(target_object)]
                target_obj_ids.append(target_obj_id)
            
            # Manually create the point cloud from RGD and depth images
            start_time_process = time.time()
            
            target_masks = []
            traced_pred_classes = copy.deepcopy(pred_classes)
            for target_obj_id in target_obj_ids:
                target_mask = masks[:, :, traced_pred_classes.index(target_obj_id)]
                target_masks.append(target_mask)
                traced_pred_classes[traced_pred_classes.index(target_obj_id)] = -1      # Mark as done
            print("len(target_masks) = ", len(target_masks))
            
            depth_data = np.asanyarray(aligned_depth_frame.get_data())

            exp_id = "" if count_subfolders(CAPTURE_FOLDER) == 1 else count_subfolders(CAPTURE_FOLDER)
            np.savetxt(f'./{CAPTURE_FOLDER}/exp{exp_id}/depth_data.txt', depth_data, delimiter=',', fmt='%d')

            masked_pcs, mask_images = [], []
            for target_mask_i in range(len(target_masks)):
                masked_pc, mask_image = create_masked_pc(copy.deepcopy(depth_data), target_masks[target_mask_i], 
                                                        CX_RGB, CY_RGB, FX_RGB, FY_RGB,
                                                        CX_DEPTH, CY_DEPTH, FX_DEPTH, FY_DEPTH,
                                                        R, T, False)
                masked_pc = remove_outliers(masked_pc)

                masked_pcs.append(masked_pc)
                mask_images.append(mask_image)

                rectified_depth_image = cv2.bitwise_and(aligned_depth_frame_view.astype('uint8'), 
                                                        mask_image.astype('uint8'))

                np.savetxt(f"./{CAPTURE_FOLDER}/exp{exp_id}/mask_{target_mask_i}.txt", 
                           np.asanyarray(target_masks[target_mask_i]), delimiter=',', fmt='%d')
                cv2.imwrite(f"./{CAPTURE_FOLDER}/exp{exp_id}/binary_mask_image_{target_mask_i}.jpg", mask_image)
                cv2.imwrite(f"./{CAPTURE_FOLDER}/exp{exp_id}/rectified_depth_image_{target_mask_i}.jpg", rectified_depth_image)
                o3d.io.write_point_cloud(filename=f"{CAPTURE_FOLDER}/exp{exp_id}/masked_point_cloud_{target_mask_i}.pcd", 
                                         pointcloud=masked_pc, write_ascii=True)
            
            masked_depth_image = aggregate_mask_for_depth_image(aligned_depth_frame_view, mask_images)
            masked_rgb_image = aggregate_mask_for_rgb_image(color_image, mask_images)
            print("len(masked_pcs) = ", len(masked_pcs))
            
            initial_cloud = masked_pcs[0]
            initial_cloud.paint_uniform_color([round(random.uniform(0, 1), 1), \
                                               round(random.uniform(0, 1), 1), \
                                               round(random.uniform(0, 1), 1)])
            scene_pcd = initial_cloud
            
            for i in range(1, len(masked_pcs)):
                object_cloud = masked_pcs[i]
                object_cloud.paint_uniform_color([round(random.uniform(0, 1), 1), \
                                                  round(random.uniform(0, 1), 1), \
                                                  round(random.uniform(0, 1), 1)])
                scene_pcd += object_cloud


            scene_pcd = remove_outliers_in_scene(scene_pcd, Z_THRESHOLD_MIN=0, Z_THRESHOLD_MAX=50000)
            o3d.io.write_point_cloud(filename=f"{CAPTURE_FOLDER}/exp{exp_id}/segmented_scene.pcd", 
                                     pointcloud=scene_pcd, write_ascii=True)
            
            end_time_process = time.time()
            print(f">> Masked pointcloud process time: {1000*(end_time_process - start_time_process):.4f} ms.")

            # Save mask image and depth image of the target mask and masked point cloud 
            saved_results_path = f"{CAPTURE_FOLDER}/exp{exp_id}/"
            cv2.imwrite(f"./{CAPTURE_FOLDER}/exp{exp_id}/aligned_depth_image.jpg", aligned_depth_frame_view)
            cv2.imwrite(f"./{CAPTURE_FOLDER}/exp{exp_id}/masked_depth_image.jpg", masked_depth_image)
            cv2.imwrite(f"./{CAPTURE_FOLDER}/exp{exp_id}/masked_rgb_image.jpg", masked_rgb_image)
 
            # Print saving information
            if MODE == "view":
                print(f">> img0.jpg is saved to {colorstr('bold', saved_results_path)}.")
                print(f">> binary_mask_image.jpg is saved to {colorstr('bold', saved_results_path)}.")
                print(f">> aligned_depth_image.jpg is saved to {colorstr('bold', saved_results_path)}.")
                print(f">> rectified_depth_image.jpg is saved to {colorstr('bold', saved_results_path)}.")

            elif MODE == "deploy":
                print(f">> Result images and masked point cloud are saved to {colorstr('bold', saved_results_path)}.")

        index = index + 1

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    if MODE == "view":
        # Render
        now = time.time()
        out.fill(0)

        grid(out, (0, 0.5, 1), size=1, n=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(out, view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now
        cv2.setWindowTitle(state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
                            (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))
        cv2.imshow(state.WIN_NAME, out)
        key = cv2.waitKey(1)

# Stop streaming
pipeline.stop()

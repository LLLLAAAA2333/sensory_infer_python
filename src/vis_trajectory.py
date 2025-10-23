import os
import re
import cv2
import h5py
import torch
import shutil
import matplotlib
import numpy as np
from pathlib import Path


def save_to_hdf5(group, name, data):
    """
    Save data to HDF5 file according to data type, and record its original data type.
    :param group: h5py's group object.
    :param name: The name of the data.
    :param data: The data to be saved.
    """
    if data == None:
        data_type = "None"
    else:
        data_type = str(type(data))

    if isinstance(data, torch.Tensor):
        if data.dtype == torch.float16:
            data = data.to(torch.float32)
        dset = group.create_dataset(name, data = data.cpu().numpy())
    elif isinstance(data, np.ndarray):
        dset = group.create_dataset(name, data = data)
    elif isinstance(data, (str, int, float, complex)):
        dset = group.create_dataset(name, data = data)
    elif data is None:
        dset = group.create_dataset(name, shape = (0,))
    elif isinstance(data, dict):
        dset = group.create_group(name)
        for key, value in data.items():
            save_to_hdf5(dset, str(key), value)
    elif isinstance(data, (list, tuple)):
        try:
            np_array_data = np.array(data)
            dset = group.create_dataset(name, data = np_array_data)
        except (ValueError, TypeError):
            dset = group.create_group(name)
            for idx, item in enumerate(data):
                save_to_hdf5(dset, str(idx), item)
    else:
        raise TypeError("Unsupported data type")

    # Add the original data type as an attribute to the dataset
    dset.attrs['data_type'] = data_type


def alpha_blend_rgba_rgb(rgba_image, rgb_image):
    """
    Perform alpha blending of two images - one RGBA and one RGB.

    :param rgba_image: A numpy array of shape (H, W, 4) representing the RGBA image.
    :param rgb_image: A numpy array of shape (H, W, 3) representing the RGB image.
    :return: A numpy array of shape (H, W, 3) representing the blended RGB image.
    """
    if rgba_image.shape[:2] != rgb_image.shape[:2]:
        raise ValueError("The dimensions of the two images do not match")

    # Separate the RGB and Alpha channels of the RGBA image
    rgb1, alpha = rgba_image[..., :3], rgba_image[..., 3]
    # cv2.imshow('alpha', alpha)
    # cv2.waitKey(0)
    # Convert the Alpha channel to a floating point number and normalize
    alpha = alpha.astype(float) / 255

    # Calculate the output image
    out_rgb = (alpha[..., None] * rgb1 + (1 - alpha[..., None]) * rgb_image).astype(np.uint8)

    return out_rgb


def generate_colormap_colors(n, colormap_name = 'viridis'):
    """
    Generate a list of colors from a matplotlib colormap.

    :param n: The number of colors to generate.
    :param colormap_name: The name of the colormap to use.
    :return: A list of n colors.
    """
    # Get the specified colormap
    colormap = matplotlib.cm.get_cmap(colormap_name)
    # print(colormap)

    # Evenly distribute n points between 0 and 1
    points = np.linspace(0, 1, n)

    # Use these points to get colors from the colormap
    colors = [colormap(point) for point in points]

    # Convert the colors from RGBA to RGB format in the range 0-255
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]

    return colors


def convert_gray_to_rgb(gray_image,red_pseudocolor = False):
    """
    Convert a grayscale image to an RGB image.

    :param gray_image: A numpy array of shape (H, W) representing the grayscale image.
    :return: A numpy array of shape (H, W, 3) representing the RGB image.
    """
    # Copy the grayscale values to three color channels
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    if red_pseudocolor:
        # Set the blue and green channels to zero
        rgb_image[:, :, 0] = 0
        rgb_image[:, :, 1] = 0
    # rgba_image = np.stack((gray_image,)*3 + (np.full_like(gray_image, 255),), axis=-1)
    return rgb_image


def make_transparent_zero_overlay(image):
    """
    Create a transparent overlay image with the same size as the input image.

    :param image: A numpy array of shape (H, W, 3) representing the input image.
    :return: A numpy ZERO array of shape (H, W, 4) representing the transparent overlay image.
    """
    overlay = np.zeros_like(image, dtype = np.uint8)
    alpha_channel = np.full(overlay.shape[:2], 0, dtype = np.uint8)
    overlay = np.dstack((overlay, alpha_channel))
    return overlay


# def draw_transparent_rectangles(image, rectangles, colors, alpha = 0.5, thickness = 1):
#     """
#     Draw transparent rectangles on an image.

#     :param image: A numpy array of shape (H, W, 3) representing the input image.
#     :param rectangles: A numpy array of shape (N, 4) representing the N*[cx,cy,w,h] rectangles to draw.
#     :param colors: A list of N colors to use for the rectangles.
#     :param alpha: The alpha value to use for the rectangles.
#     :param thickness: The thickness of the rectangle edges.
#     :return: A numpy array of shape (H, W, 3) representing the image with the rectangles drawn on it.
#     """
#     overlay = make_transparent_zero_overlay(image)
#     for rect, color in zip(rectangles, colors):
#         # Parse the coordinates of the center point and width and height
#         cx, cy, w, h = rect
#         # Calculate the coordinates of the upper left and lower right corners of the rectangle
#         x1 = int(cx - w / 2)
#         y1 = int(cy - h / 2)
#         x2 = int(cx + w / 2)
#         y2 = int(cy + h / 2)

#         # Create a temporary image to draw opaque rectangles
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), [*color, int(alpha * 255)], thickness)  # -1 表示填充

#     image = alpha_blend_rgba_rgb(overlay, image)
#     return image

def draw_transparent_rectangles(image, rectangles, colors, alpha = 0.5, thickness = 1):
    """
    Draw transparent rectangles on an image.

    :param image: A numpy array of shape (H, W, 3) representing the input image.
    :param rectangles: A numpy array of shape (N, 4) representing the N*[cx,cy,w,h] rectangles to draw.
    :param colors: A list of N colors to use for the rectangles.
    :param alpha: The alpha value to use for the rectangles.
    :param thickness: The thickness of the rectangle edges.
    :return: A numpy array of shape (H, W, 3) representing the image with the rectangles drawn on it.
    """
    overlay = make_transparent_zero_overlay(image)
    for rect, color in zip(rectangles, colors):
        # Parse the coordinates of the center point and width and height
        cx, cy, w, h = rect
        # Calculate the coordinates of the upper left and lower right corners of the rectangle
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # Create a temporary image to draw opaque rectangles
        cv2.rectangle(overlay, (x1, y1), (x2, y2), [*color, int(alpha * 255)], thickness)  # -1 表示填充

    image = alpha_blend_rgba_rgb(overlay, image)
    return image

def draw_neuron_trace(image, traces, colors, n = 10, opacity_max = 0.5):
    """
    Draw a trace of the last n points on an image.

    :param image: A numpy array of shape (H, W, 3) representing the input image.
    :param traces: A numpy array of shape (N, T, 2) representing the N*T*[x,y] points.
    :param colors: A list of N colors to use for the traces.
    :param n: The number of points to draw.
    :param opacity_max: The maximum opacity to use for the traces.
    :return: A numpy array of shape (H, W, 3) representing the image with the trace drawn on it.
    """
    # print(traces)
    # Create a transparent layer of the same size as the original image
    overlay = make_transparent_zero_overlay(image)

    for points, color in zip(traces, colors):
        # print(points)
        trail_length = min(n, len(points))  # Ensure n does not exceed the total number of points
        color = [*color, 0]  # rgb to rgba

        # Iterate over the last n points to draw the trail
        for i in range(-trail_length, -1):
            opacity = (i + n + 1) / n * opacity_max  # Calculate opacity
            color[3] = int(opacity * 255)

            if all(points[i] >= 0) and all(points[i + 1] >= 0):  # Points not present in the current volume are set to -1 and not drawn
                cv2.line(overlay, tuple(points[i]), tuple(points[i + 1]), color, 2)

    return alpha_blend_rgba_rgb(overlay, image)


def split_points(points, split_number = 8):
    subsets = [[] for _ in range(split_number)]
    for i, _ in enumerate(points):
        subsets[i % split_number].append(i)

    return subsets


def split_points_by_id(neuron_ids, split_number = 8, recomm_ratio: float = 0.):
    """
    sort the neuron_ids and split them into split_number subsets.
    """
    if recomm_ratio:
        top_ids, left_ids = np.array_split(neuron_ids, [int(len(neuron_ids) * recomm_ratio)])
        subsets = np.array_split(np.sort(top_ids), split_number)
        subsets = subsets + np.array_split(np.sort(left_ids), len(left_ids) // len(subsets[0]) + 1)
    else:
        subsets = np.array_split(np.sort(neuron_ids), split_number)
    return subsets


class Plot3DResult:
    """
    This class can be used to draw MIP images in three directions, as well as 3D boxes and trajectories of neurons.
    Usage:
    1. Define a function F_img to load the image data of a volume.
        a. This function should accept one parameter: the current volume pointer. The pointer ranges from 0 to the number of volumes - 1, indicating the current volume, thereby loading the corresponding image data. Note that the images and neuron information pointed to by the same pointer should correspond.
        b. This function should return three [H, W, 3] numpy arrays, representing the MIP images in three directions (RGB format).
    2. Define a function F_info to load the neuron information of a volume.
        a. This function should accept one parameter: the current volume pointer. The pointer ranges from 0 to the number of volumes - 1, indicating the current volume, thereby loading the corresponding neuron information. Note that the images and neuron information pointed to by the same pointer should correspond.
        b. This function should return a tuple containing two elements, the first element is an [N*6] numpy array, containing the 3D boxes of neurons; the second element is a one-dimensional [N elements] numpy array, containing the predicted IDs of neurons.
    3. Create a Plot3DResult(F_img,F_info) object.
    4. Call the save_all method to store all tri-view subsets. Or call the show_all method to display the tri-view of all volumes.

    Precautions:
    1. This class will store the neuron information, MIP images, and tri-views with boxes and trajectories of the volumes that have already been displayed (only storing the subsets that have been displayed for that frame) in a dictionary manner, which will not be loaded and drawn repeatedly. Note that when the number of volumes is large, it may occupy a lot of memory.
    2. When the neuron information is loaded for the first time, the neurons will be divided into split_number subsets, and if neurons appear later that are not in these subsets, they will be added to the end of the subsets.
    3. The length of the first subset is equal to the length of colors, ensuring that each neuron has a corresponding color; the length of other subsets will not exceed the length of the first subset.

    Other features worth mentioning by AI:
    1. This class will automatically divide the neurons into split_number subsets and use colormap_name to draw the colors of neurons.
    2. This class will automatically draw rectangles and traces on each image.
    3. This class will automatically add a box (-1,-1,-1,0,0,0) at the end, ensuring the -1 index is not drawn.

    """

    def __init__(self, load_volume_image_func, load_volume_neuron_info_func, split_number = 1, colormap_name = 'rainbow', bbox_opacity = 0.5, bbox_thickness = 2, trace_opacity_max = 0.5, trace_length = 10,
                 **kwargs):
        """Initialize the Plot3DResult class.

        :param load_volume_image_func: A function (accepting one parameter: the current volume pointer) for loading the image data of a volume.
        :param load_volume_neuron_info_func: A function (accepting one parameter: the current volume pointer) for loading the neuron information of a volume.
        :param split_number: The number of neurons to be divided.
        :param colormap_name: The name of the color map used for drawing neurons.
        """
        # Parameters
        self.load_volume_image_func = load_volume_image_func
        self.load_volume_neuron_info_func = load_volume_neuron_info_func
        self.split_number = split_number
        self.colormap_name = colormap_name
        self.bbox_opacity = bbox_opacity
        self.bbox_thickness = bbox_thickness
        self.trace_opacity_max = trace_opacity_max
        self.trace_length = trace_length

        # Data
        self.neuron_id_subsets = None  # Store subsets of neuron IDs, elements as a list of one-dimensional np arrays
        self.colors = None
        self.neuron_3d_bboxes = {}
        self.neuron_pred_id = {}
        self.neuron_v_index_subsets = {}
        self.mip_z = {}
        self.mip_y = {}
        self.mip_x = {}
        self.neuron_trace = {-1: np.array([[-1, -1, -1]])}
        self.xy_view = {}
        self.xz_view = {}
        self.zy_view = {}

        self.all_ids = np.array(kwargs.get("all_ids", []))
        self.recomm_ratio = kwargs.get('recomm_ratio', 0.)

    def load_volume_neuron_info(self, cur_volume_ptr):
        """Load the neuron information of the specified volume.

        :param cur_volume_ptr: The pointer of the current volume.
        """
        self.neuron_3d_bboxes[cur_volume_ptr], self.neuron_pred_id[cur_volume_ptr] = self.load_volume_neuron_info_func(cur_volume_ptr)
        self.update_neuron_trace(cur_volume_ptr)

        # Sparse division of neurons
        if self.neuron_id_subsets is None or self.colors is None:
            self.init_neuron_id_subsets_and_color(cur_volume_ptr)
        else:
            self.add_new_neuron_id(cur_volume_ptr)

        # Add a box (-1,-1,-1,0,0,0) at the end, ensuring the -1 index is not drawn.
        if self.neuron_3d_bboxes[cur_volume_ptr].shape[0]:
            self.neuron_3d_bboxes[cur_volume_ptr] = np.vstack((self.neuron_3d_bboxes[cur_volume_ptr], np.array([-1, -1, -1, 0, 0, 0])))
        else:
            self.neuron_3d_bboxes[cur_volume_ptr] = np.array([[-1, -1, -1, 0, 0, 0]])

    def update_neuron_trace(self, cur_volume_ptr):
        """Update the neuron trace data.

        :param cur_volume_ptr: The pointer of the current volume.
        """
        # Add the neuron coordinates of the current volume to the trace
        for i in range(len(self.neuron_pred_id[cur_volume_ptr])):
            id = self.neuron_pred_id[cur_volume_ptr][i]
            if id not in self.neuron_trace:
                # Initialize neuron_trace[id] as a matrix of (cur_volume_ptr+1,3), all elements are -1
                self.neuron_trace[id] = np.full((cur_volume_ptr + 1, 3), -1)
                self.neuron_trace[id][cur_volume_ptr] = self.neuron_3d_bboxes[cur_volume_ptr][i][:3]
            else:
                # First, extend neuron_trace[id] to the size of cur_volume_ptr+1 (filling new elements with -1), then add the coordinates of the current volume to the corresponding position
                self.neuron_trace[id] = np.vstack((self.neuron_trace[id], np.full((cur_volume_ptr + 1 - len(self.neuron_trace[id]), 3), -1)))
                self.neuron_trace[id][cur_volume_ptr] = self.neuron_3d_bboxes[cur_volume_ptr][i][:3]
        # add ids that are not in the current volume
        for id in self.all_ids:
            if id not in self.neuron_trace:
                # Initialize neuron_trace[id] as a matrix of (cur_volume_ptr+1,3), all elements are -1
                self.neuron_trace[id] = np.full((cur_volume_ptr + 1, 3), -1)
        # Check if existing elements in the trace need to be extended
        for id in self.neuron_trace:
            if len(self.neuron_trace[id]) < cur_volume_ptr + 1:
                self.neuron_trace[id] = np.vstack((self.neuron_trace[id], np.full((cur_volume_ptr + 1 - len(self.neuron_trace[id]), 3), -1)))

    def init_neuron_id_subsets_and_color_by_all_ids(self):
        """Initialize subsets and colors of neurons.

        :param cur_volume_ptr: The pointer of the current volume.
        """

        self.colors = generate_colormap_colors(len(self.neuron_id_subsets[0]), self.colormap_name)

    def init_neuron_id_subsets_and_color(self, cur_volume_ptr):
        """Initialize subsets and colors of neurons.

        :param cur_volume_ptr: The pointer of the current volume.
        """
        if self.neuron_3d_bboxes[cur_volume_ptr].shape[0]:
            if len(self.all_ids):
                self.neuron_id_subsets = split_points_by_id(self.all_ids, self.split_number, recomm_ratio = self.recomm_ratio)
            else:
                # self.neuron_id_subsets = [self.neuron_pred_id[cur_volume_ptr][v_indexes] for v_indexes in split_points(self.neuron_3d_bboxes[cur_volume_ptr][:, 0:3], self.split_number)]
                self.neuron_id_subsets = [self.neuron_pred_id[cur_volume_ptr][v_indexes][self.neuron_pred_id[cur_volume_ptr][v_indexes] >= 0] for v_indexes in split_points(self.neuron_3d_bboxes[cur_volume_ptr][:, 0:3], self.split_number)]

            self.colors = generate_colormap_colors(len(self.neuron_id_subsets[0]), self.colormap_name)

    def add_new_neuron_id(self, cur_volume_ptr):
        """Filter and add newly appeared neuron IDs to the end of the neuron subsets.

        :param cur_volume_ptr: The pointer of the current volume.
        """
        # Filter out elements in self.neuron_pred_id[cur_volume_ptr] that do not appear in self.neuron_id_subsets
        unique_elements = np.setdiff1d(self.neuron_pred_id[cur_volume_ptr], np.concatenate(self.neuron_id_subsets))
        unique_elements = unique_elements[unique_elements >= 0]

        new_id_number = len(unique_elements)

        if new_id_number == 0:
            return

        # Add these elements to the last np array of self.neuron_id_subsets, ensuring it does not exceed the length of the first np array in self.neuron_id_subsets
        max_length = len(self.neuron_id_subsets[0])
        last_array_length = len(self.neuron_id_subsets[-1])

        # The maximum number that can be added
        available_space = max_length - last_array_length

        if available_space >= len(unique_elements):
            # If there is enough space, add directly to the last array
            self.neuron_id_subsets[-1] = np.append(self.neuron_id_subsets[-1], unique_elements)
        else:
            # If there is not enough space, add as many elements as possible to the existing array, and the rest to a new array
            if available_space > 0:
                self.neuron_id_subsets[-1] = np.append(self.neuron_id_subsets[-1], unique_elements[:available_space])
                unique_elements = unique_elements[available_space:]

            # Newly created arrays add the remaining elements
            while len(unique_elements) > 0:
                new_array_size = min(len(unique_elements), max_length)
                self.neuron_id_subsets.append(unique_elements[:new_array_size])
                unique_elements = unique_elements[new_array_size:]

        print(f"[volume_ptr: {cur_volume_ptr}] Add {new_id_number} new ids")

    def draw_views(self, cur_volume_ptr, cur_subset_index):
        """
        Read neuron information and image files, and draw a tri-view composed of mip, neuron boxes, and neuron trajectories overlaid.

        :param cur_volume_ptr: The pointer of the current volume.
        :param cur_subset_index: The index of the current subset.
        """
        # Read TIFF files and convert to numpy arrays
        if cur_volume_ptr not in self.mip_z or cur_volume_ptr not in self.mip_y or cur_volume_ptr not in self.mip_x:
            self.mip_z[cur_volume_ptr], self.mip_y[cur_volume_ptr], self.mip_x[cur_volume_ptr] = self.load_volume_image_func(cur_volume_ptr)

        # Load the neuron information for the current frame
        if cur_volume_ptr not in self.neuron_3d_bboxes or cur_volume_ptr not in self.neuron_pred_id:
            self.load_volume_neuron_info(cur_volume_ptr)

        if self.neuron_id_subsets is None or self.colors is None:
            self.xy_view[(cur_volume_ptr, 0)] = self.mip_z[cur_volume_ptr]
            self.xz_view[(cur_volume_ptr, 0)] = self.mip_y[cur_volume_ptr]
            self.zy_view[(cur_volume_ptr, 0)] = self.mip_x[cur_volume_ptr]
        else:
            # Convert the segmentation results from neuron id to neuron index in the current volume
            if cur_volume_ptr not in self.neuron_v_index_subsets:
                self.neuron_v_index_subsets[cur_volume_ptr] = np.full((len(self.neuron_id_subsets), len(self.neuron_id_subsets[0])), -1)
                transform_matrix = [np.where(np.equal(self.neuron_pred_id[cur_volume_ptr], neuron_id_subset[:, np.newaxis])) for neuron_id_subset in self.neuron_id_subsets]
                for i in range(len(self.neuron_id_subsets)):
                    self.neuron_v_index_subsets[cur_volume_ptr][i][transform_matrix[i][0]] = transform_matrix[i][1]
            elif self.neuron_v_index_subsets[cur_volume_ptr].shape[0] != len(self.neuron_id_subsets):
                self.neuron_v_index_subsets[cur_volume_ptr] = np.vstack((self.neuron_v_index_subsets[cur_volume_ptr],
                                                                         np.full((len(self.neuron_id_subsets) -
                                                                                  self.neuron_v_index_subsets[cur_volume_ptr].shape[0],
                                                                                  self.neuron_v_index_subsets[cur_volume_ptr].shape[1]), -1)))
            # print(self.neuron_v_index_subsets[cur_volume_ptr].shape)

            # Draw rectangles on each image
            if (cur_volume_ptr, cur_subset_index) not in self.xy_view or (cur_volume_ptr, cur_subset_index) not in self.xz_view or (cur_volume_ptr, cur_subset_index) not in self.zy_view:
                cur_neuron_v_index_subsets = self.neuron_v_index_subsets[cur_volume_ptr][cur_subset_index]
                self.xy_view[(cur_volume_ptr, cur_subset_index)] = draw_transparent_rectangles(self.mip_z[cur_volume_ptr],
                                                                                               np.hstack((self.neuron_3d_bboxes[cur_volume_ptr][cur_neuron_v_index_subsets, :][:, [0, 1]],
                                                                                                          self.neuron_3d_bboxes[cur_volume_ptr][cur_neuron_v_index_subsets, :][:, [3, 4]])),
                                                                                               self.colors, self.bbox_opacity, self.bbox_thickness)
                self.xz_view[(cur_volume_ptr, cur_subset_index)] = draw_transparent_rectangles(self.mip_y[cur_volume_ptr],
                                                                                               np.hstack((self.neuron_3d_bboxes[cur_volume_ptr][cur_neuron_v_index_subsets, :][:, [0, 2]],
                                                                                                          self.neuron_3d_bboxes[cur_volume_ptr][cur_neuron_v_index_subsets, :][:, [3, 5]])),
                                                                                               self.colors, self.bbox_opacity, self.bbox_thickness)
                self.zy_view[(cur_volume_ptr, cur_subset_index)] = draw_transparent_rectangles(self.mip_x[cur_volume_ptr],
                                                                                               np.hstack((self.neuron_3d_bboxes[cur_volume_ptr][cur_neuron_v_index_subsets, :][:, [2, 1]],
                                                                                                          self.neuron_3d_bboxes[cur_volume_ptr][cur_neuron_v_index_subsets, :][:, [4, 3]])),
                                                                                               self.colors, self.bbox_opacity, self.bbox_thickness)

                # Draw neuron traces on each image
                self.xy_view[(cur_volume_ptr, cur_subset_index)] = draw_neuron_trace(self.xy_view[(cur_volume_ptr, cur_subset_index)],
                                                                                     [self.neuron_trace[id][:cur_volume_ptr + 1, [0, 1]] for id in self.neuron_id_subsets[cur_subset_index]],
                                                                                     self.colors, self.trace_length, self.trace_opacity_max)
                self.xz_view[(cur_volume_ptr, cur_subset_index)] = draw_neuron_trace(self.xz_view[(cur_volume_ptr, cur_subset_index)],
                                                                                     [self.neuron_trace[id][:cur_volume_ptr + 1, [0, 2]] for id in self.neuron_id_subsets[cur_subset_index]],
                                                                                     self.colors, self.trace_length, self.trace_opacity_max)
                self.zy_view[(cur_volume_ptr, cur_subset_index)] = draw_neuron_trace(self.zy_view[(cur_volume_ptr, cur_subset_index)],
                                                                                     [self.neuron_trace[id][:cur_volume_ptr + 1, [2, 1]] for id in self.neuron_id_subsets[cur_subset_index]],
                                                                                     self.colors, self.trace_length, self.trace_opacity_max)
        # If you need to adjust the size or ratio of the image, please complete and enable the code here
        # resize_x, resize_y ,_= self.xy_view[(cur_volume_ptr, cur_subset_index)].shape
        # resize_z = self.xz_view[(cur_volume_ptr, cur_subset_index)].shape[0]
        # self.xy_view[(cur_volume_ptr, cur_subset_index)] = cv2.resize(self.xy_view[(cur_volume_ptr, cur_subset_index)], (resize_y, resize_x))
        # self.xz_view[(cur_volume_ptr, cur_subset_index)] = cv2.resize(self.xz_view[(cur_volume_ptr, cur_subset_index)], (resize_y, resize_z))
        # self.zy_view[(cur_volume_ptr, cur_subset_index)] = cv2.resize(self.zy_view[(cur_volume_ptr, cur_subset_index)], (resize_z, resize_x))

    def save_views(self, file_path, number_of_volumes, cur_subset_index, fps: float = 5):
        """
        Join the three views together and store them in sequence as a video.

        :param file_path: The path of the video file.
        :param number_of_volumes: The number of volumes.
        :param cur_subset_index: The index of the current subset.
        """
        # Check if the directory exists, if not, create it
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # Create a video encoder, encoded in MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.draw_views(0, cur_subset_index)
        # Create an empty video file
        out = cv2.VideoWriter(file_path, cv2.CAP_OPENCV_MJPEG, fourcc, fps,
                              (self.xy_view[(0, cur_subset_index)].shape[1] + self.zy_view[(0, cur_subset_index)].shape[1],
                               self.xy_view[(0, cur_subset_index)].shape[0] + self.xz_view[(0, cur_subset_index)].shape[0]))

        for i in range(number_of_volumes):
            self.draw_views(i, cur_subset_index)

            xy_view = self.xy_view[(i, cur_subset_index)]
            xz_view = self.xz_view[(i, cur_subset_index)]
            zy_view = self.zy_view[(i, cur_subset_index)]
            # Concatenate the three views, where xy_view and xz_view have the same number of columns, and xy_view and zy_view have the same number of rows
            # First, create an extended version of zy_view that can be concatenated with xy_view horizontally
            # This requires the vertical concatenation result of zy_view with xy_view to match the row count of xz_view
            # Since zy_view originally is 1024x85, we need to create a new padding area to match the total height of xy_view plus xz_view

            # Calculate the required padding height
            total_height = xy_view.shape[0] + xz_view.shape[0]  # Height of xy_view + height of xz_view
            padding_height = total_height - zy_view.shape[0]  # The padding height needed

            # Create a padding area to be vertically concatenated with zy_view_rgb, considering the color channels
            padding_rgb = np.zeros((padding_height, zy_view.shape[1], 3), dtype = np.uint8)  # Padding area, size is the required padding height x 85 x 3
            # cv2.imshow('padding_rgb', padding_rgb)

            # Create an extended zy_view_rgb
            extended_zy_view_rgb = np.vstack((padding_rgb, zy_view))

            # Concatenate xy_view_rgb and xz_view_rgb vertically
            vertical_combined_rgb = np.vstack((xz_view, xy_view))
            # cv2.imshow('vertical_combined_rgb', vertical_combined_rgb)

            # Horizontally concatenate extended_zy_view_rgb with vertical_combined_rgb
            final_combined_view_rgb = np.hstack((vertical_combined_rgb, extended_zy_view_rgb))

            out.write(final_combined_view_rgb)

            # Display progress bar
            print(f"\rSaving video in subset {cur_subset_index:02d}... {i + 1:02d}/{number_of_volumes:02d}", end = "")

            # cv2.imshow('final_combined_view_rgb', final_combined_view_rgb)
            # cv2.waitKey(1)

        out.release()

    def save_all(self, file_dir, number_of_volumes, fps: float = 5):
        """
        Store all tri-view subsets.

        :param file_dir: The directory where the video files are located.
        :param number_of_volumes: The number of volumes.
        """
        print("\nSaving videos...")
        # First, draw the tri-view of subset 0 of all volumes, loading all information in the process to determine the final number of subsets
        for i in range(number_of_volumes):
            self.draw_views(i, 0)
            print(f"\rDetermining subsets count... {i + 1:02d}/{number_of_volumes:02d}", end = "")
        subset_count = len(self.neuron_id_subsets) if self.neuron_id_subsets is not None else 1
        print(f"\r{subset_count} subsets are found." + " " * 30)
        # Then, store the tri-view of all subsets
        for cur_subset_index in range(subset_count):
            self.save_views(os.path.join(file_dir, "neuron_trace", f"subset_{cur_subset_index:02d}.avi"), number_of_volumes, cur_subset_index, fps = fps)
        print("\rVideos are saved in " + os.path.join(file_dir, "neuron_trace") + "." + " " * 30)

    def show_all(self, number_of_volumes):
        """
        Show the tri-view of all volumes.

        :param number_of_volumes: The number of volumes.
        """
        cur_subset_index = 0
        cur_volume_ptr = 0
        while (1):
            # Draw the tri-view
            self.draw_views(cur_volume_ptr, cur_subset_index)

            # Display the tri-view
            cv2.imshow('xy view', self.xy_view[(cur_volume_ptr, cur_subset_index)])
            cv2.imshow('xz view', self.xz_view[(cur_volume_ptr, cur_subset_index)])
            cv2.imshow('zy view', self.zy_view[(cur_volume_ptr, cur_subset_index)])

            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:  # 27 == Esc
                break
            elif key == ord('w'):
                cur_subset_index = cur_subset_index + 1 if cur_subset_index + 1 < len(self.neuron_id_subsets) else cur_subset_index
            elif key == ord('s'):
                cur_subset_index = cur_subset_index - 1 if cur_subset_index - 1 >= 0 else 0
            elif key == ord('a'):
                cur_volume_ptr = cur_volume_ptr - 1 if cur_volume_ptr - 1 >= 0 else 0
            elif key == ord('d'):
                cur_volume_ptr = cur_volume_ptr + 1 if cur_volume_ptr + 1 < number_of_volumes else cur_volume_ptr
            elif key == ord('m'):
                self.save_all("C:/Experiment/", number_of_volumes)

        cv2.destroyAllWindows()


def collect_volumes(name, volume_number_list, pattern = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[w|fly]\d+_?Dt\d{6}_(\d+)$"):
    """
    Collect volume numbers from an HDF5 file.

    :param name: The name of the data group.
    :param volume_number_list: A list to store volume numbers.
    """
    # Check if the item is a data group we're interested in
    match = re.match(pattern, name)
    if match:
        name = match.group()
        if name not in volume_number_list:
            volume_number_list.append(name)


def get_volume_numbers_in_h5(h5_path, collect_volume_func):
    """
    Get all volume numbers in an HDF5 file.

    :param h5_path: The path to the HDF5 file.
    :param collect_volume_func: A function with two parameters (name, volume_number_list), used to extract volume numbers from the name and add them to volume_numbers_in_h5.
    :return: A list containing all volume numbers in the HDF5 file.
    """
    volume_numbers_in_h5 = []
    with h5py.File(h5_path, 'r') as file:
        # Use the visit method to traverse data groups
        file.visit(lambda name: collect_volume_func(name, volume_numbers_in_h5))
    return volume_numbers_in_h5


def load_volume_image_from_h5(h5_path, vol_name, x_ratio = 1., y_ratio = 1., z_ratio = 1., source_min = None, source_max = None, red_pseudo_color = False):
    with h5py.File(h5_path, 'r') as file:
        volume = file[f"{vol_name}/volume"][:, 0]
    return get_mip_from_uint8_gray_volume(volume, x_ratio, y_ratio, z_ratio, source_min, source_max,red_pseudo_color)


def load_volume_neuron_info_from_h5(h5_path, vol_name):
    with h5py.File(h5_path, 'r') as file:
        name = vol_name
        neuron_3d_bbox = file[name + "/neuron_pt_tuple"][:, :6]  # Read the 3D neuron boxes [cx, cy, cz, w, h, d]

        neuron_pred_id = file[name + "/neuron_pred_ids"][:]

    return neuron_3d_bbox, neuron_pred_id


def load_idv_outcomes(h5_path):
    with h5py.File(h5_path, 'r') as file:
        id_map = file["Outcomes/id_map"][:]
    id_map: dict = {k: v for k, v in id_map}
    return id_map


def load_volume_neuron_info_from_h5_with_outcomes(h5_path, vol_name, **kwargs):
    neuron_3d_bbox, neuron_pred_id = load_volume_neuron_info_from_h5(h5_path, vol_name)
    id_map = kwargs['id_map']
    neuron_pred_id = np.array([id_map[p] if p >= 0 else p for p in neuron_pred_id])
    return neuron_3d_bbox, neuron_pred_id


def rescale_image(image, target_min, target_max, source_min = None, source_max = None):
    """
    Rescale the values in an image to a new specified range.This function includes
    checks to ensure that the target and source ranges are valid.
    Parameters:
    image (numpy.ndarray): The input image array with pixel values.
    target_min : The minimum value of the target range.
    target_max : The maximum value of the target range.
    source_min (optional): The minimum value of the image's original range.
                           If None, it is automatically computed from the image.
    source_max (optional): The maximum value of the image's original range.
                           If None, it is automatically computed from the image.
    Returns:
    numpy.ndarray: The rescaled image array where the original image values have been
                   scaled to fit within the new target range, while ensuring that all
                   values lie within this range using clipping.
    Raises:
    ValueError: If the target or source ranges are invalid (i.e., min is not less than max).
    """
    # Check that target_min is less than target_max
    if target_min >= target_max:
        raise ValueError("target_min must be less than target_max")
    # If source_min or source_max are not provided, compute them from the image
    if source_min is None:
        source_min = np.min(image)
    if source_max is None:
        source_max = np.max(image)
    image_float64 = image.astype(np.float64)
    # Check that source_min is less than source_max
    if source_min >= source_max:
        raise ValueError("source_min must be less than source_max")
    # Compute the rescaled image with values adjusted to the new range and clip to ensure
    # values stay within target_min and target_max
    rescaled_image = np.clip((image_float64 - source_min) / (source_max - source_min) * (target_max - target_min) + target_min,
                             target_min, target_max).astype(image.dtype)
    return rescaled_image


def get_mip_from_uint8_gray_volume(volume, x_ratio, y_ratio, z_ratio, source_min = None, source_max = None, red_pseudo_color = False):
    """
    Calculate MIP in three directions and convert it into a three-channel RGB image.

    :param volume: A numpy array containing volume data.
    :param z_ratio: The scaling ratio for the Z axis.
    :param source_min (optional): The minimum value of the image's original range.
                                  If None, it is automatically computed from the image.
    :param source_max (optional): The maximum value of the image's original range.
                                  If None, it is automatically computed from the image.
    :return: Three MIP images.
    """
    # Calculate MIP
    mip_z = rescale_image(np.max(volume, axis = 0), 0, 255, source_min, source_max).astype(np.uint8)  # Along the Z axis (axis 0)
    mip_y = rescale_image(np.max(volume, axis = 1), 0, 255, source_min, source_max).astype(np.uint8)  # Along the Y axis (axis 1)
    mip_x = rescale_image(np.max(volume, axis = 2), 0, 255, source_min, source_max).astype(np.uint8)  # Along the X axis (axis 2)
    # mip_z = cv2.normalize(np.max(volume, axis=0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Along the Z axis (axis 0)
    # mip_y = cv2.normalize(np.max(volume, axis=1), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Along the Y axis (axis 1)
    # mip_x = cv2.normalize(np.max(volume, axis=2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Along the X axis (axis 2)
    # The yz view (MIP X) needs to be transposed
    mip_x = mip_x.T

    # Dimensions in three directions
    size_x, size_y = mip_z.shape
    size_x = int(size_x * x_ratio)
    size_y = int(size_y * y_ratio)
    size_z = int(mip_y.shape[0] * z_ratio)

    # Resize the images to fit the view
    mip_z = cv2.resize(mip_z, (size_y, size_x))
    mip_y = cv2.resize(mip_y, (size_y, size_z))
    mip_x = cv2.resize(mip_x, (size_z, size_x))

    # Convert to three channels
    mip_z = convert_gray_to_rgb(mip_z,red_pseudo_color)
    mip_y = convert_gray_to_rgb(mip_y,red_pseudo_color)
    mip_x = convert_gray_to_rgb(mip_x,red_pseudo_color)

    return mip_z, mip_y, mip_x


def save_h5file(path, volume_name, **kwargs):
    with h5py.File(path, 'a') as file:
        for key, value in kwargs.items():
            save_to_hdf5(file, f"{volume_name}/{key}", value)


def recover_cxcy_on_pt_tuple(pt_tuple, rot, mean):
    rot_inv = torch.inverse(rot)
    recovered_pt_tuple = pt_tuple.clone()
    recovered_pt_tuple[:, :2] = torch.mm(recovered_pt_tuple[:, :2], rot_inv.to(pt_tuple.dtype))
    recovered_pt_tuple[:, 0] += mean[0]
    recovered_pt_tuple[:, 1] += mean[1]
    return recovered_pt_tuple

def draw_python_result(h5_path,z_ratio,vps):
    tri_view_name_reg = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[w|fly]\d+_?Dt\d{6}_\d{6}"
    id_map = load_idv_outcomes(h5_path)
    vol_names = get_volume_numbers_in_h5(h5_path, lambda n, v: collect_volumes(n, v, tri_view_name_reg))

    Plot3DResult(lambda ptr: load_volume_image_from_h5(h5_path, vol_names[ptr], z_ratio = z_ratio),
                    lambda ptr: load_volume_neuron_info_from_h5_with_outcomes(h5_path, vol_names[ptr], id_map = id_map),
                    all_ids = list(id_map.values()), recomm_ratio = 0.3, split_number = 1,
                    # ).show_all(len(vol_names))
                    ).save_all(Path(h5_path).parent, len(vol_names), vps)
    
def draw_CTest_result(h5_path_image,h5_path_neuron_info,z_ratio,vps):
    # tri_view_name_reg_image = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[w|fly]\d+_?Dt\d{6}_\d{6}"
    tri_view_name_reg_image = r"real-time_analysis/volume_\d{8}"
    tri_view_name_reg_info = r"real-time_analysis/volume_\d{8}"

    id_map = load_idv_outcomes(h5_path_neuron_info)
    vol_names_image = get_volume_numbers_in_h5(h5_path_image, lambda n, v: collect_volumes(n, v, tri_view_name_reg_image))
    vol_names_info = get_volume_numbers_in_h5(h5_path_neuron_info, lambda n, v: collect_volumes(n, v, tri_view_name_reg_info))

    Plot3DResult(lambda ptr: load_volume_image_from_h5(h5_path_image, vol_names_image[ptr], z_ratio = z_ratio,source_min = 200, source_max = 800,red_pseudo_color = False),
                    lambda ptr: load_volume_neuron_info_from_h5_with_outcomes(h5_path_neuron_info, vol_names_info[ptr], id_map = id_map),
                    all_ids = list(id_map.values()), recomm_ratio = 0.5, split_number = 1,
                    # ).show_all(len(vol_names_image))
                    ).save_all(Path(h5_path_neuron_info).parent, len(vol_names_image), vps)

if __name__ == '__main__':
    # draw_python_result("./data/1.11/neuron_traces.h5", 1.5 / 0.3, 5)
    draw_python_result("./data/1.11/neuron_traces_w10_by_yuxiang.h5", 1.5 / 0.3, 5)
    # draw_CTest_result(r"./data/1.11/ImgStk001_dk002_w4_Dt230525_{AF}_{red-1921-to-2920}_{semi-immoblized}.h5",r"./data/1.11/output_2024-05-18_16-01-18_ImgStk001_dk002_w4_Dt230525_{AF}_{red-1921-to-2920}_{semi-immoblized}.h5", 1.5 / 0.3, 20)
    # draw_CTest_result(r"./data/1.11/ImgStk001_dk001_w10_Dt230525_{AF}_{red-221-to-4220}.h5",r"./data/1.11/output_2024-05-31_14-41-18_ImgStk001_dk001_w10_Dt230525_{AF}_{red-221-to-4220}.h5", 1.5 / 0.3, 5)
    # h5_path = Path(os.path.join("/home/cbmi/CBMI_python/data/zone/data_20230525_w10_test_ia/neuron_traces.h5"))
    # # raw_img_root = h5_path.parent
    # # if os.path.exists(raw_img_root / "neuron_trace"):
    # #     shutil.rmtree(raw_img_root / "neuron_trace")
    # #
    # # z_ratio = 1.5 / 0.3
    # #
    # # vol_names = get_volume_numbers_in_h5(h5_path, lambda name, volume_number_list: collect_volumes(name, volume_number_list))
    # #
    # # id_map = load_idv_outcomes(h5_path)
    # #
    # # viewer = Plot3DResult(lambda cur_volume_ptr: load_volume_image_from_h5(h5_path, vol_names[cur_volume_ptr], z_ratio = z_ratio),
    # #                       lambda cur_volume_ptr: load_volume_neuron_info_from_h5_with_outcomes(h5_path, vol_names[cur_volume_ptr], id_map = load_idv_outcomes(h5_path)),
    # #                       # lambda cur_volume_ptr: (np.array([]), np.array([])),
    # #                       all_ids = list(id_map.values()),
    # #                       split_number = 3,
    # #                       recomm_ratio = 1.0,
    # #                       )
    # # viewer.save_all(raw_img_root, len(vol_names), fps = 2.5)

    # # h5_path = Path(os.path.join("/home/cbmi/CBMI_python/data/zone/red_whole_green_sparse/0402_cyofp_nemo_backup/neuron_traces.h5"))
    # raw_img_root = h5_path.parent
    # if os.path.exists(raw_img_root / "neuron_trace"):
    #     shutil.rmtree(raw_img_root / "neuron_trace")

    # z_ratio = 1.5 / 0.3

    # vol_names = get_volume_numbers_in_h5(h5_path, lambda name, volume_number_list: collect_volumes(name, volume_number_list, pattern = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[w|fly]\d+_?Dt\d{6}_\d{6}"))

    # # id_map = load_idv_outcomes(h5_path)

    # viewer = Plot3DResult(lambda cur_volume_ptr: load_volume_image_from_h5(h5_path, vol_names[cur_volume_ptr], z_ratio = z_ratio, source_min = 200, source_max = 800),
    #                       # lambda cur_volume_ptr: load_volume_neuron_info_from_h5_with_outcomes(h5_path, vol_names[cur_volume_ptr], id_map = load_idv_outcomes(h5_path)),
    #                       lambda cur_volume_ptr: (np.array([]), np.array([])),
    #                       # all_ids = list(id_map.values()),
    #                       # split_number = 1,
    #                       # recomm_ratio = 1.0,
    #                       )
    # viewer.save_all(raw_img_root, len(vol_names), fps = 1)


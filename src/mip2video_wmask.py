import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def get_color(id, colors):
    # Assigns a color from the list based on the ID
    return colors[id % len(colors)]


def calculate_brightness(image, rect):
    # Extract the rectangle area
    # x1, y1, x2, y2 = map(int, rect)
    # x1, y1, x2, y2 = rect
    rect_area = image[rect[1]:rect[3], rect[0]:rect[2]]
    # Calculate average brightness
    brightness = np.mean(rect_area)
    return brightness

def get_image4processing(volume, is_max = True):
    """
    the projection of maximum value along z-axis is better than mean value.
    ATTENTION: Volume from WenLab system need to eliminate the last two slice.
    :param volume: H x W x S, (S = 22 or 18 for 23 or 20) in WenLab system
    :param is_max:
    :return:
    """

    image = np.max(volume, axis = -1) if is_max else np.mean(volume, axis = -1)

    return image



def cxcywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    if len(x.shape) == 1:
        y[0] = x[0] - x[2] * .5  # top left x
        y[1] = x[1] - x[3] * .5  # top left y
        y[2] = x[0] + x[2] * .5  # bottom right x
        y[3] = x[1] + x[3] * .5  # bottom right y
        return y
    y[:, 0] = x[:, 0] - x[:, 2] * .5  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] * .5  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] * .5  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] * .5  # bottom right y
    return y

def add_text_to_image(image, text, position=(50, 50), font_scale=1, font_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    cv2.putText(image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    return image

def volume_mip_video(video_path, all_volume, data_list):
    """
    return the mip projection of volumes as time-dependent video
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_path, fourcc, 5.0, (512, 512), False)
    brightness_data = {}
    for i, volume in enumerate(all_volume):
        aligned_image = get_image4processing(volume)
        aligned_image = (aligned_image / aligned_image.max() * 255).astype(np.uint8)
        frame_text = f" group {(i + 31) // 30}, volume {(i+1) % 30}"
        aligned_image_with_text = add_text_to_image(aligned_image, frame_text)
        
        for idx, rect in enumerate(data_list[:20]) :
            # brightness = calculate_brightness(aligned_image_with_text, rect)
            # brightness_data[idx].append(brightness)
            cv2.rectangle(aligned_image_with_text, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (255, 0, 0), thickness = 1, lineType = cv2.LINE_AA)
        
        # fig, axes = plt.subplots(1, 2, figsize=(8, 12))
        video.write(aligned_image_with_text)
        
    cv2.destroyAllWindows()
    video.release()
    
    return None


# def volume_mip_video(video_path, all_volume, data_list):

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     video = cv2.VideoWriter(video_path, fourcc, 5.0, (1024, 512), False)
    
#     brightness_values = []

#     for i, volume in enumerate(all_volume):
#         fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Adjusted figsize to match video resolution
#         canvas = FigureCanvas(fig)

#         aligned_image = get_image4processing(volume)
#         aligned_image = (aligned_image / aligned_image.max() * 255).astype(np.uint8)
#         frame_text = f" group {(i + 31) // 30}, volume {(i+1) % 30}"
#         aligned_image_with_text = add_text_to_image(aligned_image, frame_text)
#         aligned_image_with_text = cv2.cvtColor(aligned_image_with_text, cv2.COLOR_GRAY2BGR)

#         rect = data_list  # Assuming data_list is a single rectangle
#         brightness = calculate_brightness(aligned_image_with_text, rect)
#         brightness_values.append(brightness)
#         cv2.rectangle(aligned_image_with_text, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

#         axes[0].imshow(aligned_image_with_text)
#         axes[0].axis('off')  # Turn off axis for image subplot
#         axes[1].clear()
#         axes[1].plot(brightness_values)
        
#         canvas.draw()  # Draw the canvas
#         plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
#         plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
#         # Resize plot image to match aligned_image_with_text
#         plot_image_resized = cv2.resize(plot_image, (512, 512))

#         # Combine and write frame
#         combined_frame = np.concatenate((aligned_image_with_text, plot_image_resized), axis=1)
#         video.write(combined_frame)

#         plt.close(fig)  # Close the figure to free

#     cv2.destroyAllWindows()
#     video.release()
    
#     return None


    
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter(video_path, fourcc, 5.0, (1024, 512), False)

    # fig, axes = plt.subplots(1, 2, figsize=(8, 12))
    # # canvas = FigureCanvas(fig)
    # brightness_values = []

    # for i, volume in enumerate(all_volume):
    #     aligned_image = get_image4processing(volume)
    #     aligned_image = (aligned_image / aligned_image.max() * 255).astype(np.uint8)
    #     frame_text = f" group {(i + 31) // 30}, volume {(i+1) % 30}"
    #     aligned_image_with_text = add_text_to_image(aligned_image, frame_text)
    #     aligned_image_with_text = cv2.cvtColor(aligned_image_with_text, cv2.COLOR_GRAY2BGR)

    #     # for idx, rect in enumerate(data_list):
    #     rect = data_list
    #     brightness = calculate_brightness(aligned_image_with_text, rect)
    #     brightness_values.append(brightness)
    #     cv2.rectangle(aligned_image_with_text, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (255, 0, 0), thickness = 1, lineType = cv2.LINE_AA)

    #     axes[0].imshow(aligned_image_with_text)
    #     axes[1].clear()
    #     axes[1].plot(brightness_values)
    #     video.write(fig)

    # cv2.destroyAllWindows()
    # video.release()


video_path = "/home/wenlab-user/RongWei/olfactory/20231118/w2/synthetic_volume/w2_pixel_intensity_1.avi"
all_volume_path ="/home/wenlab-user/RongWei/olfactory/20231118/w2/synthetic_volume/w2_aligned_volume_matrix.npy"
neuron_pt_tuple = np.load('/home/wenlab-user/RongWei/olfactory/20231118/w2/synthetic_volume/w2_neuron_pt_tuple.npy')


if __name__ == '__main__':

    area_ratio = 0.95
    area_reduction = torch.tensor([1, 1, area_ratio, area_ratio], dtype = torch.float32)
    
    data_list = []
    
    for neuron in neuron_pt_tuple:
        b = cxcywh2xyxy(torch.tensor(neuron[[0, 1, 3, 4]]) * area_reduction).to(dtype = torch.int32)
        data_list.append(b)

    
    all_volume = np.load(all_volume_path)
    # volume_mip_figure(all_volume, range(all_volume.shape[0]))
    volume_mip_video(video_path, all_volume, data_list)
# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from src.comm_utils.packages import *
from src.preproc.auto import preprocess
from src.comm_utils.prints import print_error_message


def auto_preprocess(mode: int, paths, zrange, load_preprocess_result_root, save_preprocess_result_root, name_reg):
    """

    :param mode: the flag of preprocessing
                0: only return volume after clipping from zrange[0] to zrange[1]

                1: 0 + return maximum projection image
                2: 1 + return the contour of head region
                3: 2 + return its 5 pts of C. elegans Coordinates system

                4: 1 + save maximum projection image
                5: 4 + save the json file of contour of head region
                6: 5 + save 5 pts of C. elegans Coordinates system

    :param args:
    :return:
        0: return volume
        1 & 4: return [volume,
                    projection]
        2 & 5: return [volume,
                projection,
                [convex_head_ctr, is_head_region_warning, convex_head_binary_region, rect_of_roi, tail_ctrs, noise_ctrs]]
        3 & 6: return [volume,
                    projection,
                    [convex_head_ctr, is_head_region_warning, convex_head_binary_region, rect_of_roi, tail_ctrs, noise_ctrs],
                    [mass_of_center, upper_y, lower_y, right_x, left_x]]
    """

    if mode not in (0, 1, 2, 3, 4, 5, 6):
        print_error_message(f"{mode} is not supported!")

    params = [[stack_path, mode, zrange, load_preprocess_result_root, save_preprocess_result_root, name_reg] for stack_path in paths]
    with Pool(min(os.cpu_count() // 2, len(params))) as p:
        with tqdm(p.imap_unordered(preprocess, params), total = len(params), desc = "Stack loading + S.1 preprocessing") as pbar:
            total_results = {n: r for out in list(pbar) if out for n, r in out[1].items()}

    return total_results


def _maxpooling(image: np.ndarray, ratio):
    dtype = image.dtype
    if len(image.shape) == 2:
        image = image[np.newaxis]
    image = torch.FloatTensor(image.astype(np.float32))
    image = F.max_pool2d(image, kernel_size = ratio, stride = ratio)
    return image.numpy().astype(dtype)


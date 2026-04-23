"""
Autofluorescence masking utilities.

Two detection methods are provided:

``'neuron_boundary'`` (default)
    Two-step binarization + morphological dilation:

    1. High-percentile threshold → main neuron cluster (core)
    2. Dilate core by *dilation_iters* pixels to define a search zone
    3. Low-percentile threshold → broad candidate mask
    4. Intersection of dilated zone and broad mask → extended neuron region
    5. y cutoff = bottom of extended region + *margin*

    Suited for datasets where autofluorescence appears as a diffuse cloud
    *below* the neuron region.

``'diffuse_ratio'``
    Uses the ratio of large-sigma to small-sigma Gaussian blurs to identify
    spatially diffuse signal in a specified image corner/edge.  Suited for
    cases where autofluorescence is confined to a corner and is visually
    distinct from the neuron cluster.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Manual masking helpers
# ---------------------------------------------------------------------------

def modify_image(file_path, y=None, x=None, background_value=102):
    """Set a rectangular edge/corner region of a .npy volume to *background_value*.

    Args:
        file_path: path to the .npy file (Y, X, Z)
        y: row index; rows ``[y:]`` are set to *background_value*
        x: column index; columns ``[x:]`` are set to *background_value*
        background_value: fill value (default 102)
    """
    image = np.load(file_path)
    if y is not None and x is not None:
        image[y:, x:, :] = background_value
    elif y is not None:
        image[y:, :, :] = background_value
    elif x is not None:
        image[:, x:, :] = background_value
    np.save(file_path, image)


def modify_folder(folder_path, y=None, x=None, background_value=102):
    """Apply :func:`modify_image` to every .npy file in *folder_path*.

    *y* and *x* can each be:
      - a scalar  — applied uniformly to every file
      - a list    — one value per file (sorted filename order)
      - ``None``  — that axis is not masked

    Args:
        folder_path: directory containing .npy files
        y: row cutoff(s)
        x: column cutoff(s)
        background_value: fill value (default 102)
    """
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    if isinstance(y, (list, np.ndarray)) and len(y) != len(files):
        raise ValueError(
            f"Length of y ({len(y)}) does not match number of files ({len(files)})"
        )
    if isinstance(x, (list, np.ndarray)) and len(x) != len(files):
        raise ValueError(
            f"Length of x ({len(x)}) does not match number of files ({len(files)})"
        )
    for i, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        current_y = y[i] if isinstance(y, (list, np.ndarray)) else y
        current_x = x[i] if isinstance(x, (list, np.ndarray)) else x
        modify_image(file_path, y=current_y, x=current_x,
                     background_value=background_value)


# ---------------------------------------------------------------------------
# Automatic detection
# ---------------------------------------------------------------------------

def detect_autofluorescence_boundary(folder_path, background_value=102,
                                      method='neuron_boundary',
                                      neuron_percentile_high=98,
                                      neuron_percentile_low=85,
                                      dilation_iters=50,
                                      margin=20,
                                      sigma_large=12.0, sigma_small=2.0,
                                      intensity_floor=0.05,
                                      min_area_fraction=0.01,
                                      corner='auto', per_frame=False):
    """Automatically detect the y/x boundary of an autofluorescence region.

    Each .npy volume must have shape ``(Y, X, Z)``.  The function projects
    along Z (max-intensity projection) and works in 2-D.

    Parameters
    ----------
    folder_path : str
        Directory containing .npy volume files.
    background_value : int or float
        Baseline background intensity (default 102).
    method : str
        Detection method.

        ``'neuron_boundary'`` *(default)*
            Two-step binarization + morphological dilation.  Use this when
            autofluorescence forms a diffuse cloud *below* the neuron region.

        ``'diffuse_ratio'``
            Gaussian blur ratio method.  Use this when autofluorescence is
            confined to a corner/edge of the image.

    neuron_percentile_high : float
        (``'neuron_boundary'``) High threshold percentile for the core neuron
        cluster (default 98).
    neuron_percentile_low : float
        (``'neuron_boundary'``) Low threshold percentile for the broad
        candidate mask that captures dimmer isolated neurons (default 85).
    dilation_iters : int
        (``'neuron_boundary'``) Morphological dilation iterations applied to
        the core cluster to define the search zone (default 50).  Can be set
        large safely because the intersection with the broad mask prevents
        false positives from far-away autofluorescence.
    margin : int
        (``'neuron_boundary'``) Extra pixels below the extended cluster bottom
        before setting the y cutoff (default 20).
    sigma_large : float
        (``'diffuse_ratio'``) Large Gaussian sigma in pixels.
    sigma_small : float
        (``'diffuse_ratio'``) Small Gaussian sigma in pixels.
    intensity_floor : float
        (``'diffuse_ratio'``) Minimum normalised intensity (0–1).
    min_area_fraction : float
        (``'diffuse_ratio'``) Minimum region area as a fraction of image area.
    corner : str
        (``'diffuse_ratio'``) Corner/edge: ``'auto'``, ``'bottom-right'``,
        ``'bottom-left'``, ``'top-right'``, ``'top-left'``, ``'bottom'``,
        ``'top'``, ``'left'``, ``'right'``.
    per_frame : bool
        If ``True`` return per-frame cutoff lists; if ``False`` use the
        temporal mean to produce a single cutoff pair.

    Returns
    -------
    (y_cutoff, x_cutoff)
        - ``per_frame=False``: each element is a scalar or ``None``.
        - ``per_frame=True``: each element is a list (``None`` where no region
          was detected) or ``None`` if nothing was detected at all.
    """
    from scipy.ndimage import label as nd_label, binary_dilation

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    if not files:
        return None, None

    def _mip(fpath):
        vol = np.load(fpath).astype(np.float32)
        return vol.max(axis=2)  # (Y, X)

    # ------------------------------------------------------------------
    # Method 1: neuron_boundary (two-step binarization + dilation)
    # ------------------------------------------------------------------
    def _neuron_boundary(mip_2d):
        signal = np.clip(mip_2d - background_value, 0, None)
        if signal.max() == 0:
            return None, None

        # Step 1: high threshold → core cluster (largest component)
        thresh_high = np.percentile(signal[signal > 0], neuron_percentile_high)
        core_mask = signal > thresh_high
        labeled, n = nd_label(core_mask)
        if n == 0:
            return None, None
        sizes = [int((labeled == i).sum()) for i in range(1, n + 1)]
        main_cluster = labeled == (int(np.argmax(sizes)) + 1)

        # Step 2: dilate core to create search zone
        dilated = binary_dilation(main_cluster, iterations=dilation_iters)

        # Step 3: low threshold → broad candidate mask
        thresh_low = np.percentile(signal[signal > 0], neuron_percentile_low)
        broad_mask = signal > thresh_low

        # Step 4: intersection — keeps only bright pixels near the cluster
        extended = dilated & broad_mask
        if not extended.any():
            return None, None

        ys, _ = np.where(extended)
        y_cut = min(int(ys.max()) + margin, mip_2d.shape[0] - 1)
        return y_cut, None

    # ------------------------------------------------------------------
    # Method 2: diffuse_ratio
    # ------------------------------------------------------------------
    def _diffuse_ratio(mip_2d):
        from scipy.ndimage import gaussian_filter

        signal = np.clip(mip_2d - background_value, 0, None)
        sig_max = signal.max()
        if sig_max == 0:
            return None, None

        signal_norm = signal / sig_max
        blurred_large = gaussian_filter(signal_norm, sigma=sigma_large)
        blurred_small = gaussian_filter(signal_norm, sigma=sigma_small)
        ratio = blurred_large / (blurred_small + 1e-6)

        mask = (signal_norm > intensity_floor) & (ratio > 0.5)
        if not mask.any():
            return None, None

        H, W = mip_2d.shape
        target = corner
        if corner == 'auto':
            q = {
                'bottom-right': mask[H // 2:, W // 2:].sum(),
                'bottom-left':  mask[H // 2:, :W // 2].sum(),
                'top-right':    mask[:H // 2, W // 2:].sum(),
                'top-left':     mask[:H // 2, :W // 2].sum(),
            }
            target = max(q, key=q.get)
            if q[target] == 0:
                return None, None

        labeled, n = nd_label(mask)
        if n == 0:
            return None, None

        y_half, x_half = H // 2, W // 2
        best_region, best_area = None, 0
        for rid in range(1, n + 1):
            region = labeled == rid
            area = int(region.sum())
            if area < min_area_fraction * H * W:
                continue
            ys, xs = np.where(region)
            cy, cx = ys.mean(), xs.mean()
            if ('bottom' in target and cy < y_half) or \
               ('top'    in target and cy >= y_half) or \
               ('right'  in target and cx < x_half) or \
               ('left'   in target and cx >= x_half):
                continue
            if area > best_area:
                best_area = area
                best_region = region

        if best_region is None:
            return None, None

        ys, xs = np.where(best_region)
        y_cut = (int(ys.min()) if 'bottom' in target
                 else int(ys.max()) if 'top' in target
                 else None)
        x_cut = (int(xs.min()) if 'right' in target
                 else int(xs.max()) if 'left' in target
                 else None)
        return y_cut, x_cut

    _detect = _neuron_boundary if method == 'neuron_boundary' else _diffuse_ratio

    if not per_frame:
        mean_mip = None
        for fname in files:
            mip = _mip(os.path.join(folder_path, fname))
            mean_mip = mip if mean_mip is None else mean_mip + mip
        mean_mip /= len(files)
        return _detect(mean_mip)
    else:
        y_list, x_list = [], []
        for fname in files:
            yc, xc = _detect(_mip(os.path.join(folder_path, fname)))
            y_list.append(yc)
            x_list.append(xc)
        return (y_list if any(v is not None for v in y_list) else None,
                x_list if any(v is not None for v in x_list) else None)


def save_boundary_visualization(folder_path, y_cuts, x_cuts=None,
                                 output_path=None):
    """Save a per-frame boundary visualization PNG.

    Shows each .npy file's MIP with the detected y/x cutline and the masked
    region highlighted in yellow.

    Args:
        folder_path: directory containing .npy volume files (Y, X, Z)
        y_cuts: list of y cutoffs (one per file, may contain None)
        x_cuts: list of x cutoffs (one per file, may contain None); optional
        output_path: path to save the PNG; defaults to
            ``<folder_path>/per_frame_boundary.png``

    Returns:
        str: path to the saved PNG
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = os.path.join(folder_path, 'per_frame_boundary.png')

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    mips = [np.load(os.path.join(folder_path, f)).astype(np.float32).max(axis=2)
            for f in files]

    if x_cuts is None:
        x_cuts = [None] * len(files)

    ncols = 3
    nrows = (len(files) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 6))
    axes = axes.flatten()

    vmin = np.percentile(np.stack(mips), 1)
    vmax = np.percentile(np.stack(mips), 99.5)

    for i, (mip, fname, yc, xc) in enumerate(zip(mips, files, y_cuts, x_cuts)):
        ax = axes[i]
        ax.imshow(mip, cmap='gray', vmin=vmin, vmax=vmax)
        H, W = mip.shape

        if yc is not None:
            ax.axhline(y=yc, color='r', linewidth=2)
        if xc is not None:
            ax.axvline(x=xc, color='b', linewidth=2)

        # shade masked region
        y0 = yc if yc is not None else 0
        x0 = xc if xc is not None else 0
        if yc is not None or xc is not None:
            ax.fill_between([x0, W], y0, H, color='yellow', alpha=0.25)

        label = f'y={yc}' + (f', x={xc}' if xc is not None else '')
        ax.set_title(f'{fname}\n{label}', fontsize=7)
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Autofluorescence boundary (per frame)', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_corrected_copy(src_folder, dst_folder, background_value=102,
                          method='neuron_boundary',
                          neuron_percentile_high=98,
                          neuron_percentile_low=85,
                          dilation_iters=50,
                          margin=20,
                          sigma_large=12.0, sigma_small=2.0,
                          intensity_floor=0.05, min_area_fraction=0.01,
                          corner='auto', vis_output_path=None):
    """Detect autofluorescence, copy *src_folder* to *dst_folder*, and apply
    the correction to the **copy only**.  The original files in *src_folder*
    are never modified.

    To undo the correction simply delete *dst_folder* and re-run with
    adjusted parameters.

    Args:
        src_folder: directory of original .npy volumes (Y, X, Z) — read-only
        dst_folder: directory to write corrected copies into (created if needed)
        background_value: fill value for the masked region (default 102)
        vis_output_path: if provided, save per-frame boundary PNG here
        (remaining kwargs are forwarded to
        :func:`detect_autofluorescence_boundary`)

    Returns:
        str: *dst_folder* path (ready to use as a drop-in for *src_folder*)
    """
    import shutil

    y_cut, x_cut = detect_autofluorescence_boundary(
        src_folder,
        background_value=background_value,
        method=method,
        neuron_percentile_high=neuron_percentile_high,
        neuron_percentile_low=neuron_percentile_low,
        dilation_iters=dilation_iters,
        margin=margin,
        sigma_large=sigma_large,
        sigma_small=sigma_small,
        intensity_floor=intensity_floor,
        min_area_fraction=min_area_fraction,
        corner=corner,
        per_frame=True,
    )

    files = sorted([f for f in os.listdir(src_folder) if f.endswith('.npy')])
    n = len(files)
    y_list = y_cut if isinstance(y_cut, list) else [y_cut] * n
    x_list = x_cut if isinstance(x_cut, list) else [x_cut] * n

    if vis_output_path is not None:
        saved = save_boundary_visualization(src_folder, y_list, x_list,
                                            output_path=vis_output_path)
        print(f"[INFO] Boundary visualization saved to {saved}")

    if y_cut is None and x_cut is None:
        print(f"[WARNING] No autofluorescence detected in {src_folder}. "
              f"Copying files unmodified to {dst_folder}.")

    os.makedirs(dst_folder, exist_ok=True)
    for fname, yc, xc in zip(files, y_list, x_list):
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, fname)
        shutil.copy2(src_path, dst_path)
        if yc is not None or xc is not None:
            modify_image(dst_path, y=yc, x=xc, background_value=background_value)

    print(f"[INFO] Corrected copies written to {dst_folder}")
    return dst_folder


def auto_modify_folder(folder_path, background_value=102,
                       method='neuron_boundary',
                       neuron_percentile_high=98,
                       neuron_percentile_low=85,
                       dilation_iters=50,
                       margin=20,
                       sigma_large=12.0, sigma_small=2.0,
                       intensity_floor=0.05, min_area_fraction=0.01,
                       corner='auto', per_frame=False,
                       vis_output_path=None, dry_run=False):
    """Detect and mask autofluorescence in all .npy volumes in a folder.

    Combines :func:`detect_autofluorescence_boundary`,
    :func:`save_boundary_visualization`, and :func:`modify_folder`.

    Parameters
    ----------
    vis_output_path : str or None
        If provided, save a per-frame boundary PNG to this path.
    dry_run : bool
        If ``True``, detect and visualise but do **not** modify any files.

    Returns
    -------
    (y_cutoff, x_cutoff)
    """
    y_cut, x_cut = detect_autofluorescence_boundary(
        folder_path,
        background_value=background_value,
        method=method,
        neuron_percentile_high=neuron_percentile_high,
        neuron_percentile_low=neuron_percentile_low,
        dilation_iters=dilation_iters,
        margin=margin,
        sigma_large=sigma_large,
        sigma_small=sigma_small,
        intensity_floor=intensity_floor,
        min_area_fraction=min_area_fraction,
        corner=corner,
        per_frame=per_frame,
    )

    if y_cut is None and x_cut is None:
        print(f"[WARNING] No autofluorescence region detected in {folder_path}.")
        return None, None

    print(f"[INFO] Detected autofluorescence boundary: y={y_cut}, x={x_cut}")

    # normalise to lists for visualisation
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    n = len(files)
    y_list = y_cut if isinstance(y_cut, list) else [y_cut] * n
    x_list = x_cut if isinstance(x_cut, list) else [x_cut] * n

    if vis_output_path is not None:
        saved = save_boundary_visualization(folder_path, y_list, x_list,
                                            output_path=vis_output_path)
        print(f"[INFO] Boundary visualization saved to {saved}")

    if not dry_run:
        modify_folder(folder_path, y=y_cut, x=x_cut,
                      background_value=background_value)
        print(f"[INFO] Applied autofluorescence mask to all files in {folder_path}.")

    return y_cut, x_cut

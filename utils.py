"""
Image preprocessing and shared utility functions.

This module provides the image normalisation and segmentation pipeline used
across MF-DFA and TDA analysis, as well as helper functions for scale
generation and multifractal moment sampling.

Key functions
-------------
normalize_image   : Resize to a fixed square canvas with reflective padding.
segment_image     : Split a normalised image into a grid of sub-regions.
preprocess_image  : End-to-end preprocessing pipeline (read → normalise → segment).
bineo             : Generate log-spaced scales for multifractal analysis.
vals_Qs           : Non-uniform q-moment sequence, denser near q = 0.
binomial_cascade_2d : 2-D binomial multiplicative cascade (synthetic benchmark).
"""

import cv2
import numpy as np
from numba import njit
 
def normalize_image(img, max_size=1380, gray = True):
    """
    Normalise an image to a square canvas of max_size × max_size pixels.

    Procedure:
    1. Downscale while preserving the aspect ratio (reduction only).
    2. Pad the remaining space with border reflection.
    
    Parameters
    ----------
    img : ndarray
        Imagen en escala de grises (H, W) o color (H, W, C).
    max_size : int
        Target size. Default 1380, chosen to be close to the minimum
        image dimension in the dataset and divisible by 1, 2, 3 and 4.
    
    Returns
    -------
    img_norm : ndarray
        Normalised image of shape (max_size, max_size) or (max_size, max_size, C).
    """
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError("La imagen debe ser 2D o 3D.")
 
    # Downscale preserving aspect ratio
    if h >= w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
 
    img_resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )
 
    # Symmetric reflective padding
    pad_top = (max_size - new_h) // 2
    pad_bottom = max_size - new_h - pad_top
    pad_left = (max_size - new_w) // 2
    pad_right = max_size - new_w - pad_left
 
    img_norm = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REFLECT
    )

    if gray == True: img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
 
    return img_norm
 
 
def segment_image(img, grid_size=1):
    """
    Split an image into a grid_size × grid_size grid of sub-regions.
    
    Parameters
    ----------
    img : ndarray
        2-D (H, W) or 3-D (H, W, C) image. Dimensions must be divisible
        by grid_size (guaranteed when max_size=1380).
    grid_size : int
        Divisions per side. Valid values: 1, 2, 3 or 4, producing
        1, 4, 9 or 16 sub-regions respectively.
    
    Returns
    -------
    segments : list of ndarray
        Sub-regions in row-major order (left-to-right, top-to-bottom).
    """
    if grid_size < 1:
        raise ValueError("grid_size debe ser >= 1.")
 
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError("La imagen debe ser 2D o 3D.")
 
    if h % grid_size != 0 or w % grid_size != 0:
        raise ValueError(
            f"Las dimensiones ({h}, {w}) no son divisibles entre grid_size={grid_size}."
        )
 
    step_h = h // grid_size
    step_w = w // grid_size
    segments = []
 
    for i in range(grid_size):
        for j in range(grid_size):
            segment = img[
                i * step_h : (i + 1) * step_h,
                j * step_w : (j + 1) * step_w
            ]
            segments.append(segment)
 
    return segments
 
 
def preprocess_image(img_path, max_size=1380, grid_sizes=(1, 2, 3, 4)):
    """
    End-to-end preprocessing pipeline for a single painting.
    
    Parameters
    ----------
    img_path : str
        Path to the image file.
    max_size : int
        Normalisation target size.
    grid_sizes : tuple of int
        Grid sizes to apply during segmentation.
    
    Returns
    -------
    results : dict
        Dictionary with key 'grid_{n}' for each grid size, where the
        value is a list of grayscale sub-region arrays.
    """
    # Read as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
 
    # Normalizar
    img_norm = normalize_image(img, max_size=max_size)
 
    # Segmentar en cada nivel de partición
    results = {}
    for gs in grid_sizes:
        segments = segment_image(img_norm, grid_size=gs)
        results[f'grid_{gs}'] = segments
 
    return results

# =============================================================================
# Auxiliary functions
# =============================================================================

def sub_mean(x):
    """Subtract the mean from an array."""
    return x - np.mean(x)


@njit
def bineo(s_min, s_max, degree=1):
    """
    Generate a log-spaced sequence of scales.

    Spacing is controlled by progressive square roots of 2:
    degree=1 uses √2 ≈ 1.414 (coarse), degree=2 uses 2^(1/4) ≈ 1.189
    (medium), degree=3 uses 2^(1/8) ≈ 1.091 (dense, more points for
    regression).

    Parameters
    ----------
    s_min : int
        Minimum scale in pixels.
    s_max : int
        Maximum scale in pixels.
    degree : int
        Controls sampling density. Larger values produce more
        intermediate scales.

    Returns
    -------
    N_s : ndarray
        Array of integer scales.
    """
    s = s_min
    val = 2
    N_s = []
    for i in range(1, degree + 1):
        val = np.sqrt(val)  # Progressive root: degree=3 → 2^(1/8)
    while s < s_max:
        N_s.append(s)
        s = int(s * val) + 1
    return np.array(N_s)


def valor_cercano(lista, valor):
    """Return the element in `lista` closest to `valor`."""
    lista = np.array(lista)
    resta = np.abs(lista - valor)
    return lista[resta == min(resta)]


def vals_Qs(q_n, q_p):
    """
    Generate non-uniformly spaced q-moment values.

    Values are denser near q = 0, where h(q) varies most rapidly,
    and sparser toward the extremes. This improves spectral resolution
    in the transition zone between weak (q < 0) and strong (q > 0)
    fluctuations.

    Parameters
    ----------
    q_n : float
        Negative extreme of the moment range (e.g. -5.0).
    q_p : float
        Positive extreme of the moment range (e.g. 5.0).

    Returns
    -------
    Qs : list of float
        q-moment values sorted in ascending order.
    """
    cuts = ([float(0)]
            + [float(i) for i in bineo(s_min=0.25, s_max=q_p, degree=3)]
            + [float(q_p)])

    l = []
    for i, cut in enumerate(cuts[1:]):
        l += [float(x) for x in np.arange(cuts[i], cut, step=cut / 2)]

    i_min = l.index(valor_cercano(l, np.abs(q_n)))
    i_max = l.index(valor_cercano(l, q_p))

    # Negative branch (mirror of the positive branch)
    l_min = [ll * -1 for ll in l[1:i_min]]
    l_min.reverse()

    l_max = l[:i_max]
    Qs = l_min + l_max
    return Qs


def binomial_cascade_2d(N,p1 = 0.1,p2 =0.2,p3 = 0.3,p4 = 0.4, q_min = -10, q_max = 10, int_img=True):
    mult = np.array([[p1, p2], [p3, p4]])  # Multiplicative probabilities in 2D
    cascade = np.array([[1]])  # Initial state of the cascade
    for _ in range(N):
        X, Y = cascade.shape  # Get the current dimensions of the cascade
        new_cascade = np.zeros([X * 2, Y * 2])  # Create a new matrix for the next iteration
        for x in range(X):
            for y in range(Y):
                # Multiply each cell by the probabilities and expand into a 2x2 matrix
                opera = cascade[x, y] * mult
                new_cascade[x * 2:x * 2 + 2, y * 2:y * 2 + 2] = opera
        cascade = new_cascade  # Update the cascade for the next iteration

        # qs = ut.vals_Qs(q_min,q_max)
        qs = np.arange(q_min - 0.25,q_max + 0.25,0.25)
        Dqs = []
        tau_q = []


        for q in qs:
            
            if q != 1:
                Dq =  np.log(p1**q + p2**q + p3**q + p4**q) / ((1 - q) * np.log(2))
                Tq = Dq * (q-1)
                
            else:
                # Dq =  np.log(p1**q + p2**q + p3**q + p4**q) / np.log(2)
                Dq = -(p1*np.log(p1) + p2*np.log(p2) + p3*np.log(p3) + p4*np.log(p4)) / np.log(2)
                Tq = 0
                

            Dqs.append( Dq )
            tau_q.append( Tq )

        # ---- Espectro multifractal f(α) via transformada de Legendre ----
        alpha = np.gradient(tau_q, qs)       # α(q) = dτ/dq
        f_alpha = qs * alpha - tau_q         # f(α) = q·α - τ(q)

        alpha = alpha[1:-1]
        f_alpha = f_alpha[1:-1]

    data = { 'alpha':alpha, 'f_alpha':f_alpha, 'tau':tau_q, 'qs':qs}
    
    if int_img == True: cascade = np.array(np.round((cascade / np.max(cascade)) * 255), dtype=np.int16)
    return cascade,data




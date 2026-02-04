import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows

def add_glcm_channels(
    image,
    base_channel=0,
    window_size=7,
    levels=16
):


    assert window_size % 2 == 1

    n, m, C = image.shape
    pad = window_size // 2

    # --- canal base ---
    base = image[..., base_channel].astype(np.float32)

    # normaliza
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    base_q = (base * (levels - 1)).astype(np.uint8)

    # padding
    base_pad = np.pad(base_q, pad, mode="reflect")

    windows = view_as_windows(base_pad, (window_size, window_size))

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm_map = np.zeros((n, m, 16), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            patch = windows[i, j]

            glcm = graycomatrix(
                patch,
                distances=[1],
                angles=angles,
                levels=levels,
                symmetric=True,
                normed=True
            )

            contrast = graycoprops(glcm, 'contrast')[0]
            energy   = graycoprops(glcm, 'energy')[0]
            corr     = graycoprops(glcm, 'correlation')[0]

            entropy = np.zeros(4)
            for k in range(4):
                P = glcm[:, :, 0, k]
                entropy[k] = -np.sum(P * np.log(P + 1e-12))

            glcm_map[i, j, :] = np.concatenate(
                [contrast, energy, entropy, corr]
            )

    # --- concatena ---
    image_out = np.concatenate(
        (image, glcm_map),
        axis=-1
    )

    return image_out

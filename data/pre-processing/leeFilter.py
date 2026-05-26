import numpy as np
from scipy.ndimage import uniform_filter

def lee_filter(img, size=7):
    """
    Aplica filtro de Lee em uma imagem 2D
    """
    img = img.astype(np.float32)

    # média local
    mean = uniform_filter(img, size)

    # média do quadrado
    mean_sq = uniform_filter(img**2, size)

    # variância local
    var = mean_sq - mean**2

    # variância do ruído (estimativa global)
    noise_var = np.mean(var)

    # peso do filtro
    W = var / (var + noise_var + 1e-8)

    # imagem filtrada
    filtered = mean + W * (img - mean)

    return filtered


def lee_filter_multichannel(img, size=7):
    """
    img: (n, m, c)
    """
    n, m, c = img.shape
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(c):
        output[:, :, i] = lee_filter(img[:, :, i], size)

    return output
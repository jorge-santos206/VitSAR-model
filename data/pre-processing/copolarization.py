import numpy as np

def copolarization(imagem):
    T = imagem.copy()
    cross_pol = T[:, :, 2] / (T[:, :, 0])


    return np.concatenate(
        (imagem, cross_pol[..., None]),
        axis=-1
    )
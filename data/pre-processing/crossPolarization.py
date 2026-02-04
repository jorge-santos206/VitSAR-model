import numpy as np
def crosspolarization(imagem):
    T = imagem.copy()

    cross_pol = T[:, :, 1] / (T[:, :, 0])

    return np.concatenate(
        (imagem, cross_pol[..., None]),
        axis=-1
    )
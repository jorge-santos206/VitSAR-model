import numpy as np
import numpy as np

def huynenDecomposition(imagem):

    T = imagem.copy()

    A0 = T[:, :, 0] / 2
    B0 = (T[:, :, 1] + T[:, :, 2]) / 2
    B  = (T[:, :, 1] - T[:, :, 2]) / 2
    C  = T[:, :, 3] / 2
    D  = -T[:, :, 6]
    E  = T[:, :, 5]
    F  = T[:, :, 8]
    G  = T[:, :, 7]
    H  = T[:, :, 4]

    new_channels = np.stack(
        [A0, B0, B, C, D, E, F, G, H],
        axis=-1
    )

    return np.concatenate((imagem, new_channels), axis=-1)

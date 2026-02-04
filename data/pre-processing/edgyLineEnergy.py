import numpy as np
from scipy.ndimage import convolve, rotate

def add_edge_energy_channels(
    image,
    base_channel=0,
    n_directions=18,
    n_scales=4
):
    """
    Entrada:
        image : (n, m, C)
    Saída:
        image : (n, m, C + 4)
    """

    base = image[..., base_channel].astype(np.float32)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    sobel = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float32
    )

    angles = np.linspace(0, 180, n_directions, endpoint=False)

    responses = []
    angle_bins = [[] for _ in range(4)]

    for s in range(1, n_scales + 1):
        for a in angles:
            kernel = rotate(sobel, a, reshape=False)
            resp = np.abs(convolve(base, kernel, mode="reflect"))
            responses.append((a, resp))

    # agrupa nas 4 direções principais
    for a, resp in responses:
        idx = int(np.floor(a / 45)) % 4
        angle_bins[idx].append(resp)

    edge_channels = [
        np.mean(bin_resp, axis=0) for bin_resp in angle_bins
    ]

    edge_channels = np.stack(edge_channels, axis=-1)

    return np.concatenate((image, edge_channels), axis=-1)


import numpy as np

def add_line_energy_channels(image, edge_start_channel=-4):
    """
    Usa os 4 canais de edge energy já adicionados.
    Espera que esses canais estejam em posições consecutivas.
    """

    edge = image[..., edge_start_channel:]

    if edge.shape[-1] < 4:
        raise ValueError(
            f"Esperado 4 canais de edge energy, mas recebi {edge.shape[-1]}"
        )

    # pares ortogonais: (0,2) e (1,3)
    line_0 = np.minimum(edge[..., 0], edge[..., 2])
    line_1 = np.minimum(edge[..., 1], edge[..., 3])

    # replica para manter 4 direções (como no paper)
    line_channels = np.stack(
        [line_0, line_1, line_0, line_1],
        axis=-1
    )

    return np.concatenate((image, line_channels), axis=-1)

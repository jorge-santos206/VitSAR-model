
import numpy as np

try:
    from scipy.optimize import nnls
    _HAS_NNLS = True
except Exception:
    _HAS_NNLS = False


def _vec_T_matrix(T):
        """
        Converte matriz 3x3 complexa T para um vetor real 9x1 com a ordem:
        [Re(T11), Re(T22), Re(T33), Re(T12), Im(T12), Re(T13), Im(T13), Re(T23), Im(T23)]
        """
        return np.array([
            T[0, 0].real,
            T[1, 1].real,
            T[2, 2].real,
            T[0, 1].real,
            T[0, 1].imag,
            T[0, 2].real,
            T[0, 2].imag,
            T[1, 2].real,
            T[1, 2].imag
        ], dtype=float)


def build_model_matrices():
        """
        Retorna os 3 modelos T_s, T_d, T_v (no espaço Pauli) como matrizes 3x3 complexas.
        Observação: usamos as matrizes usualmente adotadas na literatura
        (modelo de dipolos aleatórios para volume). A decomposição é então resolvida
        pela inversão dos 9 observáveis reais.
        """
        # surface (odd-bounce) modelo (parametrizado aqui no caso simples)
        Ts = np.zeros((3, 3), dtype=complex)
        Ts[0, 0] = 1.0

        # double-bounce (even-bounce)
        Td = np.zeros((3, 3), dtype=complex)
        Td[1, 1] = 1.0

        # volume (randomly oriented dipoles) -- padrão Freeman: diag(2/3,1/3,1/3)
        Tv = np.zeros((3, 3), dtype=complex)
        Tv[0, 0] = 2.0 / 3.0
        Tv[1, 1] = 1.0 / 3.0
        Tv[2, 2] = 1.0 / 3.0

        return Ts, Td, Tv

def multilook_mean(Timg, window=5):
        """
        Simples média por janela (box filter) sobre cada canal real do vetor 9.
        Timg: (H,W,9) real-valued representation
        window: odd integer
        """
        from scipy.ndimage import uniform_filter
        out = np.empty_like(Timg, dtype=float)
        for k in range(9):
            out[..., k] = uniform_filter(Timg[..., k], size=window, mode='reflect')
        return out



def freeman_decomposition_auto(img, window=None, use_nnls=True):
    """
    img: ndarray (H,W,C) com C >= 9
         Os 9 primeiros canais são:
           [T11, T22, T33,
            Re(T12), Im(T12),
            Re(T13), Im(T13),
            Re(T23), Im(T23)]

    Retorna:
        img_with_freeman: ndarray (H,W,C+3)
            (imagem original + Ps + Pd + Pv)
    """
    H, W, C = img.shape
    assert C >= 9, "A imagem precisa ter pelo menos 9 canais (matriz T)."

    # Extrair sempre os 9 primeiros canais
    Timg_9 = img[..., :9]

    # Multilook opcional
    if window is not None and window > 1:
        Timg_proc = multilook_mean(Timg_9, window=window)
    else:
        Timg_proc = Timg_9

    # Construir modelos Freeman (superfície, duplo, volumétrico)
    Ts, Td, Tv = build_model_matrices()

    # Matriz A (9×3)
    Acols = [_vec_T_matrix(M) for M in (Ts, Td, Tv)]
    A = np.stack(Acols, axis=1)

    # Alocar Ps,Pd,Pv
    Ps = np.zeros((H, W))
    Pd = np.zeros((H, W))
    Pv = np.zeros((H, W))

    # Loop pixel a pixel
    for i in range(H):
        for j in range(W):
            b = Timg_proc[i, j, :]

            if use_nnls and _HAS_NNLS:
                x, _ = nnls(A, b)
            else:
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
                x = np.maximum(x, 0)

            Ps[i, j], Pd[i, j], Pv[i, j] = x

    # Empilhar imagem original + Ps + Pd + Pv
    img_with_freeman = np.concatenate(
        [img, Ps[..., None], Pd[..., None], Pv[..., None]],
        axis=2
    )

    return img_with_freeman

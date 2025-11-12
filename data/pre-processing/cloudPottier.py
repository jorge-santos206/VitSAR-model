import numpy as np

def calcular_entropia(imagem):

    n, m, _ = imagem.shape
    H_map = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            T11 = imagem[i, j, 0]
            T12 = imagem[i, j, 1]
            T13 = imagem[i, j, 2]
            T22 = imagem[i, j, 3]
            T23 = imagem[i, j, 4]
            T33 = imagem[i, j, 5]
            T12c = imagem[i, j, 6]
            T13c = imagem[i, j, 7]
            T23c = imagem[i, j, 8]

            # Constrói a matriz hermitiana 3x3
            T = np.array([
                [T11,  T12,  T13],
                [T12c, T22,  T23],
                [T13c, T23c, T33]
            ], dtype=complex)

            # Garante simetria hermitiana (caso de ruído numérico)
            T = (T + T.conj().T) / 2

            # Calcula autovalores reais e não-negativos
            eigvals, _ = np.linalg.eigh(T)
            eigvals = np.maximum(eigvals, 0)

            soma = np.sum(eigvals)
            if soma == 0:
                H_map[i, j] = 0
                continue

            p = eigvals / soma

            # Entropia de Cloude–Pottier (base 3 → H ∈ [0, 1])
            H = -np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(3)
            H_map[i, j] = H

    return H_map



import numpy as np

def calcular_anisotropia(imagem):
    """
    Calcula o mapa de anisotropia (Cloude–Pottier)
    para uma imagem SAR de shape (n, n, 9),
    onde cada pixel contém os elementos achatados da matriz de coerência 3x3:
    [T11, T12, T13, T22, T23, T33, conj(T12), conj(T13), conj(T23)].
    """
    n, m, _ = imagem.shape
    A_map = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            T11 = imagem[i, j, 0]
            T12 = imagem[i, j, 1]
            T13 = imagem[i, j, 2]
            T22 = imagem[i, j, 3]
            T23 = imagem[i, j, 4]
            T33 = imagem[i, j, 5]
            T12c = imagem[i, j, 6]
            T13c = imagem[i, j, 7]
            T23c = imagem[i, j, 8]

            # Constrói matriz hermitiana 3x3
            T = np.array([
                [T11,  T12,  T13],
                [T12c, T22,  T23],
                [T13c, T23c, T33]
            ], dtype=complex)

            T = (T + T.conj().T) / 2

            # Autovalores (reais e não-negativos)
            eigvals, _ = np.linalg.eigh(T)
            eigvals = np.maximum(eigvals, 0)
            eigvals = np.sort(eigvals)[::-1]  # ordena em ordem decrescente

            # Evita divisão por zero
            if eigvals[1] + eigvals[2] == 0:
                A_map[i, j] = 0
                continue

            # Cálculo da anisotropia
            A = (eigvals[1] - eigvals[2]) / (eigvals[1] + eigvals[2])
            A_map[i, j] = A

    return A_map


import numpy as np

def calcular_angulo_alpha(imagem):
    """
    Calcula o mapa do ângulo médio alpha (Cloude–Pottier)
    para uma imagem SAR de shape (n, n, 9),
    onde cada pixel contém os elementos achatados da matriz de coerência 3x3:
    [T11, T12, T13, T22, T23, T33, conj(T12), conj(T13), conj(T23)].
    Retorna o ângulo médio alpha em radianos (0–π/2).
    """
    n, m, _ = imagem.shape
    alpha_map = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            T11 = imagem[i, j, 0]
            T12 = imagem[i, j, 1]
            T13 = imagem[i, j, 2]
            T22 = imagem[i, j, 3]
            T23 = imagem[i, j, 4]
            T33 = imagem[i, j, 5]
            T12c = imagem[i, j, 6]
            T13c = imagem[i, j, 7]
            T23c = imagem[i, j, 8]

            # Constrói matriz hermitiana
            T = np.array([
                [T11,  T12,  T13],
                [T12c, T22,  T23],
                [T13c, T23c, T33]
            ], dtype=complex)
            T = (T + T.conj().T) / 2

            # Autovalores e autovetores
            eigvals, eigvecs = np.linalg.eigh(T)
            eigvals = np.maximum(eigvals, 0)

            soma = np.sum(eigvals)
            if soma == 0:
                alpha_map[i, j] = 0
                continue

            # Ordena por autovalor decrescente
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            p = eigvals / soma

            # Calcula alpha_i para cada autovetor
            alphas = np.zeros(3)
            for k in range(3):
                e = eigvecs[:, k]
                alphas[k] = np.arctan2(np.abs(e[1]) + np.abs(e[2]), np.abs(e[0]))

            # Ângulo médio ponderado
            alpha_mean = np.sum(p * alphas)
            alpha_map[i, j] = alpha_mean

    return alpha_map

def cloudPottierDecom(imagem):
    entropia= calcular_entropia(imagem)
    angulo= calcular_angulo_alpha(imagem)
    anisotropia=calcular_anisotropia(imagem)
    cloude_pottier = np.stack((entropia, angulo, anisotropia), axis=-1)
    imagem_completa = np.concatenate((imagem, cloude_pottier), axis=-1)
    return np.array(imagem_completa)


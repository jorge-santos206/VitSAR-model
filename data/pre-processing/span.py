import numpy as np

def spanImage(imagem):
    soma = imagem[:, :, :3].sum(axis=2)
    imagem = np.concatenate([imagem, soma[:, :, np.newaxis]], axis=2)
    return np.array(imagem)





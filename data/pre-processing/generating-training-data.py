import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import span as s
import cloudPottier as cloud
with rasterio.open('image-sar1200x900.tif') as dataset:
    sarImage=dataset.read()
    sarImage=np.moveaxis(sarImage,0,-1)
print(sarImage.shape)
sarImage= s.spanImage(sarImage)
sarImage= cloud.cloudPottierDecom(sarImage)

HH=sarImage[:,:,0]
HV=sarImage[:,:,1]
VV=sarImage[:,:,2]
hhhv = sarImage[:,:,3]
hhvv = sarImage[:,:,4]
hvvv = sarImage[:,:,5]
ihhhv = sarImage[:,:,6]
ihhvv = sarImage[:,:,7]
ihvvv = sarImage[:,:,8]



r= np.abs(HH-VV)
g= np.abs(2*HV)
b= np.abs(HH+VV)

rgb= cv2.merge((r,g,b))
rgb=rgb**0.4

#train regions plot
train_region_forest1 = (187, 602, 400, 48)
train_region_forest2=(430,310,240,80)
train_region_ocean=(20,278,80,480)
train_region_cidade=(340,470,480,80)

# image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
# cv2.rectangle(image_rgb, (train_region_forest1[0], train_region_forest1[1]), (train_region_forest1[0] + train_region_forest1[2], train_region_forest1[1] + train_region_forest1[3]), (0, 255, 0), 1)
# cv2.rectangle(image_rgb, (train_region_forest2[0], train_region_forest2[1]), (train_region_forest2[0] + train_region_forest2[2], train_region_forest2[1] + train_region_forest2[3]), (0, 255, 0), 2)
# cv2.rectangle(image_rgb, (train_region_ocean[0], train_region_ocean[1]), (train_region_ocean[0] + train_region_ocean[2], train_region_ocean[1] + train_region_ocean[3]), (0, 255, 0), 2)
# cv2.rectangle(image_rgb, (train_region_cidade[0], train_region_cidade[1]), (train_region_cidade[0] + train_region_cidade[2], train_region_cidade[1] + train_region_cidade[3]), (0, 255, 0), 2)
# plt.imshow(image_rgb)
# plt.axis('off')
# plt.title("imagem da região")
# plt.show()

def colher_amostras(imagem, tamanho):
  h, w = imagem.shape[:2]
  amostras = []

  for y in range(0, h - tamanho + 1, tamanho):
    for x in range(0, w - tamanho + 1, tamanho):
      janela = imagem[y:y + tamanho, x:x + tamanho]
      if janela.shape[0] == tamanho and janela.shape[1] == tamanho:
        amostras.append(janela)

  return np.array(amostras)
imagem_oceano=sarImage [ 278:278+480,20:20+80,:]
imagem_cidade=sarImage [ 470:470+80,340:340+480,:]
imagem_floresta1=sarImage[602:602+48,187:187+400,:]
imagem_floresta2=sarImage[310:310+80,430:430+240,:]
#coleting
amostra_oceano=colher_amostras(imagem_oceano,16)
amostra_cidade=colher_amostras(imagem_cidade,16)
amostra_floresta1=colher_amostras(imagem_floresta1,16)
amostra_floresta2=colher_amostras(imagem_floresta2,16)

for i in range(amostra_cidade.shape[0]):
    filename = f"../dataset/images/sample_{i}.npy"
    np.save(filename, amostra_cidade[i])
    print(f"Salvo: {filename}")

for i in range(amostra_oceano.shape[0]):
    filename = f"../dataset/images/sample_{i+150}.npy"
    np.save(filename, amostra_oceano[i])
    print(f"Salvo: {filename}")


for i in range(amostra_floresta1.shape[0]):
    filename = f"../dataset/images/sample_{i+300}.npy"
    np.save(filename, amostra_floresta1[i])
    print(f"Salvo: {filename}")


for i in range(amostra_floresta2.shape[0]):
    filename = f"../dataset/images/sample_{i+375}.npy"
    np.save(filename, amostra_floresta2[i])
    print(f"Salvo: {filename}")


# generating masks

ocean_masks= np.zeros(amostra_oceano.shape)
cidade_masks=np.ones(amostra_cidade.shape)
forest1_masks=np.full(amostra_floresta1.shape,2)
forest2_masks=np.full(amostra_floresta2.shape,2)

for i in range(cidade_masks.shape[0]):
    filename = f"../dataset/masks/mask_{i}.npy"
    np.save(filename,cidade_masks[i])
    print(f"Salvo: {filename}")

for i in range(ocean_masks.shape[0]):
    filename = f"../dataset/masks/mask_{i+150}.npy"
    np.save(filename, ocean_masks[i])
    print(f"Salvo: {filename}")


for i in range(forest1_masks.shape[0]):
    filename = f"../dataset/masks/mask_{i+300}.npy"
    np.save(filename, forest1_masks[i])
    print(f"Salvo: {filename}")


for i in range(forest2_masks.shape[0]):
    filename = f"../dataset/masks/mask_{i+375}.npy"
    np.save(filename, forest2_masks[i])
    print(f"Salvo: {filename}")


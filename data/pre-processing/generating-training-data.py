from unicodedata import decomposition

import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import span as s
import cloudPottier as cloud
import  freemanDecomposition as free
import crossPolarization as cross
import copolarization as co
import huynenDecomposition as hu
import edgyLineEnergy as edgy
import glcm as glcm
import leeFilter as lee
with rasterio.open('image-sar1200x900.tif') as dataset:
    sarImage=dataset.read()
    sarImage=np.moveaxis(sarImage,0,-1)
print(sarImage.shape)
sarImage=lee.lee_filter_multichannel(sarImage,size=5)
sarImage= s.spanImage(sarImage)
sarImage= cloud.cloudPottierDecom(sarImage)
sarImage= free.freeman_decomposition_auto(sarImage)
sarImage= cross.crosspolarization(sarImage)
sarImage= co.copolarization(sarImage)
sarImage= hu.huynenDecomposition(sarImage)
sarImage=glcm.add_glcm_channels(sarImage)
sarImage= edgy.add_edge_energy_channels(sarImage)
sarImage= edgy.add_line_energy_channels(sarImage)
np.save('sarImage.npy', sarImage)
print(sarImage.shape)
#channels (indexation)
# HH=sarImage[:,:,0]
# HV=sarImage[:,:,1]
# VV=sarImage[:,:,2]
# hhhv = sarImage[:,:,3]
# hhvv = sarImage[:,:,4]
# hvvv = sarImage[:,:,5]
# ihhhv = sarImage[:,:,6]
# ihhvv = sarImage[:,:,7]
# ihvvv = sarImage[:,:,8]
#
#code for train regions
#
# r= np.abs(HH-VV)
# g= np.abs(2*HV)
# b= np.abs(HH+VV)
#
# rgb= cv2.merge((r,g,b))
# rgb=rgb**0.4
#
# #train regions plot
# train_region1 = (120, 570, 160, 160)
# train_region2=(43, 31, 160, 160)
# train_region3=(20, 278, 160, 160)
# train_region4=(540, 500, 160, 160)
# train_region5=(120, 740, 160, 160)
# train_region6=(520, 320, 160, 160)
# train_region7=(320, 550, 160, 160)
# train_region8=(320, 270, 160, 160)
# train_region9=(220, 31, 160, 160)
# train_region10=(760,40,160,160)
# train_region11=(850,460,160,160)
# train_region12=(850,280,160,160)
# train_region13=(500,1,160,160)
# train_region14=(350,740,160,160)
# train_region15=(550,740,160,160)
#
# image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
# cv2.rectangle(image_rgb, (train_region1[0], train_region1[1]), (train_region1[0] + train_region1[2], train_region1[1] + train_region1[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region2[0], train_region2[1]), (train_region2[0] + train_region2[2], train_region2[1] + train_region2[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region3[0], train_region3[1]), (train_region3[0] + train_region3[2], train_region3[1] + train_region3[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region4[0], train_region4[1]), (train_region4[0] + train_region4[2], train_region4[1] + train_region4[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region5[0], train_region5[1]), (train_region5[0] + train_region5[2], train_region5[1] + train_region5[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region6[0], train_region6[1]), (train_region6[0] + train_region6[2], train_region6[1] + train_region6[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region7[0], train_region7[1]), (train_region7[0] + train_region7[2], train_region7[1] + train_region7[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region8[0], train_region8[1]), (train_region8[0] + train_region8[2], train_region8[1] + train_region8[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region9[0], train_region9[1]), (train_region9[0] + train_region9[2], train_region9[1] + train_region9[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region10[0], train_region10[1]), (train_region10[0] + train_region10[2], train_region10[1] + train_region10[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region11[0], train_region11[1]), (train_region11[0] + train_region11[2], train_region11[1] + train_region11[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region12[0], train_region12[1]), (train_region12[0] + train_region12[2], train_region12[1] + train_region12[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region13[0], train_region13[1]), (train_region13[0] + train_region13[2], train_region13[1] + train_region13[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb, (train_region14[0], train_region14[1]), (train_region14[0] + train_region14[2], train_region14[1] + train_region14[3]), (0, 255, 0), 3)
# cv2.rectangle(image_rgb,(train_region15[0], train_region15[1]), (train_region15[0] + train_region15[2], train_region15[1] + train_region15[3]), (0, 255, 0), 3)
#
# plt.imshow(image_rgb)
# plt.axis('off')
# plt.title("imagem da região")
# plt.show()
#

#code for samples and masks bellow


def colher_amostras(imagem, tamanho):
  h, w = imagem.shape[:2]
  amostras = []

  for y in range(0, h - tamanho + 1, tamanho):
    for x in range(0, w - tamanho + 1, tamanho):
      janela = imagem[y:y + tamanho, x:x + tamanho]
      if janela.shape[0] == tamanho and janela.shape[1] == tamanho:
        amostras.append(janela)

  return np.array(amostras)



imagem1 =sarImage[ 570:570+160,120:120+160,:]
imagem2 =sarImage[ 31:31+160,43:43+160,:]
imagem3 =sarImage[ 278:278+160,20:20+160,:]
imagem4 =sarImage[ 400:400+160,740:740+160,:]
imagem5 =sarImage[ 740:740+160,120:120+160,:]
imagem6 =sarImage[ 320:320+160,550:550+160,:]
imagem7 =sarImage[ 550:550+160,320:320+160,:]
imagem8 =sarImage[ 270:270+160,320:320+160,:]
imagem9 =sarImage[ 31:31+160,220:220+160,:]
imagem10=sarImage[ 40:40+160,760:760+160,:]
imagem11=sarImage[560:560+160,850+160:850+160,:]
imagem12=sarImage[280:280+160,850:850+160,:]
imagem13=sarImage[1:1+160,500:500+160,:]
imagem14=sarImage[740:740+160,350:350+160,:]
imagem15=sarImage[740:740+160,550:550+160,:]


#coleting
amostra1=colher_amostras(imagem1,16)
amostra2=colher_amostras(imagem2,16)
amostra3=colher_amostras(imagem3,16)
amostra4=colher_amostras(imagem4,16)
amostra5=colher_amostras(imagem5,16)
amostra6=colher_amostras(imagem6,16)
amostra7=colher_amostras(imagem7,16)
amostra8=colher_amostras(imagem8,16)
amostra9=colher_amostras(imagem9,16)
amostra10=colher_amostras(imagem10,16)
amostra11=colher_amostras(imagem11,16)
amostra12=colher_amostras(imagem12,16)
amostra13=colher_amostras(imagem13,16)
amostra14=colher_amostras(imagem14,16)
amostra15=colher_amostras(imagem15,16)

for i in range(amostra1.shape[0]):
    filename = f"../dataset/images/sample_{i}.npy"
    np.save(filename, amostra1[i])
    print(f"Salvo: {filename}")

for i in range(amostra2.shape[0]):
    filename = f"../dataset/images/sample_{i+100}.npy"
    np.save(filename, amostra2[i])
    print(f"Salvo: {filename}")


for i in range(amostra3.shape[0]):
    filename = f"../dataset/images/sample_{i+200}.npy"
    np.save(filename, amostra3[i])
    print(f"Salvo: {filename}")


for i in range(amostra4.shape[0]):
    filename = f"../dataset/images/sample_{i+300}.npy"
    np.save(filename, amostra4[i])
    print(f"Salvo: {filename}")

for i in range(amostra5.shape[0]):
    filename = f"../dataset/images/sample_{i+400}.npy"
    np.save(filename, amostra5[i])
    print(f"Salvo: {filename}")

for i in range(amostra6.shape[0]):
    filename = f"../dataset/images/sample_{i+500}.npy"
    np.save(filename, amostra6[i])
    print(f"Salvo: {filename}")


for i in range(amostra7.shape[0]):
    filename = f"../dataset/images/sample_{i+600}.npy"
    np.save(filename, amostra7[i])
    print(f"Salvo: {filename}")


for i in range(amostra8.shape[0]):
    filename = f"../dataset/images/sample_{i+700}.npy"
    np.save(filename, amostra8[i])
    print(f"Salvo: {filename}")


for i in range(amostra9.shape[0]):
    filename = f"../dataset/images/sample_{i+800}.npy"
    np.save(filename, amostra9[i])
    print(f"Salvo: {filename}")


for i in range(amostra10.shape[0]):
    filename = f"../dataset/images/sample_{i+900}.npy"
    np.save(filename, amostra10[i])
    print(f"Salvo: {filename}")



for i in range(amostra11.shape[0]):
    filename = f"../dataset/images/sample_{i+1000}.npy"
    np.save(filename, amostra11[i])
    print(f"Salvo: {filename}")




for i in range(amostra12.shape[0]):
    filename = f"../dataset/images/sample_{i+1100}.npy"
    np.save(filename, amostra12[i])
    print(f"Salvo: {filename}")



for i in range(amostra13.shape[0]):
    filename = f"../dataset/images/sample_{i+1200}.npy"
    np.save(filename, amostra13[i])
    print(f"Salvo: {filename}")



for i in range(amostra14.shape[0]):
    filename = f"../dataset/images/sample_{i+1300}.npy"
    np.save(filename, amostra14[i])
    print(f"Salvo: {filename}")



for i in range(amostra15.shape[0]):
    filename = f"../dataset/images/sample_{i+1400}.npy"
    np.save(filename, amostra15[i])
    print(f"Salvo: {filename}")


#reading true labels
mapa_cores = {
    (0, 0, 0): 0,       # background
    (0, 0, 255): 1,     # classe 1
    (0, 255, 0): 2,     # classe 2
    (0, 255, 255): 3,   # classe 3
    (255, 0, 0): 4,     # classe 4
    (255, 255, 0): 5    # classe 5
}



mask = cv2.imread("SF-AIRSAR-label3d.png")  # BGR

mask_classe = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

for cor, classe in mapa_cores.items():
    match = np.all(mask == cor, axis=-1)
    mask_classe[match] = classe


# generating masks


mask1 =mask_classe[ 570:570+160,120:120+160]
print(mask1.shape)
mask2 =mask_classe[ 31:31+160,43:43+160]
mask3 =mask_classe[ 278:278+160,20:20+160]
mask4 =mask_classe[ 400:400+160,740:740+160]
mask5 =mask_classe[ 740:740+160,120:120+160]
mask6 =mask_classe[ 320:320+160,550:550+160]
mask7 =mask_classe[ 550:550+160,320:320+160]
mask8 =mask_classe[ 270:270+160,320:320+160]
mask9 =mask_classe[ 31:31+160,220:220+160]
mask10 =mask_classe[ 40:40+160,760:760+160]
mask11=mask_classe[560:560+160,850+160:850+160]
mask12=mask_classe[280:280+160,850:850+160]
mask13=mask_classe[1:1+160,500:500+160]
mask14=mask_classe[740:740+160,350:350+160]
mask15=mask_classe[740:740+160,550:550+160]
amostra11=colher_amostras(mask1,16)
amostra22=colher_amostras(mask2,16)
amostra33=colher_amostras(mask3,16)
amostra44=colher_amostras(mask4,16)
amostra55=colher_amostras(mask5,16)
amostra66=colher_amostras(mask6,16)
amostra77=colher_amostras(mask7,16)
amostra88=colher_amostras(mask8,16)
amostra99=colher_amostras(mask9,16)
amostra1010=colher_amostras(mask10,16)
amostra1011=colher_amostras(mask11,16)
amostra1012=colher_amostras(mask12,16)
amostra1013=colher_amostras(mask13,16)
amostra1014=colher_amostras(mask14,16)
amostra1015=colher_amostras(mask15,16)

print(mask9.shape)

for i in range(amostra11.shape[0]):
    filename = f"../dataset/masks/mask_{i}.npy"
    np.save(filename, amostra11[i])
    print(f"Salvo: {filename}")


for i in range(amostra22.shape[0]):
    filename = f"../dataset/masks/mask_{i+100}.npy"
    np.save(filename, amostra22[i])
    print(f"Salvo: {filename}")


for i in range(amostra33.shape[0]):
    filename = f"../dataset/masks/mask_{i+200}.npy"
    np.save(filename,amostra33[i])
    print(f"Salvo: {filename}")

for i in range(amostra44.shape[0]):
    filename = f"../dataset/masks/mask_{i+300}.npy"
    np.save(filename, amostra44[i])
    print(f"Salvo: {filename}")

#
for i in range(amostra55.shape[0]):
    filename = f"../dataset/masks/mask_{i+400}.npy"
    np.save(filename,amostra55[i])
    print(f"Salvo: {filename}")

for i in range(amostra66.shape[0]):
    filename = f"../dataset/masks/mask_{i+500}.npy"
    np.save(filename, amostra66[i])
    print(f"Salvo: {filename}")


for i in range(amostra77.shape[0]):
    filename = f"../dataset/masks/mask_{i+600}.npy"
    np.save(filename, amostra77[i])
    print(f"Salvo: {filename}")


for i in range(amostra88.shape[0]):
    filename = f"../dataset/masks/mask_{i+700}.npy"
    np.save(filename, amostra88[i])
    print(f"Salvo: {filename}")

for i in range(amostra99.shape[0]):
    filename = f"../dataset/masks/mask_{i+800}.npy"
    np.save(filename, amostra99[i])
    print(f"Salvo: {filename}")



for i in range(amostra1010.shape[0]):
    filename = f"../dataset/masks/mask_{i+900}.npy"
    np.save(filename, amostra1010[i])
    print(f"Salvo: {filename}")


for i in range(amostra1011.shape[0]):
    filename = f"../dataset/masks/mask_{i+1000}.npy"
    np.save(filename, amostra1011[i])
    print(f"Salvo: {filename}")

for i in range(amostra1012.shape[0]):
    filename = f"../dataset/masks/mask_{i + 1100}.npy"
    np.save(filename, amostra1012[i])
    print(f"Salvo: {filename}")

for i in range(amostra1013.shape[0]):
    filename = f"../dataset/masks/mask_{i + 1200}.npy"
    np.save(filename, amostra1013[i])
    print(f"Salvo: {filename}")


for i in range(amostra1014.shape[0]):
    filename = f"../dataset/masks/mask_{i + 1300}.npy"
    np.save(filename, amostra1014[i])
    print(f"Salvo: {filename}")

for i in range(amostra1015.shape[0]):
    filename = f"../dataset/masks/mask_{i + 1400}.npy"
    np.save(filename, amostra1015[i])
    print(f"Salvo: {filename}")















library(terra)
load("~/Documentos/Vit-SAR/VitSAR-model/data/AirSAR_SanFrancisc_Enxu.RData")

canal1 <- San_Francisc_Enxuto[,,1]  
canal2 <- San_Francisc_Enxuto[,,2]  
canal3 <- San_Francisc_Enxuto[,,3]  
canal4 <- San_Francisc_Enxuto[,,4]
canal5 <- San_Francisc_Enxuto[,,5]  
canal6 <- San_Francisc_Enxuto[,,6]  
canal7 <- San_Francisc_Enxuto[,,7]  
canal8 <- San_Francisc_Enxuto[,,8]  
canal9 <- San_Francisc_Enxuto[,,9]  
par(mfrow=c(3,3))

# Plotando as imagens com valores elevados a 0.01
image(canal1^0.1, main = "HH", col = gray.colors(256)) 
image(canal2^0.1, main = "VV", col = gray.colors(256))  
image(canal3^0.1, main = "HV", col = gray.colors(256)) 
image(canal4^0.1, main = "HHHV", col = gray.colors(256)) 
image(canal5^0.1, main = "HHVVV", col = gray.colors(256)) 
image(canal6^0.1, main = "HVVV", col = gray.colors(256)) 
image(canal7^0.1, main = "iHVVV", col = gray.colors(256)) 
image(canal8^0.1, main = "iHVVV", col = gray.colors(256)) 
image(canal9^0.1, main = "iHVVV", col = gray.colors(256)) 

raster1 <- rast(canal1)  
raster2 <- rast(canal2)  
raster3 <- rast(canal3)  
raster4 <- rast(canal4)
raster5 <- rast(canal5)
raster6 <- rast(canal6) 
raster7 <- rast(canal7)
raster8 <- rast(canal8)
raster9 <- rast(canal9)

sar_image <- c(raster1,raster2,raster3,raster4,raster5,raster6,raster7,raster8,raster9)


writeRaster(sar_image, "image-sar1200x900.tif", overwrite=TRUE)

cat("Imagem SAR gerada 'image-sar1200x900.tiff")




import keras
from keras import layers
from keras import ops
import random
import os
import numpy as np
from glob import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
SEED = 42
# Python
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED)



IMAGE_SIZE = 16
NUM_CHANNELS = 51
BATCH_SIZE = 2
NUM_CLASSES = 3
DATA_DIR = "../data/dataset"
NUM_TRAIN_IMAGES = 450
NUM_VAL_IMAGES = 50

train_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "masks/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "masks/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]

def read_image(image_path, mask=False):
    # Carrega o arquivo .npy (usando numpy)
    image = np.load(image_path.decode())  # decode() converte string tf -> str Python

    if mask:
        # Máscaras geralmente têm 1 canal (classes)
        if image.ndim == 3:
            image = image[..., 0]  # garante canal único
        image = np.expand_dims(image, axis=-1)
    else:
        # Garante número de canais consistente
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        elif image.ndim == 3 and image.shape[-1] != NUM_CHANNELS:
            raise ValueError(f"Esperado {NUM_CHANNELS} canais, mas imagem tem {image.shape[-1]}.")

    # Redimensiona para IMAGE_SIZE x IMAGE_SIZE
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    # Converte para float32 (necessário para TensorFlow)
    image = tf.cast(image, tf.float32)

    return image


def load_data(image_path, mask_path):
    # Usa tf.numpy_function para integrar np.load no pipeline TF
    image, mask = tf.numpy_function(
        func=lambda x, y: (read_image(x, mask=False), read_image(y, mask=True)),
        inp=[image_path, mask_path],
        Tout=[tf.float32, tf.float32],
    )

    # Define as formas esperadas (ajuda o TensorFlow a saber o tamanho)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    mask.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])

    return image, mask


def data_generator(image_list, mask_list, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)

    return ops.nn.relu(x)
def backbone(backbone_input):
    x= convolution_block(
        backbone_input,
        num_filters=256,
        kernel_size=5
    )
    x=layers.MaxPooling2D(pool_size=(2,2))(x)
    x = convolution_block(x, num_filters=256, kernel_size=3)

    return x


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=3, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=2)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=4)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output




def decoder(x):
    # refinamento após concat
    x = convolution_block(
        x,
        num_filters=256,
        kernel_size=3
    )

    # upsampling
    # x = layers.UpSampling2D(
    #     size=(2, 2),
    #     interpolation="bilinear"
    # )(x)

    return x


def attention_block(x, reduction_ratio=8):
    """
    Channel Attention conforme descrito no artigo.
    x: tensor (H, W, C)
    """
    channels = x.shape[-1]

    # Global Average Pooling
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channels))(avg_pool)

    # Global Max Pooling
    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channels))(max_pool)

    # Shared MLP
    shared_dense_1 = layers.Dense(
        channels // reduction_ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True
    )

    shared_dense_2 = layers.Dense(
        channels,
        kernel_initializer="he_normal",
        use_bias=True
    )

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))

    # Soma + Sigmoid
    attention = layers.Add()([avg_out, max_out])
    attention = layers.Activation("sigmoid")(attention)

    # Reponderação canal a canal
    output = layers.Multiply()([x, attention])

    return output



def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, NUM_CHANNELS))

    # Encoder / Backbone
    features = backbone(model_input)

    # Low-level features
    branch_direct = convolution_block(
        features,
        num_filters=48,
        kernel_size=1,
        use_bias=True
    )

    # ASPP
    branch_aspp = DilatedSpatialPyramidPooling(features)
    branch_aspp = convolution_block(
        branch_aspp,
        num_filters=256,
        kernel_size=1,
        use_bias=True
    )

    # Upsampling
    up_factor = image_size // branch_aspp.shape[1]

    branch_aspp = layers.UpSampling2D(
        size=(up_factor, up_factor),
        interpolation="bilinear"
    )(branch_aspp)

    branch_direct = layers.UpSampling2D(
        size=(up_factor, up_factor),
        interpolation="bilinear"
    )(branch_direct)

    # Concat
    x = layers.Concatenate(axis=-1)([branch_aspp, branch_direct])

    # Decoder
    x = decoder(x)

    # 🔥 ATENÇÃO (artigo)
    x = attention_block(x)

    # Classificador final (conv 1x1)
    output = layers.Conv2D(
        num_classes,
        kernel_size=1,
        padding="same",
        activation="softmax"
    )(x)
    return keras.Model(model_input, output)

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.summary()


loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)



history = model.fit(train_dataset, validation_data=val_dataset, epochs=54)

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

model.save("deeplabv3+.keras")

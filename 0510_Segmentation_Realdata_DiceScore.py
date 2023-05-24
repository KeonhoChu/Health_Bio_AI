
import numpy as np
import os
import cv2
from os import walk
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
import matplotlib.pyplot as plt

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    return loss

# Load the dataset and preprocess the images and masks
data_path = "C:\\Users\\CHUUUU\\Desktop\\dataset\\"
image_path = os.path.join(data_path, "images")
mask_path = os.path.join(data_path, "masks")

# Load the images and masks
images = []
masks = []
count = 0
for path, subdirs, files in os.walk(image_path):
    for name in files:
        imagepath = os.path.join(path, name)
        maskpath = imagepath.replace('images', 'masks').replace('tiff', 'png')

        print(imagepath + "  " + maskpath)
        # Load the image
        image = Image.open(imagepath)
        image = np.array(image)
        image = cv2.resize(image, (256, 256))

        # Load the mask
        mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)

        # print(np.unique(mask)) # 0 76 149
        mask[mask != 0] = 1
        # mask[mask == 0] = 0
        # mask[mask == 76] = 1

        mask = cv2.resize(mask, (256, 256), interpolation=None)

        images.append(image)
        masks.append(mask)

# Convert the lists to numpy arrays
images = np.array(images)
masks = np.array(masks)
# masks = np.array(masks)

# GT = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 2))
# for i in range(2):
#     GT[:, :, :, i] = (masks == i)
print("%d images and %d masks detected" % (len(images), len(masks)))

# N = 5
# for i in range(N):
#     image = images[i]
#     GT0 = GT[i, :, :, 0]
#     GT1 = GT[i, :, :, 1]
#     GT2 = GT[i, :, :, 2]
#
#     plt.subplot(4, N, i + 1)
#     plt.imshow(image, cmap='gray')
#
#     plt.subplot(4, N, N + i + 1)
#     plt.imshow(GT0, cmap='gray')
#
#     plt.subplot(4, N, 2 * N + i + 1)
#     plt.imshow(GT1, cmap='gray')
#
#     plt.subplot(4, N, 3 * N + i + 1)
#     plt.imshow(GT2, cmap='gray')
# Normalize the images
images = images / 255.0

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)



def unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Contracting Path (Encoder)
    conv1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Expansive Path (Decoder)
    up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(16, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same')(conv9)

    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
model = unet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=dice_loss,
              metrics=['accuracy', dice_coefficient])


model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=16)
model.save('brain_mri_v1.h5')

preds = model.predict(x_val)
fig, ax = plt.subplots(1,2)
ax[0].imshow(x_val[0])
ax[1].imshow(preds[0])
plt.show()
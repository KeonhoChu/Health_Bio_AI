import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

pal = sns.color_palette('Set3')
sns.set_palette(pal)

# 해당 경로에 있는 모든 DWI 이미지의 tiff 파일 경로 가져오기.
file_path = glob.glob('C:/Users/CHUUUU/Desktop/dataset/images/*DWI*.tiff')
mans = []

# 이미지 파일명에서 환자 번호를 추출.
for i in file_path:
    if int(i[39:42]) not in mans:
        mans.append(int(i[39:42]))  # 해당 환자 번호가 mans 리스트에 있으면 append 시키지 않음.


thres = len(mans) * 0.15     # train과 test를 나누기 위한 threshlod 값.

# train test 공간을 미리 만들어 줌.
adcs_train = []
adcs_test = []
dwis_train = []
dwis_test = []
masks_train = []
masks_test = []

# file_path에 있는 경로를 하나씩 image에 넣어줌.
for image in file_path:
    # 환자번호를 정수형으로 변환.
    idx = int(image[39:42])

    # DWI 이미지 파일 경로에서 ADC 이미지 파일 경로 추출.
    adc_image = image.replace('DWI', 'ADC')

    # DWI 이미지 파일 경로에서 mask 이미지 파일 경로 추출.
    png_image = image.replace('images','masks').replace('tiff', 'png')

    # 각 이미지를 불러옴.
    dwi = cv2.imread(image, flags=cv2.IMREAD_UNCHANGED)
    dwi = cv2.resize(dwi, (256, 256))
    adc = cv2.imread(adc_image, flags=cv2.IMREAD_UNCHANGED)
    adc = cv2.resize(adc, (256, 256))
    mask = cv2.imread(png_image)
    mask = cv2.resize(mask, (256, 256))

    # mask 이미지를 BGR순에서 RGB순으로 변환.
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # threshold 값으로 train test 분할.
    if thres < mans.index(idx):
        adcs_train.append(adc)
        dwis_train.append(dwi)
        masks_train.append(mask)
    else:
        adcs_test.append(adc)
        dwis_test.append(dwi)
        masks_test.append(mask)

# numpy로 변환.
adcs_train = np.array(adcs_train)
dwis_train = np.array(dwis_train)
adcs_test = np.array(adcs_test)
dwis_test = np.array(dwis_test)
y_train = np.array(masks_train)
y_test = np.array(masks_test)
del masks_train
del masks_test


# DWI 이미지와 ADC 이미지를 2채널의 이미지로 변경
x_train = np.stack((adcs_train,dwis_train), axis=3) / [255, 255]
del adcs_train
del dwis_train

x_test = np.stack((adcs_test,dwis_test), axis=3) / [255, 255]
del adcs_test
del dwis_test

# mask 이미지에서 R채널만 가져옴
y_train = y_train[:,:,:,0] / 255
y_test = y_test[:,:,:,0] / 255

# 각 데이터의 크기가 맞는지 확인
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


def unet(input_size=(256, 256, 2)):
  inputs = Input(input_size)

  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)

  up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
  merge6 = concatenate([drop4, up6], axis=3)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

  up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
  merge7 = concatenate([conv3, up7], axis=3)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

  up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
  merge8 = concatenate([conv2, up8], axis=3)

  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

  up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
  merge9 = concatenate([conv1, up9], axis=3)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
  conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
  outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

  model = Model(inputs=inputs, outputs=outputs)

  return model

model = unet()
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=20)

preds = model.predict(x_test)

fig, ax = plt.subplots(1,3)
ax[0].imshow(x_test[0,:,:,0], cmap='gray')
ax[1].imshow(x_test[1,:,:,1], cmap='gray')
ax[2].imshow(preds[0], cmap='gray')
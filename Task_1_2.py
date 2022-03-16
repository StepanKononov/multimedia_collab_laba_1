from skimage.io import imread, imshow
from skimage import img_as_float
import matplotlib.pyplot as plt
import numpy as np

img = imread('lenna.jpg')

imshow(img)
plt.show()

img_f = img_as_float(img)

R = img_f[:, :, 0]
G = img_f[:, :, 1]
B = img_f[:, :, 2]

Y = R * 0.299 + G * 0.587 + B * 0.114
Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128


def chroma_subsampling_1(or_im):
    temp_im = [[np.nan for i in range(or_im.shape[1] // 2)] for j in range(or_im.shape[0] // 2)]

    for j in range(0, or_im.shape[0], 2):
        for i in range(0, or_im.shape[1], 2):
            temp_im[j // 2][i // 2] = or_im[j][i]

    temp_im = np.array(temp_im)
    return temp_im


def recovery_img(or_im):
    temp_im = [[np.nan for i in range(or_im.shape[1] * 2)] for j in range(or_im.shape[0] * 2)]
    for j in range(0, or_im.shape[0] * 2, 2):
        for i in range(0, or_im.shape[1] * 2, 2):
            temp_im[j][i] = or_im[j // 2][i // 2]

            b_1 = j + 1 < len(temp_im)
            b_2 = i + 1 < len(temp_im[0])
            b_3 = b_1 and b_2
            if b_1:
                temp_im[j + 1][i] = or_im[j // 2][i // 2]
            if b_2:
                temp_im[j][i + 1] = or_im[j // 2][i // 2]
            if b_3:
                temp_im[j + 1][i + 1] = or_im[j // 2][i // 2]

    temp_im = np.array(temp_im)

    return temp_im


Cr_1 = recovery_img(chroma_subsampling_1(Cr))
Cb_1 = recovery_img(chroma_subsampling_1(Cb))

R = Y + 1.402 * (Cr_1 - 128)

G = Y - 0.34414 * (Cb_1 - 128) - 0.71414 * (Cr_1 - 128)

B = Y + 1.772 * (Cb_1 - 128)

imshow(np.dstack((R, G, B)))

plt.show()
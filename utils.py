import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_images(images):  # 定义画图工具
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        img = img.reshape((3, 28, 28))
        img = img.transpose(1, 2, 0)
        plt.imshow(img.astype(np.uint8))
    return


def preprocess_img(x):
    face_width = face_height = 108
    i = (x.size[1] - face_height) // 2
    j = (x.size[0] - face_width) // 2
    x = x.crop([j, i, j + face_width, i + face_width])

    transform = tfs.Compose([
        tfs.Resize((28, 28)),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(x)


def deprocess_img(x):
    x = (x + 1.0) / 2.0 * 255
    return x.clip(0, 255)

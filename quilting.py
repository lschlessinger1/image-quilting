import matplotlib.pyplot as plt
import numpy as np

texture_1 = plt.imread('data/texture1.jpg')


def synth_texture(texture, block_size):
    h, w, c = texture.shape
    assert block_size < min(h, w)

    y_max, x_max = h - block_size, w - block_size

    # desired size of new image is twice original one
    dh = h * 2
    dw = w * 2

    nx_blocks = ny_blocks = max(dh, dw) // block_size
    w_new = h_new = nx_blocks * block_size

    n_blocks = nx_blocks * ny_blocks

    texture_img = np.zeros((h_new, w_new, c), dtype=texture.dtype)

    # Choose random blocks
    xs = np.random.randint(0, x_max, size=n_blocks)
    ys = np.random.randint(0, y_max, size=n_blocks)
    ind = np.vstack((xs, ys)).T

    blocks = np.array([texture_1[y:y + block_size, x:x + block_size] for x, y in ind])

    b = 0
    for y in range(ny_blocks):
        for x in range(nx_blocks):
            x1, y1 = x * block_size, y * block_size
            x2, y2 = x1 + block_size, y1 + block_size
            texture_img[y1:y2, x1:x2] = blocks[b]
            b += 1

    return texture_img


def show_fig2a(texture_img):
    plt.title('Figure 2a')
    plt.imshow(texture_img)
    plt.axis('off')
    plt.show()


show_fig2a(synth_texture(texture_1, block_size=75))

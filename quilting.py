import matplotlib.pyplot as plt
import numpy as np

texture_1 = plt.imread('data/texture1.jpg')


def synth_texture_rand(texture, block_size):
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


def synth_texture_neighborhood(texture, block_size):
    h, w, c = texture.shape

    assert block_size < min(h, w)

    # desired size of new image is twice original one
    dh = h * 2
    dw = w * 2

    y_max, x_max = h - block_size, w - block_size
    nx_blocks = ny_blocks = max(dh, dw) // block_size
    w_new = h_new = nx_blocks * block_size

    xs = np.arange(x_max)
    ys = np.arange(y_max)
    all_blocks = np.array([texture_1[y:y + block_size, x:x + block_size] for x in xs for y in ys])

    # pad with block_size zeros
    texture_img = np.zeros((h_new + 2 * block_size, w_new + 2 * block_size, c), dtype=texture_1.dtype)

    # place block
    # neighborhood size
    n_rc = 10
    for y in range(ny_blocks):
        for x in range(nx_blocks):
            x1, y1 = x * block_size + block_size, y * block_size + block_size
            x2, y2 = x1 + block_size, y1 + block_size

            # search for block in all_blocks that minimized the cost
            top_cost = np.sum((all_blocks[:, :n_rc, :, :] - texture_img[y1 - n_rc:y1, x1:x2]) ** 2, axis=(1, 2, 3))
            left_cost = np.sum((all_blocks[:, :, :n_rc, :] - texture_img[y1:y2, x1 - n_rc:x1]) ** 2, axis=(1, 2, 3))

            total_cost = top_cost + left_cost

            min_block = all_blocks[np.argmin(total_cost)]
            texture_img[y1:y2, x1:x2] = min_block

    # get rid of padding
    texture_img = texture_img[block_size:h_new + block_size, block_size:w_new + block_size]
    return texture_img


def show_fig2a(texture_img):
    plt.title('Figure 2a')
    plt.imshow(texture_img)
    plt.axis('off')
    plt.show()


def show_fig2b(texture_img):
    plt.title('Figure 2b')
    plt.imshow(texture_img)
    plt.axis('off')
    plt.show()


blk_size = 75
show_fig2a(synth_texture_rand(texture_1, block_size=blk_size))
show_fig2b(synth_texture_neighborhood(texture_1, block_size=blk_size))

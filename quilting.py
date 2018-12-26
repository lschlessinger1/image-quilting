import matplotlib.pyplot as plt
import numpy as np


# normalize_img normalizes our output to be between 0 and 1
def normalize_img(im):
    img = im.copy()
    img += np.abs(np.min(img))
    img /= np.max(img)
    return img


def l2_top_bottom(patch_top, patch_bottom):
    block_top = patch_top[-overlap_size:, :]

    if patch_bottom.ndim == 3:
        block_bottom = patch_bottom[:overlap_size]
    elif patch_bottom.ndim == 4:
        block_bottom = patch_bottom[:, :overlap_size]
    else:
        raise ValueError('patch_right must have 3 or 4 dimensions')

    top_cost = l2_loss(block_top, block_bottom)

    return top_cost


def l2_left_right(patch_left, patch_right):
    block_left = patch_left[:, -overlap_size:]

    if patch_right.ndim == 3:
        block_right = patch_right[:, :overlap_size]
    elif patch_right.ndim == 4:
        block_right = patch_right[:, :, :overlap_size]
    else:
        raise ValueError('patch_right must have 3 or 4 dimensions')

    left_cost = l2_loss(block_left, block_right)

    return left_cost


def l2_loss(block_1, block_2):
    sqdfs = np.sum((block_1 - block_2) ** 2, axis=-1)
    return np.sqrt(np.sum(np.sum(sqdfs, axis=-1), axis=-1))


def select_min_patch(patches, cost):
    return patches[np.argmin(cost)]


def select_min_patch_tol(patches, cost):
    min_cost = np.min(cost)
    tolerance = 0.1
    upper_cost_bound = min_cost + tolerance * min_cost
    # pick random patch within tolerance
    patch = patches[np.random.choice(np.argwhere(cost <= upper_cost_bound).flatten())]
    return patch


def compute_error_surface(block_1, block_2):
    return np.sum((block_1 - block_2) ** 2, axis=-1)


def min_vert_path(error_surf_vert):
    top_min_path = np.zeros(block_size, dtype=np.int)
    top_min_path[0] = np.argmin(error_surf_vert[0, :], axis=0)
    for i in range(1, block_size):
        err_mid_v = error_surf_vert[i, :]
        mid_v = err_mid_v[top_min_path[i - 1]]

        err_left = np.roll(error_surf_vert[i, :], 1, axis=0)
        err_left[0] = np.inf
        left = err_left[top_min_path[i - 1]]

        err_right = np.roll(error_surf_vert[i, :], -1, axis=0)
        err_right[-1] = np.inf
        right = err_right[top_min_path[i - 1]]

        next_poss_pts_v = np.vstack((left, mid_v, right))
        new_pts_ind_v = top_min_path[i - 1] + (np.argmin(next_poss_pts_v, axis=0) - 1)
        top_min_path[i] = new_pts_ind_v

    return top_min_path


def min_hor_path(error_surf_hor):
    left_min_path = np.zeros(block_size, dtype=np.int)
    left_min_path[0] = np.argmin(error_surf_hor[:, 0], axis=0)
    for i in range(1, block_size):
        err_mid_h = error_surf_hor[:, i]
        mid_h = err_mid_h[left_min_path[i - 1]]

        err_top = np.roll(error_surf_hor[:, i], 1, axis=0)
        err_top[0] = np.inf
        top = err_top[left_min_path[i - 1]]

        err_bot = np.roll(error_surf_hor[:, i], -1, axis=0)
        err_bot[-1] = np.inf
        bot = err_bot[left_min_path[i - 1]]

        next_poss_pts_h = np.vstack((top, mid_h, bot))
        new_pts_ind_h = left_min_path[i - 1] + (np.argmin(next_poss_pts_h, axis=0) - 1)
        left_min_path[i] = new_pts_ind_h

    return left_min_path


def compute_lr_join(block_left, block_right, error_surf_vert=None):
    if error_surf_vert is None:
        error_surf_vert = compute_error_surface(block_right, block_left)

    vert_path = min_vert_path(error_surf_vert)
    yy, xx = np.meshgrid(np.arange(block_size), np.arange(overlap_size))
    vert_mask = xx.T <= np.tile(np.expand_dims(vert_path, 1), overlap_size)

    lr_join = np.zeros_like(block_left)
    lr_join[:, :][vert_mask] = block_left[vert_mask]
    lr_join[:, :][~vert_mask] = block_right[~vert_mask]

    return lr_join


def compute_bt_join(block_top, block_bottom, error_surf_hor=None):
    if error_surf_hor is None:
        error_surf_hor = compute_error_surface(block_bottom, block_top)

    hor_path = min_hor_path(error_surf_hor)
    yy, xx = np.meshgrid(np.arange(block_size), np.arange(overlap_size))
    hor_mask = (xx.T <= np.tile(np.expand_dims(hor_path, 1), overlap_size)).T

    bt_join = np.zeros_like(block_top)
    bt_join[:, :][hor_mask] = block_top[hor_mask]
    bt_join[:, :][~hor_mask] = block_bottom[~hor_mask]

    return bt_join


def lr_bt_join_double(best_left_block, right_block, best_top_block, bottom_block):
    error_surf_hor = compute_error_surface(best_left_block, right_block)
    error_surf_vert = compute_error_surface(best_top_block, bottom_block)

    vert_contrib = np.zeros_like(error_surf_vert)
    hor_contrib = np.zeros_like(error_surf_hor)

    vert_contrib[:, :overlap_size] += (error_surf_hor[:overlap_size, :] + error_surf_vert[:, :overlap_size]) / 2
    hor_contrib[:overlap_size, :] += (error_surf_vert[:, :overlap_size] + error_surf_hor[:overlap_size, :]) / 2

    error_surf_vert += vert_contrib
    error_surf_hor += hor_contrib

    left_right_join = compute_lr_join(right_block, best_left_block, error_surf_vert=error_surf_hor)
    bottom_top_join = compute_bt_join(bottom_block, best_top_block, error_surf_hor=error_surf_vert)

    return left_right_join, bottom_top_join


def synth_texture_rand(texture, blk_size):
    h, w, c = texture.shape
    assert blk_size < min(h, w)

    y_max, x_max = h - blk_size, w - blk_size

    # desired size of new image is twice original one
    dh = h * 2
    dw = w * 2

    nx_blocks = ny_blocks = max(dh, dw) // blk_size
    w_new = h_new = nx_blocks * blk_size

    n_blocks = nx_blocks * ny_blocks

    texture_img = np.zeros((h_new, w_new, c), dtype=texture.dtype)

    # Choose random blocks
    xs = np.random.randint(0, x_max, size=n_blocks)
    ys = np.random.randint(0, y_max, size=n_blocks)
    ind = np.vstack((xs, ys)).T

    blocks = np.array([texture_1[y:y + blk_size, x:x + blk_size] for x, y in ind])

    b = 0
    for y in range(ny_blocks):
        for x in range(nx_blocks):
            x1, y1 = x * blk_size, y * blk_size
            x2, y2 = x1 + blk_size, y1 + blk_size
            texture_img[y1:y2, x1:x2] = blocks[b]
            b += 1

    return texture_img


def synth_texture_neighborhood(texture, blk_size):
    h, w, c = texture.shape

    assert blk_size < min(h, w)

    # desired size of new image is twice original one
    dh = h * 2
    dw = w * 2

    y_max, x_max = h - blk_size, w - blk_size
    nx_blocks = ny_blocks = max(dh, dw) // blk_size
    w_new = h_new = nx_blocks * blk_size - (nx_blocks - 1) * overlap_size

    xs = np.arange(x_max)
    ys = np.arange(y_max)
    all_blocks = np.array([texture_1[y:y + blk_size, x:x + blk_size] for x in xs for y in ys])

    target_height = h_new
    target_width = w_new
    target = np.zeros((target_height, target_width, c), dtype=texture_1.dtype)

    step = blk_size - overlap_size

    y_begin = 0
    y_end = blk_size

    for y in range(ny_blocks):

        x_begin = 0
        x_end = blk_size

        for x in range(nx_blocks):
            if x == 0 and y == 0:
                # randomly select top left patch
                r = np.random.randint(len(all_blocks))
                random_patch = all_blocks[r]
                target[y_begin:y_end, x_begin:x_end] = random_patch

                x_begin = x_end
                x_end += step

                continue

            xa, xb = x_begin - blk_size, x_begin
            ya, yb = y_begin - blk_size, y_begin

            if y == 0:
                y1 = 0
                y2 = blk_size

                left_patch = target[y1:y2, xa:xb]
                left_block = left_patch[:, -overlap_size:]
                left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks)
                best_right_patch = select_min_patch(all_blocks, left_cost)

                best_right_block = best_right_patch[:, :overlap_size]

                # join left and right blocks
                left_right_join = np.hstack(
                    (left_block[:, :overlap_size // 2], best_right_block[:, overlap_size // 2:]))
                full_join = np.hstack(
                    (target[y1:y2, xa:xb - overlap_size], left_right_join, best_right_patch[:, overlap_size:]))

                target[y1:y2, xa:x_end] = full_join
            else:
                if x == 0:
                    x1 = 0
                    x2 = blk_size
                    top_patch = target[ya:yb, x1:x2]
                    top_block = top_patch[-overlap_size:, :]
                    top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks)
                    best_bottom_patch = select_min_patch(all_blocks, top_cost)
                    best_bottom_block = best_bottom_patch[:overlap_size, :]

                    # join top and bottom blocks
                    top_bottom_join = np.vstack(
                        (top_block[:overlap_size // 2, :], best_bottom_block[overlap_size // 2:, :]))
                    full_join = np.vstack(
                        (target[ya:yb - overlap_size, x1:x2], top_bottom_join, best_bottom_patch[overlap_size:, :]))

                    target[ya:y_end, x1:x2] = full_join
                else:
                    # overlap is L-shaped
                    y1, y2 = y_begin - overlap_size, y_end
                    x1, x2 = x_begin - overlap_size, x_end

                    left_patch = target[y1:y2, xa:xb]
                    top_patch = target[ya:yb, x1:x2]

                    left_block = left_patch[:, -overlap_size:]
                    top_block = top_patch[-overlap_size:, :]

                    left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks)
                    top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks)

                    best_right_patch = best_bottom_patch = select_min_patch(all_blocks, top_cost + left_cost)

                    best_right_block = best_right_patch[:, :overlap_size]
                    best_bottom_block = best_bottom_patch[:overlap_size, :]

                    # join left and right blocks
                    left_right_join = np.hstack(
                        (left_block[:, :overlap_size // 2], best_right_block[:, overlap_size // 2:]))
                    full_lr_join = np.hstack(
                        (target[y1:y2, xa:xb - overlap_size], left_right_join, best_right_patch[:, overlap_size:]))

                    # join top and bottom blocks
                    top_bottom_join = np.vstack(
                        (top_block[:overlap_size // 2, :], best_bottom_block[overlap_size // 2:, :]))
                    full_tb_join = np.vstack(
                        (target[ya:yb - overlap_size, x1:x2], top_bottom_join, best_bottom_patch[overlap_size:, :]))

                    target[ya:y_end, x1:x2] = full_tb_join
                    target[y1:y2, xa:x_end] = full_lr_join

            x_begin = x_end
            x_end += step

        y_begin = y_end
        y_end += step

    return target


def synth_texture(src_texture, blk_size):
    h, w, c = src_texture.shape

    assert blk_size < min(h, w)

    y_max, x_max = h - blk_size, w - blk_size
    dh = h * 2
    dw = w * 2
    nx_blocks = ny_blocks = max(dh, dw) // blk_size
    w_new = h_new = nx_blocks * blk_size - (nx_blocks - 1) * overlap_size

    xs = np.arange(x_max)
    ys = np.arange(y_max)
    all_blocks = np.array([src_texture[y:y + blk_size, x:x + blk_size] for x in xs for y in ys])

    target_height = h_new
    target_width = w_new
    target = np.zeros((target_height, target_width, c), dtype=texture_1.dtype)

    step = blk_size - overlap_size

    y_begin = 0
    y_end = blk_size

    for y in range(ny_blocks):

        x_begin = 0
        x_end = blk_size

        for x in range(nx_blocks):
            if x == 0 and y == 0:
                # randomly select top left patch
                r = np.random.randint(len(all_blocks))
                random_patch = all_blocks[r]
                target[y_begin:y_end, x_begin:x_end] = random_patch

                x_begin = x_end
                x_end += step

                continue

            xa, xb = x_begin - blk_size, x_begin
            ya, yb = y_begin - blk_size, y_begin

            if y == 0:
                y1 = 0
                y2 = blk_size

                left_patch = target[y1:y2, xa:xb]
                left_block = left_patch[:, -overlap_size:]
                left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks)
                best_right_patch = select_min_patch_tol(all_blocks, left_cost)
                best_right_block = best_right_patch[:, :overlap_size]

                left_right_join = compute_lr_join(left_block, best_right_block)
                # join left and right blocks
                full_join = np.hstack(
                    (target[y1:y2, xa:xb - overlap_size], left_right_join, best_right_patch[:, overlap_size:]))

                target[y1:y2, xa:x_end] = full_join
            else:
                if x == 0:
                    x1 = 0
                    x2 = blk_size
                    top_patch = target[ya:yb, x1:x2]
                    top_block = top_patch[-overlap_size:, :]
                    top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks)
                    best_bottom_patch = select_min_patch_tol(all_blocks, top_cost)
                    best_bottom_block = best_bottom_patch[:overlap_size, :]

                    # join top and bottom blocks
                    top_bottom_join = compute_bt_join(top_block, best_bottom_block)
                    full_join = np.vstack(
                        (target[ya:yb - overlap_size, x1:x2], top_bottom_join, best_bottom_patch[overlap_size:, :]))

                    target[ya:y_end, x1:x2] = full_join
                else:
                    # overlap is L-shaped
                    y1, y2 = y_begin - overlap_size, y_end
                    x1, x2 = x_begin - overlap_size, x_end

                    left_patch = target[y1:y2, xa:xb]
                    top_patch = target[ya:yb, x1:x2]

                    left_block = left_patch[:, -overlap_size:]
                    top_block = top_patch[-overlap_size:, :]

                    left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks)
                    top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks)

                    best_right_patch = best_bottom_patch = select_min_patch_tol(all_blocks, top_cost + left_cost)

                    best_right_block = best_right_patch[:, :overlap_size]
                    best_bottom_block = best_bottom_patch[:overlap_size, :]

                    left_right_join, top_bottom_join = lr_bt_join_double(best_right_block, left_block,
                                                                         best_bottom_block, top_block)
                    # join left and right blocks
                    full_lr_join = np.hstack(
                        (target[y1:y2, xa:xb - overlap_size], left_right_join, best_right_patch[:, overlap_size:]))

                    # join top and bottom blocks
                    full_tb_join = np.vstack(
                        (target[ya:yb - overlap_size, x1:x2], top_bottom_join, best_bottom_patch[overlap_size:, :]))

                    target[ya:y_end, x1:x2] = full_tb_join
                    target[y1:y2, xa:x_end] = full_lr_join

            x_begin = x_end
            x_end += step

        y_begin = y_end
        y_end += step

    return target


def show_fig2a(texture_img):
    plt.title('Figure 2a')
    plt.imshow(normalize_img(texture_img))
    plt.axis('off')
    plt.show()


def show_fig2b(texture_img):
    plt.title('Figure 2b')
    plt.imshow(normalize_img(texture_img))
    plt.axis('off')
    plt.show()


def show_fig2c(texture_img):
    plt.title('Figure 2c')
    plt.imshow(normalize_img(texture_img))
    plt.axis('off')
    plt.show()


texture_1 = plt.imread('data/texture1.jpg').astype(np.float32)

block_size = 100
overlap_size = int(block_size / 6)

show_fig2a(synth_texture_rand(texture_1, block_size))
show_fig2b(synth_texture_neighborhood(texture_1, block_size))
show_fig2c(synth_texture(texture_1, block_size))

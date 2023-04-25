import cv2
import numpy as np
import random


# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray


# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name, img):
    cv2.imshow(window_name, img)


# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def knn_query(a, b, k=2):
    diff = np.abs(b - a[:, np.newaxis])
    dis = np.linalg.norm(diff, axis=2)
    knn_idx = np.argsort(dis, axis=1)[:, :k]

    return knn_idx, np.array([row[ix] for row, ix in zip(dis, knn_idx)])


def findHomo(matches: np.ndarray):
    orig, tgt = matches[:, 0, :], matches[:, 1, :]
    ones_vec = np.ones((4, 1))
    point_block = np.concatenate((orig, ones_vec), axis=1)
    zeros_block = np.zeros_like(point_block)
    left_block = np.block([
        [point_block, zeros_block],
        [zeros_block, point_block]
    ])
    last2col = orig.T[:, np.newaxis] * tgt.T
    last2col = last2col.reshape(2, -1).T
    A = np.concatenate((left_block, -last2col, -tgt.T.reshape(-1, 1)), axis=1)
    v, s, vt = np.linalg.svd(A)
    H = vt[-1].reshape((3, 3))
    H /= H.item(8)
    return H


def ransac(num_iter, gmi: np.ndarray):
    max_inliner = 0
    best_H = None
    for i in range(num_iter):
        sampled_idxes = random.sample(range(len(gmi)), 4)
        H = findHomo(gmi[sampled_idxes])

        inliner = 0
        for j in range(len(gmi)):
            if j not in sampled_idxes:
                src = np.concatenate((gmi[j, 0], [1]))
                dst_prime = H @ src
                dst_prime /= dst_prime[2]
                if np.linalg.norm(dst_prime[:2] - gmi[j, 1]) < 5:
                    inliner += 1

        if inliner > max_inliner:
            max_inliner = inliner
            best_H = H
    print("The Number of Maximum Inliner: ", max_inliner)
    return best_H


def stitchTwoImgs(img1: np.ndarray, img2: np.ndarray, H: np.ndarray):
    (h2, w2) = img2.shape[:2]
    lb_corner = np.array([[0, 0, 1], [h2-1, 0, 1], [h2-1, w2-1, 1], [0, w2-1, 1]])
    corner_p = H @ lb_corner.T
    # corner_p /= corner_p[2]
    x1_prime, y1_prime = corner_p.min(axis=1)[:2]
    x1_prime = min(x1_prime, 0)
    y1_prime = min(y1_prime, 0)
    size = (int(w2 + abs(x1_prime)), int(h2 + abs(y1_prime)))
    A = np.eye(3)
    A[0, 2], A[1, 2] = -int(x1_prime), -int(y1_prime)
    warped_1 = cv2.warpPerspective(img1, M=A@H, dsize=size)
    warped_2 = cv2.warpPerspective(img2, M=A, dsize=size)
    img1_mask = np.any(warped_1, axis=-1)
    img2_mask = np.any(warped_2, axis=-1)

    # stitch_img = np.zeros((size[1], size[0], 3))
    # only_img1_ix = np.logical_and(img1_mask, ~img2_mask)
    # stitch_img[only_img1_ix] = warped_1[only_img1_ix]
    # only_img2_ix = np.logical_and(~img1_mask, img2_mask)
    # stitch_img[only_img2_ix] = warped_2[only_img2_ix]
    # common_mask = np.logical_and(img1_mask, img2_mask)
    # stitch_img[common_mask] = cv2.addWeighted(warped_1[common_mask], 0.5, warped_2[common_mask], 0.5, 0)
    # stitch_img = stitch_img.astype(np.uint8)

    only_img1_ix = np.logical_and(img1_mask, ~img2_mask)
    only_img2_ix = np.logical_and(~img1_mask, img2_mask)
    common_mask = np.logical_and(img1_mask, img2_mask)
    h, w = np.where(common_mask)  # (row_idx array, col_idx array)
    bound = np.argwhere(np.diff(h).astype(bool)).ravel()
    bound = np.concatenate(([-1], bound, [-1]))
    w_start_idx, w_end_idx = bound[:-1]+1, bound[1:]
    w_bound_idx = w[np.vstack((w_start_idx, w_end_idx))]
    common_area_w = np.diff(w_bound_idx, axis=0).ravel()
    middle_idx = w[w_start_idx] + common_area_w//2
    const_w = 3

    alpha_mask = np.zeros_like(common_mask)
    alpha_mask[only_img1_ix] = 1
    alpha_mask[only_img2_ix] = 0
    for hei, s, m, e in zip(h[w_start_idx], w[w_start_idx], middle_idx, w[w_end_idx]):
        alpha_mask[hei, s:m-const_w] = 1
        alpha_mask[hei, m+const_w+1:e+1] = 0
    # 只 linear blending common area 的中線附近
    for i in range(-const_w, const_w+1):
        decrease_step = 1 / common_area_w
        # 越靠右的會採用右方越多
        alpha_mask[h[w_start_idx], middle_idx+i] = 1 - (decrease_step * (middle_idx+i-w[w_start_idx]))

    stitch_img = alpha_mask[..., np.newaxis] * warped_1 + (1 - alpha_mask[..., np.newaxis]) * warped_2

    return stitch_img.astype(np.uint8)


# def val_matches(descriptors, i, d):
#     matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors[0], descriptors[1], 2)
#     ki, vs = [], []
#     for match in matches:
#         if match[0].distance <= 0.7 * match[1].distance:
#             ki.append([match[0].trainIdx, match[1].trainIdx])
#             vs.append([match[0].distance, match[1].distance])
#     ki = np.array(ki)
#     vs = np.array(vs)
#     assert np.allclose(i, ki)
#     assert np.allclose(d, vs)


if __name__ == '__main__':
    num_image = 2
    imgs = np.stack([read_img(f"test/m{i}.jpg")[0] for i in range(1, num_image+1)])[::-1]
    grey_imgs = np.stack([read_img(f"test/m{i}.jpg")[1] for i in range(1, num_image+1)])[::-1]
    SIFT_Detector = cv2.SIFT_create()
    kps, descriptors = [], []
    for grey_img in grey_imgs:
        kp, des = SIFT_Detector.detectAndCompute(grey_img, None)
        kps.append(kp)
        descriptors.append(des)

    result = None
    for ((dr, dl), (kr, kl), (imr, iml)) in zip(zip(descriptors, descriptors[1:]),
                                                zip(kps, kps[1:]),
                                                zip(imgs, imgs[1:])):
        if result is not None:
            kr, dr = SIFT_Detector.detectAndCompute(
                cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), None)
        else:
            result = imr

        idx, distance = knn_query(dl, dr, k=2)
        good_match = (distance[:, 0] <= 0.7 * distance[:, 1]).nonzero()[0]
        # val_matches(descriptors, idx[good_match], distance[good_match])

        good_match_idxes = []
        for i in good_match:
            pos_a = tuple(int(v) for v in kl[i].pt)
            pos_b = tuple(int(v) for v in kr[idx[i, 0]].pt)
            good_match_idxes.append([pos_a, pos_b])
        good_match_idxes = np.array(good_match_idxes)
        H = ransac(10000, good_match_idxes)

        result = stitchTwoImgs(iml, result, H)

    # creat_im_window(f"result_{num_image}imgs", result)
    # im_show()
    # you can use this function to store the result
    cv2.imwrite(f"result_{num_image}imgs.jpg", result)

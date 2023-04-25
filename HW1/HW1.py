import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
import open3d as o3d
import matplotlib.pyplot as plt

image_row = 0 
image_col = 0


# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')


# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vector of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')


# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row, image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')


# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z, filepath):
    Z_map = np.reshape(Z, (image_row, image_col)).copy()
    data = np.zeros((image_row*image_col, 3), dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)


# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])


# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = image.shape
    return image


def read_ls(filepath):
    with open(filepath) as f:
        return np.array([eval(line.strip('\n').split(' ')[-1]) for line in f.readlines()])


def divide_handle_zero(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(b != 0))


def normalize(ls):
    n_zer = np.sqrt((ls ** 2).sum(axis=1, keepdims=True))
    return divide_handle_zero(ls, n_zer)


def normal_est(imgs, imgs_ls):
    imgs = np.reshape(imgs, (6, -1))
    n_imgs_ls = normalize(imgs_ls)
    ls_inv = np.linalg.pinv(n_imgs_ls)
    KdN = (ls_inv @ imgs).swapaxes(0, 1)
    return normalize(KdN)


def create_V(N):
    # one dim array
    nx, ny, nz = N[:, 0], N[:, 1], N[:, 2]
    not_all_zero = np.invert(np.all(N == 0, 1))
    nx, ny, nz = nx[not_all_zero], ny[not_all_zero], nz[not_all_zero]
    V_upper = -nx / nz
    V_lower = ny / nz
    V = np.concatenate((V_upper, V_lower), axis=0)
    not_all_zero_orig_idx = np.where(not_all_zero)[0]

    # deal with x edge cases
    x_edge_cases = [((idx + 1 not in not_all_zero_orig_idx)
                     and (idx - 1 in not_all_zero_orig_idx))
                    for idx in not_all_zero_orig_idx]
    dwe = np.ones_like(V_upper)
    dwe[x_edge_cases] *= -1
    V[:len(V_upper)] *= dwe

    # deal with y edge cases
    y_edge_cases = [((idx + image_col not in not_all_zero_orig_idx)
                     and (idx - image_col in not_all_zero_orig_idx))
                    for idx in not_all_zero_orig_idx]
    dwe = np.ones_like(V_lower)
    dwe[y_edge_cases] *= -1
    V[len(V_upper):] *= dwe

    return V, not_all_zero_orig_idx


def create_M(orig_nz_idx):
    row_upper = np.repeat(np.arange(len(orig_nz_idx)), 2)
    col_upper = row_upper.copy()
    z_x_next = np.array([np.where(orig_nz_idx == (idx + 1))[0][0]
                         if np.where(orig_nz_idx == (idx + 1))[0].size > 0
                         else np.where(orig_nz_idx == (idx - 1))[0][0]
                         if np.where(orig_nz_idx == (idx - 1))[0].size > 0
                         else -1
                         for idx in orig_nz_idx])
    col_upper[1::2] = z_x_next

    row_lower = np.repeat(np.arange(len(orig_nz_idx)), 2)
    col_lower = row_lower.copy()
    row_lower += len(orig_nz_idx)
    z_y_next = np.array([np.where(orig_nz_idx == (idx + image_col))[0][0]
                         if np.where(orig_nz_idx == (idx + image_col))[0].size > 0
                         else np.where(orig_nz_idx == (idx - image_col))[0][0]
                         if np.where(orig_nz_idx == (idx - image_col))[0].size > 0
                         else -1
                         for idx in orig_nz_idx])
    col_lower[1::2] = z_y_next

    row = np.concatenate((row_upper, row_lower), axis=0)
    col = np.concatenate((col_upper, col_lower), axis=0)
    data = np.ones_like(row)
    data[::2] -= 2

    row = row[col != -1]
    data = data[col != -1]
    col = col[col != -1]

    return csr_matrix((data, (row, col)), shape=(2*len(orig_nz_idx), len(orig_nz_idx)), dtype=np.int8)


# N: (num_pixels x 3)
def surface_recon(N):
    V, orig_nz_idx = create_V(N)

    M = create_M(orig_nz_idx)

    z = inv(M.T @ M) @ M.T @ V

    depth = np.zeros(shape=(len(N), ))
    depth[orig_nz_idx] = z

    return depth


if __name__ == '__main__':
    # showing the windows of all visualization function
    for tgt in ['bunny', 'star', 'venus']:
        ply_path = f'./{tgt}/{tgt}.ply'
        images = np.array([read_bmp(f"test/{tgt}/pic{i}.bmp") for i in range(1, 7)])
        images_light_source = read_ls(f"test/{tgt}/LightSource.txt")
        normal_vectors = normal_est(images, images_light_source)
        normal_visualization(normal_vectors)
        z_depth = surface_recon(normal_vectors)
        depth_visualization(z_depth)
        save_ply(z_depth, ply_path)
        show_ply(ply_path)
    plt.show()

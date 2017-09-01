import sys, os, glob, cv2
import numpy as np

# make textfile for cones
# filepath, x1, y1, x2, y2, class_name`
# For example:
# /data/imgs/img_001.jpg, 837, 346, 981, 456, cow
# /data/imgs/img_002.jpg, 215, 312, 279, 391, cat

# img folders
train_imgs_folder = './final/train'
valid_imgs_folder = './final/valid'


def make_bbox_data(is_train=False, do_visualize=False):
    img_folder = train_imgs_folder if is_train else valid_imgs_folder
    output_file = 'cone_data%s.txt' % ('_train' if is_train else '_valid')

    filepath = "%s/*.png" % (img_folder)
    pnames = sorted(list(glob.glob(filepath)))
    for idx, pname in enumerate(pnames):
        print('%d/%d; %s' % (idx+1, len(pnames), pname))
        if 'mask' not in pname: continue

        # mask has the coordinates for creating target
        img_name = '%s.png' % pname.split('_mask')[0]

        img = cv2.imread(os.path.join(pname), cv2.IMREAD_GRAYSCALE)
        img_nonzeros = np.transpose(np.nonzero(img))    # find cone centers in mask file

        with open(output_file, 'a') as fout:
            all_coords = []
            for nonzero in img_nonzeros:
                [x1, y1, x2, y2] = compute_cone_coords(nonzero[0], nonzero[1])
                cur_val = [img_name, str(x1), str(y1), str(x2), str(y2), 'cone']
                fout.write(','.join(cur_val))
                fout.write('\n')

                all_coords.append((x1, y1, x2, y2))

        if do_visualize:
            visualize_bbox_data(img_name, mask=img, all_coords=all_coords, center_coords=img_nonzeros)
    return


def compute_cone_coords(x, y, num_rows=32, num_cols=32):
    cone_size = 5
    x1 = max(x-(cone_size//2), 0)             # lower bound
    x2 = min(x+(cone_size//2), num_rows-1)   # upper bound
    y1 = max(y-(cone_size//2), 0)             # lower bound
    y2 = min(y+(cone_size//2), num_cols-1)    # upper bound
    # return x1, y1, x2, y2
    return y1, x1, y2, x2   # imshow flips coords


def visualize_bbox_data(img_name, mask, all_coords, center_coords):
    from matplotlib import pyplot as plt
    from matplotlib import patches
    # plt.switch_backend('agg')
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    # # as is
    # plt.figure(10)
    # plt.imshow(img)
    # plt.title('original image')
    # plt.figure(11)
    # plt.imshow(mask)
    # plt.title('mask center coordinates')

    # actual centers from mask on same plot
    plt.figure(3)
    plt.imshow(img)
    # plt.scatter(x=center_coords[:,0], y=center_coords[:,1], c='blue', s=10)
    # NB - imshow flips coordinates
    plt.scatter(x=center_coords[:,1], y=center_coords[:,0], c='red', s=10)
    plt.title('original data with labeled cone centers')

    # plot derived center points on img
    plt.figure(2)
    coord_centers_x = []
    coord_centers_y = []
    for coord in all_coords:
        (real_x1, real_y1, real_x2, real_y2) = coord
        coord_centers_x.append((real_x1 + real_x2) / 2.0)
        coord_centers_y.append((real_y1 + real_y2) / 2.0)
    plt.imshow(img)
    plt.scatter(x=coord_centers_x, y=coord_centers_y, c='blue', s=10)
    # plt.scatter(x=coord_centers_y, y=coord_centers_x, c='red', s=10)   # imshow flips coords
    plt.title('original data with derived cone centers from my labeled bounding boxes')

    # img and bounding boxes
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    ax1.scatter(x=center_coords[:,1], y=center_coords[:,0], c='red', s=10)
    for coord in all_coords:
        (real_x1, real_y1, real_x2, real_y2) = coord
        ax1.add_patch(
            patches.Rectangle((real_x1, real_y1), real_x2 - real_x1, real_y2 - real_y1, fill=False, color='green'))
        # (real_x1, real_y1, real_x2, real_y2) = coord
        # ax1.add_patch(
        #     patches.Rectangle((real_y1, real_x1), real_y2 - real_y1, real_x2 - real_x1, fill=False, color='green'))
    ax1.set_title('img with labeled centers and my hacky bounding box labels')
    return


def create_sub_images(img_folder, img_path, nrows, ncols, is_overlapping=False):
    img = cv2.imread('{}/{}'.format(img_folder, img_path))
    img_shape = img.shape
    img_name_base = img_path.split('.tiff')[0]

    step_size_r = 1 if is_overlapping else nrows
    step_size_c = 1 if is_overlapping else ncols
    for i in range(0, img_shape[0], step_size_r):
        for j in range(0, img_shape[1], step_size_c):
            cur_img = img[i:i+nrows, j:j+ncols, :]
            cur_name = '{}/{}_{}_{}.png'.format('./testImages', img_name_base, i, j)
            cv2.imwrite(cur_name, cur_img)
    return


# make larger images for testing
def make_large_test_images():
    data_folder = './data'
    file_name = 'PrefixS_V011.tiff'
    data = cv2.imread('{}/{}'.format(data_folder, file_name))
    sub_row = 64
    sub_col = 64
    create_sub_images(data_folder, file_name, sub_row, sub_col, is_overlapping=False)

    # # why is cropped testImages incorrect?
    # test_image_path = '{}/{}.png'.format('./testImages', 'testImage')
    # test_data = cv2.imread(test_image_path)

    # # compare against other data files
    # valid_img_path = '{}/{}.png'.format(valid_imgs_folder, 'PrefixS_V011_cropped_master_image_43-51_29-51_98-98_101-98-txt-0-0')
    # valid_data = cv2.imread(valid_img_path)
    return data[0:sub_row, 0:sub_col, :]


if __name__ == '__main__':
    # make_bbox_data(is_train=True)
    # make_bbox_data(is_train=False, do_visualize=True)
    make_large_test_images()
import sys, os, glob, cv2, json
import numpy as np

# make textfile for cones
# filepath, x1, y1, x2, y2, class_name`
# For example:
# /data/imgs/img_001.jpg, 837, 346, 981, 456, cow
# /data/imgs/img_002.jpg, 215, 312, 279, 391, cat

# img folders
train_imgs_folder = './final/train'
valid_imgs_folder = './final/valid'
from sys import platform
if platform == "linux" or platform == "linux2":
    # linux
    segmentation_json_folder = '/home/ayl/data/pepple/accell/jsons'
    img_folder = '/home/ayl/data/pepple/accell/segmentations'
    output_folder = '/home/yue/pepple/accell/processed'
    output_folder_test = '/home/yue/pepple/accell/test'
elif platform == "win32":
    # Windows...
    segmentation_json_folder = './accell/jsons'
    img_folder = './accell/segmentations'
    output_folder = './accell/processed'
    output_folder_test = './accell/test'

nrows=496*2   # correspond to y-axis
ncols=128   # correspond to x-axis
step_size_r=8*2
step_size_c=8*2


def visualize_bbox_data(img, img_coords, cur_img, cur_coords, cur_x, cur_y, flipAxis=False):
    from matplotlib import pyplot as plt
    from matplotlib import patches
    # plt.switch_backend('agg')

    # view large image - with coords
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    # show img strip
    ax1.add_patch(patches.Rectangle((cur_x, cur_y), ncols, nrows, fill=False, color='red'))

    for coord in img_coords:    # add cells to image
        (xs, ys) = coord['mousex'], coord['mousey']
        if flipAxis:    # shouldnt need to flip
            ax1.scatter(x=ys, y=xs, c='red', s=10)  # be careful of orientation - imshow flips coords
        else:
            ax1.scatter(x=xs, y=ys, c='red', s=10)  # not flipped

    # np.sum(img[cur_y:cur_y+nrows, cur_x:cur_x+ncols] == cur_img)
    # view cropped image in situ - with coords
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(cur_img)
    for coord in cur_coords:    # add cells to image
        (real_x1, real_y1, real_x2, real_y2) = coord    # these bounding box processed
        if flipAxis:    # imshow flips coords - shouldnt need to flip
            ax1.add_patch(
                patches.Rectangle((real_y1, real_x1), real_y2 - real_y1, real_x2 - real_x1, fill=False, color='green'))
        else:
            ax1.add_patch(
                patches.Rectangle((real_x1, real_y1), real_x2 - real_x1, real_y2 - real_y1, fill=False, color='green'))
            ax1.scatter(x=[(real_x1+real_x2)/2], y=[(real_y1+real_y2)/2], c='red', s=10)
    return


def make_ac_coords(img_coords, cur_x, cur_y):
    cur_coords = []
    # cell_size = 5   # when zoomed in - larger than 1 pixel
    cell_size = 10   # make it look like islands
    for coord in img_coords:
        xs = coord['mousex']
        ys = coord['mousey']
        for idx, x in enumerate(xs):
            y = ys[idx]
            if (x>=cur_x and x<cur_x+ncols) and (y>=cur_y and y<cur_y+nrows):  # if within bounds then process
                1
            else:
                continue

            lower_x = max(x-(cell_size//2), cur_x) - cur_x
            upper_x = min(x+(cell_size//2), cur_x+ncols) - cur_x

            lower_y = max(y-(cell_size//2), cur_y) - cur_y
            upper_y = min(y+(cell_size//2), cur_y+nrows) - cur_y
            cur_coords.append((lower_x, lower_y, upper_x, upper_y))     # imshow flips coords
    return cur_coords


def create_cropped_images(base_name, img, img_coords, do_write=True, visualise=False):
    img_shape = img.shape
    for y in range(0, img_shape[0]-nrows+1, step_size_r):   # y-axis
        for x in range(0, img_shape[1]-ncols+1, step_size_c):   # x-axis
            cur_img = img[y:y+nrows, x:x+ncols]     # height(y) by width(x)
            # shouldn't hit this bit
            cur_shape = cur_img.shape
            if not (cur_shape[0]==nrows and cur_shape[1]==ncols):   # undersized strip and ignore
                continue

            if do_write:
                cur_name = '{}/{}_{}_{}.png'.format(output_folder, base_name, x, y)
            else:
                cur_name = '{}/{}_{}_{}.png'.format(output_folder_test, base_name, x, y)
            cv2.imwrite(cur_name, cur_img)

            if do_write==1:
                outfile = './{}.txt'.format('training_coords')
            else:
                outfile = './{}.txt'.format('test_coords')
            with open(outfile, 'a') as fout:
                cur_coords = make_ac_coords(img_coords, x, y)
                # print(len(cur_coords))
                for coord in cur_coords:
                    vals = [cur_name, str(coord[0]), str(coord[1]), str(coord[2]), str(coord[3]), 'cell']
                    print(','.join(vals))
                    fout.write('{}\n'.format(','.join(vals)))

            if do_write and visualise and len(cur_coords)>0:
                visualize_bbox_data(img, img_coords, cur_img, cur_coords, x, y)
    return


# for each seg and corresponding image
# make strip data
# and record labeled bounding boxes in text file
# keep valid/training images distinct
def make_test_images():
    pnames = sorted(list(glob.glob('{}/*.json'.format(segmentation_json_folder))))

    seen_names = []
    num_samples = 0
    sample_threshold = 216*.8
    for pname in pnames:
        sample_name = pname.split('mouse')[1]
        if sample_name in seen_names: continue  # no duplicate samples
        else:
            seen_names.append(sample_name)
            num_samples +=1

        coord_data = get_coords(pname)

        # corresponding image
        base_name = pname.split('.json')[0].replace('{}/'.format(segmentation_json_folder), '')
        # base_name = pname.split('.json')[0].replace('{}\\'.format(segmentation_json_folder), '')
        img_path = make_img_name(base_name)
        # print(base_name, img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # create cropped images and write to text file
        print(pname, num_samples, sample_threshold, num_samples<sample_threshold)
        create_cropped_images(base_name, img, coord_data, do_write=num_samples<sample_threshold)
    return 1


def make_img_name(base_name):
    # img_path = base_name.replace(segmentation_json_folder, img_folder)
    # image_path = '{}.png'.format(img_path)
    image_path = '{}/{}.png'.format(img_folder, base_name)
    return image_path


def get_coords(coord_file):
    fin = open(coord_file).read()
    json_data = json.loads(fin)
    return json_data


def check_json_consistency():
    pnames = sorted(list(glob.glob('{}/*.json'.format(segmentation_json_folder))))
    total_coords = 0
    for pname in pnames:
        coord_data = get_coords(pname)
        for coord in coord_data:
            total_coords +=len(coord['mousex'])
            if len(coord['mousex'])>1 or len(coord['mousey'])>1:
                print(pname, coord['mousex'], coord['mousey'])
    return total_coords     # about 2531 cells over 216(1024*1000) files (0.2%)


def add_sim_ac_cells(img, img_coords, num_new):
    # avoid overlapping points
    # avoid high intensity areas
    # choose coords inside chamber or just randomly?
    # cell intensities
    return


if __name__ == '__main__':
    make_test_images()
    # check_json_consistency()
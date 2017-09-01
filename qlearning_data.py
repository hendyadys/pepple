import os, cv2
import numpy as np
import json

from matplotlib import pyplot as plt
# plt.switch_backend('agg')

image_rows = 512
image_cols = 500
data_rows = 64
data_cols = 64
min_num_examples = 10

from sys import platform
if platform == "linux" or platform == "linux2":
    # linux
    json_path = '/data/pepple/acseg/jsons'
    data_path = '/data/pepple/acseg/segmentations'
elif platform == "darwin":
    # OS X - not currently supported
    json_path = '/data/pepple/acseg/jsons'
    data_path = '/data/pepple/acseg/segmentations'
elif platform == "win32":
    # Windows...
    json_path = './acseg/jsons'
    data_path = './acseg/segmentations'
train_npy = './npy/ql_train.npy'
train_target_npy = './npy/ql_train_target.npy'
valid_npy = './npy/ql_valid.npy'
valid_target_npy = './npy/ql_valid_target.npy'


def make_qlearning_data(visualise=False):
    import json
    json_files = os.listdir(json_path)

    max_samples = len(json_files)*(image_rows+image_cols)*2     # too big an upper limit?
    max_samples = len(json_files)*500     # too big an upper limit?
    my_counter = 0
    effective_images = 0
    val_img_start = 60
    all_data = np.zeros((max_samples, data_rows, data_cols, 2))
    direction_data = np.zeros((max_samples, 8))
    for idx, json_file in enumerate(json_files):
        if 'DeRuyter' not in json_file: continue    # nick filter
        print('%d/%d; processing %s; effective image=%d' % (idx, len(json_files), json_file, effective_images))

        json_data, img = get_data_for_image(json_file)
        if json_data:
            effective_images +=1
            if effective_images==val_img_start:
                training_counter = my_counter
                print('%d effective images at counter=%d' % (effective_images, my_counter))
        else:
            continue

        temp_train_data, temp_direction_data = make_data_for_image(json_data, img, visualise=visualise)
        num_img_data = temp_train_data.shape[0]
        all_data[my_counter:my_counter+num_img_data,] = temp_train_data
        direction_data[my_counter:my_counter+num_img_data,] = temp_direction_data
        my_counter += num_img_data

    np.save(train_npy, all_data[:training_counter+1, ])
    np.save(train_target_npy, direction_data[:training_counter+1, ])
    np.save(valid_npy, all_data[training_counter+1:my_counter, ])
    np.save(valid_target_npy, direction_data[training_counter+1:my_counter, ])
    return


def get_data_for_image(json_file):
    json_file_path = '{}\{}'.format(json_path, json_file)
    # load appropriate image
    image_name = '{}.png'.format(json_file.split('.json')[0])
    image_path = '{}\{}'.format(data_path, image_name)
    if not os.path.isfile(image_path):  # new image not available locally
        return None, None

    fin = open(json_file_path).read()
    json_data = json.loads(fin)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return json_data, img


def make_data_for_image(json_data, img, visualise=False):
    # create data
    # all_xs = json_data[0]['mousex']
    # all_ys = json_data[0]['mousey']
    all_xs, all_ys = get_coords(json_data)
    all_xs, all_ys = remove_duplicates(all_xs, all_ys)

    num_points = len(all_xs)
    my_counter = 0
    all_data = np.zeros((num_points, data_rows, data_cols, 2))
    direction_data = np.zeros((num_points, 8))
    for i, coordx in enumerate(all_xs):
        if i > min_num_examples:  # have at least some points to learn from
            cur_xs = all_xs[:i]  # data that i have seen so far
            cur_ys = all_ys[:i]
            next_x = all_xs[i]
            next_y = all_ys[i]
            cur_patch, cur_intensity_patch, cur_target = make_patch(img, cur_xs, cur_ys, next_x, next_y,
                                                                    data_rows, data_cols, visualise=visualise)
            all_data[my_counter, :, :, 0] = cur_patch
            all_data[my_counter, :, :, 1] = cur_intensity_patch
            direction_data[my_counter, :] = cur_target
            my_counter += 1
    return all_data[0:my_counter,], direction_data[0:my_counter,]


def get_coords(json_data):
    xs = []
    ys = []
    for coord in json_data:
        cur_xs = coord['mousex']
        cur_ys = coord['mousey']
        if len(cur_xs)>len(xs):
            xs = cur_xs
            ys = cur_ys
    return xs, ys


def remove_duplicates(xs, ys):
    true_xs = []
    true_ys = []
    last_x = None
    last_y = None
    for idx, x in enumerate(xs):
        y = ys[idx]
        if x==last_x and y==last_y:
            1  # ignore
        else:
            true_xs.append(x)
            true_ys.append(y)
        last_x = x
        last_y = y

    return true_xs, true_ys


# NB moving along x-axis corresponds to traversing cols, y-axis to traversing rows
def make_patch(img, xs, ys, next_x, next_y, nrows, ncols, visualise=False):
    img_shape = img.shape
    x = xs[-1]
    y = ys[-1]
    x_lower = max(x - ncols//2, 0)
    x_upper = min(x + ncols//2, img_shape[0])
    y_lower = max(y - nrows//2, 0)
    y_upper = min(y + nrows//2, img_shape[1])
    patch = np.zeros((nrows, ncols))
    patch[0:nrows, 0:ncols] = img[y_lower:y_upper, x_lower:x_upper]     # NB rows and cols meaning

    temp = np.zeros(img_shape)
    for idx, cur_x in enumerate(xs):
        cur_y = ys[idx]
        euclidean_dist = np.sqrt((cur_x - x) **2 + (cur_y - y)**2)
        # temp[cur_y, cur_x] = 2**(-(adj_x+adj_y))
        # NB careful with meaning of vertical and horizontal as relevant for rows and cols
        temp[cur_y, cur_x] = max(0, (1 - euclidean_dist*.05))*255    # adjust img copy for intensity
    # temp[ys, xs] = 1
    intensity_patch = np.zeros((nrows, ncols))
    intensity_patch[0:nrows, 0:ncols] = temp[y_lower:y_upper, x_lower:x_upper]     # NB rows and cols meaning

    # one_hot vector target
    target = calc_target(x, y, next_x, next_y, visualise)

    if visualise:
        # patch in situ
        from matplotlib import patches
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img)
        # patch requires coords and x-axis and y-axis displacements
        ax1.add_patch(patches.Rectangle((x_lower, y_lower), x_upper - x_lower, y_upper - y_lower, fill=False, color='red'))
        ax1.scatter(x=xs[-1], y=ys[-1], c='red', s=10)
        ax1.scatter(x=np.asarray(xs), y=np.asarray(ys), c='blue', s=3)
        ax1.set_title('original with patch boundaries and trace points')

        # patches
        plt.figure()
        plt.imshow(patch)
        plt.title('local patch')
        plt.scatter(x=xs[-1]-x_lower, y=ys[-1]-y_lower, c='red', s=10)
        plt.scatter(x=np.asarray(xs)-x_lower, y=np.asarray(ys)-y_lower, c='blue', s=3)

        # intensity patch
        plt.figure()
        plt.imshow(intensity_patch)
        plt.title('intensity patch')

    return patch, intensity_patch, target


def calc_target(cur_x, cur_y, next_x, next_y, visualise=False):
    # square around current coords
    box = np.zeros((3,3))
    box[1,1] = 1    # current value marker, which will be deleted

    # should interpolate if user skips pixels
    x_loc = 1
    if next_y > cur_y:
        x_loc = 2
    elif next_y < cur_y:
        x_loc = 0
    else:
        x_loc = 1
    y_loc = 1
    if next_x > cur_x:
        y_loc = 2
    elif next_x < cur_x:
        y_loc = 0
    else:
        y_loc = 1

    if visualise:
        box_orig = np.copy(box)
        plt.imshow(box_orig)    # careful of meaning of x,y
        plt.scatter(x=y_loc, y=x_loc)
        plt.title('x={}; y={}; next_x={}; next_y={}'.format(cur_x, cur_y, next_x, next_y))
        # NB row change means vertical shift

    box[x_loc, y_loc] = 1     # careful with meaning of x and y vs visualization meaning
    target = box.flatten()
    target = np.delete(target, 4)

    return target


def interpolate(json_x, json_y):
    num_coords = len(json_x)
    interpolated_xs = []
    interpolated_ys = []
    for idx, x in enumerate(json_x):
        y = json_y[idx]
        if idx > 0:  # check if last point within 1 pixel
            last_x = json_x[idx-1]
            last_y = json_y[idx-1]
            if abs(last_x-x) <= 1 and abs(last_y-y) <= 1:
                1  # do nothing
            else:
                x_diff = last_x-x
                y_diff = last_y-y
                slope = float(y_diff)/x_diff
                if abs(last_x-x)>1:   # HACK interpolation
                    for diff in range(0, x_diff, np.sign(x_diff)):
                        cur_x = x + diff
                        cur_y = int(slope*diff + y)  # since we are taking steps
                        interpolated_xs.append(cur_x)
                        interpolated_ys.append(cur_y)
                else:
                    for diff in range(0, y_diff, np.sign(y_diff)):
                        cur_y = y + diff
                        cur_x = int(1.0/slope*diff + x)  # since we are taking steps
                        interpolated_xs.append(cur_x)
                        interpolated_ys.append(cur_y)
        else:   # first point do nothing
            1
        interpolated_xs.append(x)
        interpolated_ys.append(y)
    return interpolated_xs, interpolated_ys


def load_qtrain_data():
    imgs_train = np.load(train_npy)
    target_train = np.load(train_target_npy)
    print('valid size', imgs_train.shape, target_train.shape)
    return imgs_train, target_train


def load_qvalid_data():
    imgs_valid = np.load(valid_npy)
    target_valid = np.load(valid_target_npy)
    print('valid size', imgs_valid.shape, target_valid.shape)
    return imgs_valid, target_valid


if __name__ == '__main__':
    make_qlearning_data()
    # make_qlearning_data(visualise=True)
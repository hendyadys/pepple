# from keras.models import Model
# from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
# from keras.models import load_model
from keras.optimizers import Adam

import numpy as np
import glob
import matplotlib.pyplot as plt

from qlearning_data import get_data_for_image, make_data_for_image, get_coords, remove_duplicates, make_patch
from qtrain import simple_vgg

from sys import platform
if platform == "linux" or platform == "linux2":
    # linux
    results_base = '/home/yue/pepple/runs/2017-08-22-11-25-00'
    results_folder = '/home/yue/pepple/runs/2017-08-22-11-25-00/weights'
    results_figs = '/home/yue/pepple/runs/2017-08-22-11-25-00/figs'
    test_weights = 'weights-improvement-009-0.65264048.hdf5'
elif platform == "darwin":
    # OS X - not currently supported
    1
elif platform == "win32":
    # Windows...
    results_base = './runs/runQlearning'
    results_folder = './runs/runQlearning/weights'
    results_figs = './runs/runQlearning/figs'
    test_weights = 'weights-improvement-009-0.65264048.hdf5'

img_rows = 64
img_cols = 64


def load_params():
    params = []
    with open("%s/params.txt" % results_base) as fin:
        for l in fin:
            arr = l.rstrip().split("\t")
            params.append(np.float32(arr[1]))
    return params


# center and scale data appropriately
def center_scale_imgs(imgs):
    params = load_params()
    imgs = imgs - params[0]
    imgs = imgs/params[1]
    return imgs


def plot_loss(filename):
    iter = []
    loss_val = []
    with open("%s/%s" % (results_base, filename)) as fin:
        for l in fin:
            arr = l.rstrip().split("\t")
            iter.append(np.int(arr[1]))
            loss_val.append(np.float32(arr[-1]))

    if 'train' in filename:
        # plt.plot(iter[0::20], loss_val[0::20])
        k = 21
        plt.plot(iter[0::k], np.mean(np.array(loss_val).reshape(-1, k), axis=1))
    else:
        plt.plot(iter, loss_val)
    plt.title('loss')
    plt.xlabel('iter', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.show()
    return


def predict_for_image(json_file, visualise=False):
    json_data, img = get_data_for_image(json_file)

    img = center_scale_imgs(img)   # normalize prediction data as in training
    img_data, target_data = make_data_for_image(json_data, img, visualise=False)

    # predictions based on model
    model = simple_vgg()
    lr = 1e-5
    model.load_weights('{}\{}'.format(results_folder, test_weights))
    output = model.predict(img_data, batch_size=10)  # inference step

    if visualise:
        plt.switch_backend('agg')
        visualise_predictions(img_data, target_data, output, savepath=json_file)
    return


def visualise_predictions(img_data, target_data, output, savepath=None):
    img_shape = img_data.shape

    total_correct = 0
    for i in range(0, img_shape[0]):
        img = img_data[i,]
        target = target_data[i,]
        cur_pred = output[i,]

        plt.figure(1)
        plt.clf()
        plt.imshow(img[:,:,0])
        x_center, y_center = get_img_center(img)
        plt.scatter(x=x_center, y=y_center, c='yellow', s=10)   # plot actual center

        # actual target
        x_diff, y_diff, box = get_pred_coord(target)
        x_new = x_center + y_diff   # nb meaning of coords shifts and smaller then prediction is above
        y_new = y_center + x_diff
        plt.scatter(x=x_new, y=y_new, c='blue', s=10)

        # predicted
        cur_pred = np.round(cur_pred)
        x_diff2, y_diff2, box = get_pred_coord(cur_pred)
        x_pred = x_center + y_diff2     # nb meaning of coords shifts and smaller then prediction is above
        y_pred = y_center + x_diff2
        plt.scatter(x=x_pred, y=y_pred, c='red', s=10)

        is_correct = np.array_equal(target, cur_pred)
        if is_correct:
            total_correct +=1
        plt.title('pred_correct={}; total_correct/total_preds={}/{}'.format(is_correct, total_correct, i+1))

        if savepath:
            plt.savefig('{}\{}_i{}.png'.format('.\qLearning', savepath, i), bbox_inches='tight')
        else:
            plt.show()
    return


def get_img_center(img):
    img_shape = img.shape
    x = img_rows // 2
    y = img_cols // 2
    return x, y


def get_pred_coord(pred, visualise=False):   # 1- hot vector to direction
    pred2 = np.insert(pred, 4, 0)   # insert current position at center
    box = pred2.reshape((3, 3))

    new_point = np.transpose(np.nonzero(box))
    x_diff = new_point[0][0] - 1
    y_diff = new_point[0][1] - 1

    if visualise:
        box_orig = np.copy(box)
        box_orig[1, 1] = 1      # center point
        plt.imshow(box_orig)    # careful of meaning of x,y
        plt.scatter(y=x_diff+1, x=y_diff+1, c='blue', s=10)
        # NB row change means vertical shift
    return x_diff, y_diff, box


def predict_for_image_dynamic(json_file, visualise=False):
    json_data, img = get_data_for_image(json_file)
    img_shape = img.shape

    img = center_scale_imgs(img)  # normalize prediction data as in training
    all_xs, all_ys = get_coords(json_data)
    all_xs, all_ys = remove_duplicates(all_xs, all_ys)
    all_xs = np.asarray(all_xs).astype(int)   # integer coordinates
    all_ys = np.asarray(all_ys).astype(int)

    # img_data, target_data = make_data_for_image(json_data, img, visualise=visualise)
    min_num_examples = 10
    predicted_xs = np.zeros(all_xs.shape, dtype=int)
    predicted_ys = np.zeros(all_ys.shape, dtype=int)

    # initialize
    predicted_xs[:min_num_examples+1,] = all_xs[:min_num_examples+1,]  # data that i have seen so far
    predicted_ys[:min_num_examples+1,] = all_ys[:min_num_examples+1,]

    # predictions based on model
    model = simple_vgg()
    lr = 1e-5
    model.load_weights('{}\{}'.format(results_folder, test_weights))

    num_preds = 1000
    for i in range(min_num_examples, num_preds):
        prev_xs = predicted_xs[i-min_num_examples:i,]
        prev_ys = predicted_ys[i-min_num_examples:i,]
        cur_x = predicted_xs[i,]
        cur_y = predicted_ys[i,]

        cur_patch, cur_intensity_patch, cur_target = make_patch(img, prev_xs, prev_ys, cur_x, cur_y,
                                                                img_rows, img_cols, visualise=False)
        cur_img_data = np.zeros((1, img_rows, img_cols, 2))
        cur_img_data[0, :, :, 0] = cur_patch
        cur_img_data[0, :, :, 1] = cur_intensity_patch
        output = model.predict(cur_img_data)  # inference step

        # update predicted_xs
        x_diff, y_diff, box = get_pred_coord(np.round(output), visualise=False)
        predicted_xs[i + 1, ] = int(cur_x + y_diff)
        predicted_ys[i + 1, ] = int(cur_y + x_diff)

        if visualise:
            plt.switch_backend('agg')
            # patch in situ
            x = prev_xs[-1]
            y = prev_ys[-1]
            x_lower = max(x - img_cols // 2, 0)
            x_upper = min(x + img_cols // 2, img_shape[0])
            y_lower = max(y - img_rows // 2, 0)
            y_upper = min(y + img_rows // 2, img_shape[1])

            from matplotlib import patches
            fig1, ax1 = plt.subplots(1)
            ax1.imshow(img)
            # patch requires coords and x-axis and y-axis displacements
            ax1.add_patch(
                patches.Rectangle((x_lower, y_lower), x_upper - x_lower, y_upper - y_lower, fill=False, color='red'))
            ax1.scatter(x=cur_x, y=cur_y, c='blue', s=10)
            ax1.scatter(x=int(cur_x + y_diff), y=int(cur_y + x_diff), c='red', s=10)
            ax1.scatter(x=predicted_xs[:i-1,], y=predicted_ys[:i-1,], c='green', s=3)
            ax1.set_title('original with patch boundaries and trace points')
            # plt.show()
            plt.savefig('{}\{}_dyn_i{}.png'.format('.\qLearning', json_file, i), bbox_inches='tight')
    return


if __name__ == '__main__':
    # plot_loss('valid.txt')
    # plot_loss('train.txt')

    json_file = 'DeRuyter-Inflamed_20170710mouse8_Day1_Right_807.json'
    predict_for_image(json_file, True)
    # predict_for_image_dynamic(json_file, True)
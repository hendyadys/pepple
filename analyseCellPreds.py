import numpy as np
import glob, cv2, os, json
from ast import literal_eval

import matplotlib.pyplot as plt
from matplotlib import patches
# plt.switch_backend('agg')
from make_bbox_data import downsample, DOWNSAMPLE_RATIO, get_coords

from sys import platform
if platform == "linux" or platform == "linux2":
    base_folder = '/home/yue/pepple/accell'
    orig_img_folder = os.path.join(base_folder, 'segmentations')
    empty_img_folder = os.path.join(base_folder, 'empty_segmentations')
    json_folder = os.path.join(base_folder, 'jsons')

    # # vertical transform
    # results_base = '/home/yue/pepple/runs/2017-08-09-10-20-24'
    # test_weights = 'weights-improvement-050--0.95407502.hdf5'

    # with empty seg training data
    results_base = '/home/yue/pepple/runs/2017-11-09-10-26-19'
    # test_weights = 'weights-improvement-025--0.95883078.hdf5'
    # test_weights = 'weights-improvement-050--0.95777710.hdf5'
    test_weights = 'weights-improvement-100--0.96128662.hdf5'

    # # re-ran vertical transform (lr=1e-5)
    # results_base = '/home/yue/pepple/runs/2017-11-09-23-47-29'
    # test_weights = 'weights-improvement-100--0.96851523.hdf5'
elif platform == "win32":
    base_folder = './accell'
    orig_img_folder = os.path.join(base_folder, 'segmentations')
    empty_img_folder = os.path.join(base_folder, 'empty_segmentations')
    json_folder = os.path.join(base_folder, 'jsons')

    results_base = './runs/runEmptySeg'
    test_weights = 'weights-improvement-100--0.96128662.hdf5'

    # # re-ran vertical transform (lr=1e-5)
    # results_base = './runs/runVerticalNew'
    # test_weights = 'weights-improvement-100--0.96851523.hdf5'

results_figs = os.path.join(results_base, 'figs')
results_folder = os.path.join(results_base, 'weights')

img_rows = 1024  # height
img_cols = 1000  # width

from predict_image_cells import calc_img_chamber_size, any_middle_stripes, is_cell_in_ac, get_img_predictions, DOWNSAMPLE_RATIO
# import predict_image_cells


def make_image_patches(img, img_name, output_folder, remove_chamber=False, save_img=True, visualise=False):
    if not visualise:
        plt.switch_backend('agg')

    zero_pad = np.zeros((img_rows, img_rows - img_cols), dtype=np.uint8)
    img = np.concatenate((img, zero_pad), axis=1)   # right-edge (width) padding
    image_rows, image_cols = img.shape  # should be 1024*1000

    patch_size = 32
    patches_per_row = int(np.ceil(image_rows/float(patch_size)))
    patches_per_col = int(np.ceil(image_cols/float(patch_size)))
    img_patches = np.ndarray((patches_per_row, patches_per_col, patch_size, patch_size), dtype=np.float32)

    for h in range(0, image_rows, patch_size):
        for w in range(0, image_cols, patch_size):
            cur_patch = img[h:h+patch_size, w:w+patch_size]
            img_patches[int(h/patch_size), int(w/patch_size), ] = cur_patch

            if visualise:
                plt.figure(1)
                plt.imshow(cur_patch)
                plt.figure(2)
                plt.clf()
                plt.imshow(img)
                plt.scatter(x=[w, w+patch_size], y=[h, h+patch_size], c='red', s=2)
                1

            # save the patches prediction purposes
            if save_img:
                patch_name = '{}_h{}_w{}.png'.format(img_name.replace('.png', ''), h, w)
                patch_path = os.path.join(output_folder, patch_name)
                cv2.imwrite(patch_path, cur_patch)
                # plt.imshow(cur_patch)
                # plt.savefig(patch_path)
    return img_patches


def create_pred_img_patches(is_traced=True):
    ac_imgs, img_names = get_traced_images(is_traced)
    output_fname = 'seg_accell_img_patches' if is_traced else 'empty_accell_img_patches'
    output_folder = os.path.join(base_folder, output_fname)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_images = (img_names)
    all_patches = np.ndarray((num_images, 32, 32, 32, 32), dtype=np.float32)
    for idx, img_name in enumerate(img_names):
        img_patches = make_image_patches(ac_imgs[idx, ], img_name, output_folder, remove_chamber=False, save_img=False, visualise=True)
        all_patches[idx, ] = img_patches

    np.save('{}/img_patches.npy'.format(output_folder), all_patches)
    # save img names
    with open('{}/img_names.txt'.format(output_folder), 'w') as fout:
        for img_name in img_names:
            fout.write('{}\n'.format(img_name))
    return all_patches


def load_patch_npy(is_traced=True):
    output_fname = 'seg_accell_img_patches' if is_traced else 'empty_accell_img_patches'
    npy_file = os.path.join(base_folder, output_fname, 'img_patches.npy')
    all_patch_data = np.load(npy_file)
    img_names = []
    img_file = os.path.join(base_folder, output_fname, 'img_names.txt')
    with open(img_file, 'r') as fin:
        for l in fin:
            img_names.append(l.rstrip())
    return all_patch_data, img_names


def get_ac_preds_for_images(is_traced=True):
    from train import get_unet
    from analyser import predict_image_mask

    ac_imgs, img_names = get_traced_images(is_traced)

    pred_npy = os.path.join(base_folder, 'traced_ac_mask_preds.npy' if is_traced else 'empty_ac_mask_preds.npy')
    if os.path.isfile(pred_npy):
        pred_data = np.load(pred_npy)
        return pred_data

    # otherwise predict segmentations
    model = get_unet()
    weight_path = '{}/{}'.format(results_folder, test_weights)
    model.load_weights(weight_path)

    data_shape = ac_imgs.shape
    num_images = data_shape[0]
    pred_data = np.ndarray(data_shape, dtype=np.float32)
    for idx in range(num_images):
        cur_img = ac_imgs[idx, ]
        mask_pred, strip_preds = predict_image_mask(model, cur_img)  # slice, predict and re-constitute
        pred_data[idx, ] = mask_pred
        # np.save('{}/{}_preds.npy'.format(base_folder, img_names[idx]), strip_preds)
    np.save(pred_npy, pred_data)

    return ac_imgs, img_names, pred_data


def get_traced_images(is_traced=True):
    target_folder = orig_img_folder if is_traced else empty_img_folder
    if is_traced:
        save_path = os.path.join(base_folder, 'traced_ac_images.npy')
    else:
        save_path = os.path.join(base_folder, 'empty_ac_images.npy')

    images = sorted(os.listdir(target_folder))
    real_images = [x for x in images if 'mask' not in x and '.png' in x]
    if os.path.isfile(save_path):
        img_npy = np.load(save_path)
        return img_npy, real_images

    total = len(real_images)
    img_npy = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    counter = 0
    for image in real_images:
        cur_image = cv2.imread(os.path.join(target_folder, image), cv2.IMREAD_GRAYSCALE)
        img_npy[counter, ] = cur_image
        counter +=1

    np.save(save_path, img_npy)
    return img_npy, real_images


def num_cells_within_seg():
    return


def visualize_all_preds(all_patch_data, img_names, pred_dict, true_dict):
    patch_size = 32
    for key, value in sorted(pred_dict.items()):
        key_toks = key.split('_')
        img_key = '_'.join(key_toks[:-2])
        h_adj = int(key_toks[-2].replace('h', ''))
        w_adj = int(key_toks[-1].replace('w', ''))
        img_base = img_key+'.png'
        img_idx = img_names.index(img_base)
        # img = all_patch_data[img_idx, :]
        img = cv2.imread('{}/{}'.format(orig_img_folder, img_base), cv2.IMREAD_GRAYSCALE)
        cur_patch = all_patch_data[img_idx, int(h_adj/patch_size), int(w_adj/patch_size), ]
        cur_patch2 = img[h_adj:h_adj+patch_size, w_adj:w_adj+patch_size]
        np.sum(cur_patch==cur_patch2)   # should be patch_size*patch_size

        # plt.figure(1)
        # plt.imshow(img)
        # plt.title('current img')
        #
        # plt.figure(2)
        # plt.imshow(cur_patch)
        # plt.title('current patch in img')

        pred_coords = value
        true_coords = true_dict[img_key]

        fig1, ax1 = plt.subplots(1)
        ax1.imshow(cur_patch)
        for coord in pred_coords:  # add cells to image
            (x1, y1, x2, y2, prob) = unpack_pred_coord()
            ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=1))

        true_coords_in_patch = []
        for coord in true_coords:  # add cells to image
            (x, y) = coord
            if (x >= w_adj and x < w_adj+32) and (y >= h_adj and y < h_adj+32):
                true_coords_in_patch.append(coord)
                ax1.scatter(x=[x - w_adj], y=[y - h_adj], c='white', s=2)

        fig2, ax2 = plt.subplots(1)
        ax2.imshow(img)
        ax2.add_patch(patches.Rectangle((w_adj, h_adj), patch_size, patch_size, fill=False, color='yellow', linewidth=1))
        for coord in pred_coords:  # add cells to image
            (x1, y1, x2, y2, prob) = unpack_pred_coord(coord)
            ax2.add_patch(patches.Rectangle((x1+w_adj, y1+h_adj), x2 - x1, y2 - y1, fill=False, color='red', linewidth=1))

        for coord in true_coords:  # add cells to image
            (x, y) = coord
            ax2.scatter(x=[x], y=[y], c='white', s=2)
        print(key, img_key, len(pred_coords), len(true_coords_in_patch))

    return


def combined_patch_preds(is_traced=True):
    # get patch data for plotting purposes
    all_patch_data, img_names = load_patch_npy(is_traced)

    # read coords file into dictionary
    true_dict = get_true_coords(json_folder)
    # parse coords file
    pred_dict = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'))
    pred_dict = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'))
    # visualize_all_preds(all_patch_data, img_names, pred_dict, true_dict)

    # combined_pred_dict = recombine_predictions(pred_dict, all_patch_data, img_names)    # combined predicted coords
    combined_pred_dict = recombine_predictions(pred_dict)  # combined predicted coords
    visualise_pred_vs_truth(true_dict, combined_pred_dict)

    # pred closeness based on predicted and actual
    # precision/sensitivity measures overall
    # precision/sensitivity measures within seg chamber
    return


def recombine_predictions(pred_dict, patch_data=None, img_names=None):
    visualise = False
    # if patch_data is not None or img_names is not None:
    #     visualise=True

    combined_pred_dict = {}
    for key, value in pred_dict.items():
        key_toks = key.split('_')
        img_key = '_'.join(key_toks[:-2])
        h_adj = int(key_toks[-2].replace('h', ''))
        w_adj = int(key_toks[-1].replace('.png', '').replace('w', ''))

        adj_value = []
        for coord in value:
            x1, y1, x2, y2, prob = unpack_pred_coord(coord=coord)
            adj_value.append((x1+w_adj, y1+h_adj, x2+w_adj, y2+h_adj, prob))
            # adj_value.append((x1+h_adj, y1+w_adj, x2+h_adj, y2+w_adj))

        if img_key in combined_pred_dict:
            combined_pred_dict[img_key] += adj_value
        else:
            combined_pred_dict[img_key] = adj_value

        if visualise:
            # cur_idx = img_names.index(img_key + '.png')
            # cur_patch = patch_data[cur_idx, int(h_adj/32), int(w_adj/32), ]
            # # visualise_on_patch(key, cur_patch, h_adj, w_adj)

            # read image and visualise
            cur_patch = cv2.imread(os.path.join('z:/yue/pepple/pepple_test_data2_avg/', '{}.png'.format(key)), cv2.IMREAD_GRAYSCALE)
            print(patch_stats(cur_patch))
            plot_img_boxes(key, cur_patch, value, fig_num=1)
            plt.title('cell in patch')

            # patch in big image
            cur_img = cv2.imread(os.path.join('z:/yue/pepple/accell/segmentations/', '{}.png'.format(img_key)), cv2.IMREAD_GRAYSCALE)
            plot_img_boxes(img_key, cur_img, [(w_adj, h_adj, w_adj+cur_patch.shape[1], h_adj+cur_patch.shape[1])], fig_num=3, color='red')
            plt.title('patch in orig image')

            # cell in big image
            plot_img_boxes(img_key, cur_img, adj_value, fig_num=2)
            plt.title('cell in orig image')
    return combined_pred_dict


def unpack_pred_coord(coord):
    if len(coord)==5:
        x1, y1, x2, y2, prob = coord
    else:
        x1, y1, x2, y2 = coord
        prob = None
    return x1, y1, x2, y2, prob


def visualise_pred_vs_truth(true_dict, combined_pred_dict, class_type='cell'):
    for key, value in combined_pred_dict.items():
        true_data = true_dict[key]
        img = plot_img_coords(orig_img_folder, key, true_data)
        # plot_img_boxes(key, img, value, np.asarray(true_data))
        # plot_img_boxes(key, img, value, None)
        plot_img_boxes(key, img, value, np.asarray(true_data))
        plt.title('{} predictions for {}'.format(class_type, key))
    return


# multi-class on same plot
def visualise_preds_multi_class(true_dict, combined_pred_dicts, class_names):
    colors = ['white', 'yellow', 'pink']
    file_names = combined_pred_dicts[0].keys()

    for fname in file_names:
        true_coords = np.asarray(true_dict[fname])
        fpath = os.path.join(orig_img_folder, fname + '.png')
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img)
        ax1.scatter(x=true_coords[:,0], y=true_coords[:, 1], c='red', s=2)

        for idx, combined_pred_dict in enumerate(combined_pred_dicts):
            class_type = class_names[idx]
            color = colors[idx]
            file_class_coords = combined_pred_dict[fname]
            for box_coords in file_class_coords:
                x1, y1, x2, y2, prob = unpack_pred_coord(box_coords)
                if y1 > img_rows or y2 > img_rows:
                    print('y sizing issue for {}; coords={}'.format(fname, box_coords))
                if x1 > img_cols or x2 > img_cols:
                    print('y sizing issue for {}; coords={}'.format(fname, box_coords))
                ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=1))

        plt.title('{} predictions for {}'.format(class_names, fname))
    return


def patch_stats(img_patch, visualise=False):
    patch_mean = np.mean(img_patch)
    patch_std = np.std(img_patch)
    # np.histogram
    # plt.hist(np.flatten(img_patch))
    if visualise:
        plt.figure()
        plt.imshow(img_patch)
    return patch_mean, patch_std, img_patch.shape


def plot_img_boxes(key, img, box_data, true_coords=[], color='white', fig_num=1):
    # fig1, ax1 = plt.subplots(1)
    # ax1.imshow(img)
    plt.figure(fig_num)
    plt.imshow(img)
    for box_coords in box_data:
        if len(box_coords)==4:
            x1, y1, x2, y2 = box_coords
        else:
            x1, y1, x2, y2, prob = box_coords
        if y1 > img_rows or y2 > img_rows:
            print('y sizing issue for {}; coords={}'.format(key, box_coords))
        if x1 > img_cols or x2 > img_cols:
            print('y sizing issue for {}; coords={}'.format(key, box_coords))
        plt.axes().add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=1))

    if len(true_coords)>0:
        plt.scatter(x=true_coords[:,0], y=true_coords[:, 1], c='red', s=2)
    return


def get_true_coords(folder, visualise=False):
    json_files = [x for x in os.listdir(folder) if '.json' in x]
    coord_dict = {}
    for json_file in json_files:
        fname = json_file.replace('.json', '')
        file_data = get_coords(os.path.join(folder, json_file))
        file_coords = []
        for coords in file_data:
            x_coords = coords['mousex']
            y_coords = coords['mousey']
            for idx, x_coord in enumerate(x_coords):
                file_coords.append((x_coord, y_coords[idx]))
        coord_dict[fname] = file_coords

        if visualise:
            plot_img_coords(orig_img_folder, fname, file_coords)
    return coord_dict


def plot_img_coords(folder, fname, file_coords):
    fpath = os.path.join(folder, fname+'.png')
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    plt.figure(1)
    plt.clf()
    plt.imshow(img)
    file_coords_np = np.asarray(file_coords)
    plt.scatter(x=file_coords_np[:, 0], y=file_coords_np[:, 1], c='red', s=2)
    return img


# copied functions over from analyseBoxResults
def get_predicted_coords(folder, file_name="coords_ac.txt"):
    predicted_dict = {}
    with open("{}/{}".format(folder, file_name)) as fin:
        for l in fin:
            arr = l.rstrip().split("\t")
            # file_name = arr[0].replace('.png', '')
            file_name = arr[0]
            if file_name in predicted_dict:
                predicted_dict[file_name] += literal_eval(arr[1])
            else:
                predicted_dict[file_name] = literal_eval(arr[1])
    return predicted_dict


def get_coords(coord_file):
    fin = open(coord_file).read()
    json_data = json.loads(fin)
    return json_data


# utils
def get_file_names(folder):
    files = os.listdir(folder)
    img_files = [x for x in files if 'mask' not in x and '.png' in x]
    return img_files


def review_preds(is_traced=True):
    from analyser import predict_image_mask, combine_predicted_strips_into_image, load_params
    ac_imgs, img_names = get_traced_images(is_traced)

    pred_npy = os.path.join(base_folder, 'traced_ac_mask_preds.npy' if is_traced else 'empty_ac_mask_preds.npy')
    ac_preds = np.load(pred_npy)

    # compare against individual images
    for idx, img_name in enumerate(img_names):
        img_npy = '{}/{}_preds.npy'.format(base_folder, img_name)
        if os.path.isfile(img_npy):
            img_preds = np.load(img_npy)
        else:
            continue
        plt.figure(1)
        plt.clf()
        plt.imshow(ac_imgs[idx,])
        plt.figure(2)
        plt.clf()
        plt.imshow(ac_preds[idx, ])

        raw_rows = img_rows
        raw_cols = img_cols
        cropped_rows = 496
        cropped_cols = 128
        pixel_overlap = 8
        raw_aug = (raw_rows - cropped_rows) / pixel_overlap * (raw_cols - cropped_cols) / pixel_overlap
        img_combined = combine_predicted_strips_into_image(img_preds, 0, num_aug=int(raw_aug), img_rows=raw_rows,
                                                           img_cols=raw_cols, debug_mode=False)
        plt.figure(3)
        plt.clf()
        plt.imshow(img_combined)
    return


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(np.ceil(result.size/2)):]


def simple_corr_analysis(data):
    nrows, ncols = data.shape
    row_corrs = []
    for idx in range(nrows):
        x = data[idx, ]
        row_corrs.append(autocorr(x))

    col_corrs = []
    for idx in range(ncols):
        x = data[:, idx]
        col_corrs.append(autocorr(x))

    return row_corrs, col_corrs


def compare_img_conversions():
    # visualise some test images
    # tiffs vs png vs downsampled png
    tiff_path = './acseg/Inflamed/20170703mouse6_Day2_Right/20170703mouse6_Day2_Right (656).tiff'
    png_path = './Inflamed_201703mouse6_Day2_Right_656_from_tiff_orig.png'
    png_ds_path = './Inflamed_201703mouse6_Day2_Right_656_from_tiff.png'
    png_seg_path = './Kathryn-Inflamed_20170703mouse6_Day2_Right_656.png'
    png_seg_ds_path = './Inflamed_20170703mouse6_Day2_Right_656_ds.png'     # this is from png

    tiff_img = cv2.imread(tiff_path, cv2.IMREAD_GRAYSCALE)
    png_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    png_ds_img = cv2.imread(png_ds_path, cv2.IMREAD_GRAYSCALE)

    png_seg_img = cv2.imread(png_seg_path, cv2.IMREAD_GRAYSCALE)
    png_seg_ds_img = cv2.imread(png_seg_ds_path, cv2.IMREAD_GRAYSCALE)

    # some stats
    patch_stats(tiff_img, visualise=True)
    patch_stats(png_img, visualise=True)
    patch_stats(png_seg_img, visualise=True)
    np.sum(png_img==png_seg_img)    # these are the same

    patch_stats(png_ds_img, visualise=True)
    patch_stats(png_seg_ds_img, visualise=True)
    np.sum(png_ds_img==png_seg_ds_img)  # but scaling 50% from tiff and png are not - though similar
    np.mean(abs(png_ds_img.astype(np.float32) - png_seg_ds_img.astype(np.float32)))

    # look at higher moments
    from scipy.stats import moment
    plt.figure(1);
    plt.clf();
    plt.grid()
    # plt.title()
    # plt.hist(png_img.flatten(), bins=[1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150])
    # plt.hist([png_img.flatten(), png_ds_img.flatten(), png_seg_ds_img.flatten(), png_img[::2, ::2].flatten()],
    #          bins=[1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150])
    # plt.legend(['png from tiff', 'scaled png from tiff', 'scaled png from png', 'ds png from png'])
    plt.hist([png_ds_img.flatten(), png_seg_ds_img.flatten(), png_img[::2, ::2].flatten()],
             bins=[1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150])
    plt.legend(['scaled png from tiff', 'scaled png from png', 'ds png from png'])

    print(moment(png_img.flatten(), moment=[1, 2, 3, 4, 5]))
    print(moment(png_ds_img.flatten(), moment=[1, 2, 3, 4, 5]))
    print(moment(png_seg_ds_img.flatten(), moment=[1, 2, 3, 4, 5]))
    print(moment(png_img[::2, ::2].flatten(), moment=[1, 2, 3, 4, 5]))

    # auto correlate for some middle columns
    row_corrs, col_corrs = simple_corr_analysis(png_img)
    row_corrs, col_corrs = simple_corr_analysis(png_ds_img)
    row_corrs, col_corrs = simple_corr_analysis(png_seg_ds_img)
    row_corrs, col_corrs = simple_corr_analysis(png_img[::2, ::2])
    return


def pred_experiment():
    # tiff_path = './acseg/Inflamed/20170703mouse6_Day2_Right/20170703mouse6_Day2_Right (656).tiff'
    png_path = './Inflamed_201703mouse6_Day2_Right_656_from_tiff_orig.png'
    png_ds_path = './Inflamed_201703mouse6_Day2_Right_656_from_tiff.png'
    png_seg_path = './Kathryn-Inflamed_20170703mouse6_Day2_Right_656.png'
    png_seg_ds_path = './Inflamed_20170703mouse6_Day2_Right_656_ds.png'     # this is from png

    png_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)    # from tiff
    png_ds_img = cv2.imread(png_ds_path, cv2.IMREAD_GRAYSCALE)  # from tiff
    png_seg_img = cv2.imread(png_seg_path, cv2.IMREAD_GRAYSCALE)    # manually segmented png
    png_seg_ds_img = cv2.imread(png_seg_ds_path, cv2.IMREAD_GRAYSCALE)  # from png
    png_naive_ds_img = png_img[::2, ::2]    # naive downsampling

    from train import get_unet
    from analyser import predict_image_mask
    model = get_unet()
    weight_path = '{}/{}'.format(results_folder, test_weights)
    model.load_weights(weight_path)

    png_ds_mask, strip_ds_preds = predict_image_mask(model, png_ds_img)
    png_seg_ds_mask, strip_seg_ds_preds = predict_image_mask(model, png_seg_ds_img)
    png_naive_ds_mask, strip_naive_dspreds = predict_image_mask(model, png_naive_ds_img)

    # grab prediction for this image in pred stack
    is_traced = True
    ac_imgs, img_names = get_traced_images(is_traced)
    pred_npy = os.path.join(base_folder, 'traced_ac_mask_preds.npy' if is_traced else 'empty_ac_mask_preds.npy')
    ac_preds = np.load(pred_npy)

    pred_idx = img_names.index(png_seg_path.replace('./', ''))
    plt.figure()
    plt.imshow(ac_preds[pred_idx, ])
    return


def count_num_volume(folder):
    files = os.listdir(folder)
    vol_name_dict = {}
    for file in files:
        if '.png' in file and 'mask' not in file:
            vol_name = '_'.join(file.split('_')[1:-1])
            if vol_name not in vol_name_dict:
                vol_name_dict[vol_name] = [file]
            else:
                vol_name_dict[vol_name].append(file)
    return vol_name_dict, len(vol_name_dict.keys())


# OLD code pepple/accell/seg_accell_img_patches
def combined_patch_class_preds(is_traced=True, results_folder='pepple_test_data', suffix='_ac_training3_32_32', do_avg=True):
    # get patch data for plotting purposes
    all_patch_data, img_names = load_patch_npy(is_traced)

    # read coords file into dictionary
    # labelled_cell_jsons = json_folder
    if do_avg:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered')
    else:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered_1scan')
    true_dict = get_true_coords(labelled_cell_jsons)
    # # parse coords file
    # pred_dict_cell = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'), 'coords_cell.txt')
    # pred_dict_med = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'), 'coords_cell_medium.txt')
    # pred_dict_lite = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'), 'coords_cell_lite.txt')
    # # visualize_all_preds(all_patch_data, img_names, pred_dict, true_dict)
    #
    # # combined_pred_dict = recombine_predictions(pred_dict, all_patch_data, img_names)    # combined predicted coords
    # combined_pred_dict_cell = recombine_predictions(pred_dict_cell, all_patch_data, img_names=img_names)  # combined predicted coords
    # combined_pred_dict_med = recombine_predictions(pred_dict_med, all_patch_data, img_names=img_names)  # combined predicted coords
    # combined_pred_dict_lite = recombine_predictions(pred_dict_lite, all_patch_data, img_names=img_names)  # combined predicted coords

    # new format coords
    num_classes=3
    if num_classes==3:
        all_dict = predict_image_cells.parse_predictions(results_folder, classes=['cell', 'cell_medium', 'cell_lite'],
                                                        suffix=suffix, true_dict=None)
        combined_pred_dict_cell = all_dict['cell']
        combined_pred_dict_med = all_dict['cell_medium']
        combined_pred_dict_lite = all_dict['cell_lite']
    elif num_classes==2:
        all_dict = predict_image_cells.parse_predictions(results_folder, classes=['cell', 'cell_lite'],
                                                         suffix='_ac_training2_32_32', true_dict=None)
        combined_pred_dict_cell = all_dict['cell']
        combined_pred_dict_med = all_dict['cell_lite']
    elif num_classes==1:
        all_dict = predict_image_cells.parse_predictions(results_folder, classes=['cell'],
                                                         suffix='_ac_training_32_32', true_dict=None)
        combined_pred_dict_cell = all_dict['cell']
        # combined_pred_dict_med = all_dict['cell_lite']

    # # TODO - add chamber seg for sensitivity and precision analysis!
    # visualise_pred_vs_truth(true_dict, combined_pred_dict_cell, class_type='cell')
    # visualise_pred_vs_truth(true_dict, combined_pred_dict_med, class_type='cell_medium')
    # visualise_pred_vs_truth(true_dict, combined_pred_dict_lite, class_type='cell_lite')
    # # plot all classes together
    # visualise_preds_multi_class(true_dict, [combined_pred_dict_cell, combined_pred_dict_med], class_names=['cell', 'cell_medium'])
    # visualise_preds_multi_class(true_dict, [combined_pred_dict_cell, combined_pred_dict_med, combined_pred_dict_lite], class_names=['cell', 'cell_medium', 'cell_lite'])

    raw_images, converted_imgs, img_names, img_preds = predict_image_cells.get_img_predictions(folder='./accell/segmentations')
    # combined_dict = combine_class_preds(combined_pred_dict_cell, combined_pred_dict_med)
    # combined_dict = combine_class_preds(combined_dict, combined_pred_dict_lite)   # overkill
    # fnames = combined_dict.keys()
    if num_classes==1:
        fnames = combined_pred_dict_cell.keys()
    else:
        fnames = list(set(list(combined_pred_dict_med.keys()) + list(combined_pred_dict_cell.keys()) + list(combined_pred_dict_lite.keys())))
    # fnames = ['DeRuyter-Inflamed_20170710mouse2_Day1_Right_343', #'DeRuyter-Inflamed_20170710mouse4_Day1_Right_867',
    #           'DeRuyter-Inflamed_20170710mouse2_Day1_Right_665', 'DeRuyter-Inflamed_20170703mouse4_Day2_Right_626']

    orig_scale = False
    conservative = 5
    no_mid = 1  # enforce??
    ac_threshold = .2   # smaller threshold allows bigger ac chamber prediction
    show_medium = False
    if num_classes==3:
        show_medium = True
    show_all = False
    if num_classes==3:
        save_dir = 'pepple_test_results_3class'
    elif num_classes==2:
        save_dir = 'pepple_test_results_2class'
    elif num_classes==1:
        save_dir = 'pepple_test_results_1class'
    save_dir = '{}_{}'.format(save_dir, suffix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    valid_coords_dict = {}  # for valid coords
    for fname in fnames:
        f_idx = img_names.index('{}.png'.format(fname))

        # if fname in ['DeRuyter-Inflamed_20170710mouse3_Day1_Right_497',
        #              'DeRuyter-Inflamed_20170703mouse4_Day2_Right_595',
        #              'DeRuyter-Inflamed_20170710mouse4_Day1_Right_867']:
        #     ac_threshold = .15
        # elif fname in ['DeRuyter-Inflamed_20170710mouse2_Day1_Right_685', 'DeRuyter-Inflamed_20170710mouse2_Day1_Right_343',
        #                'DeRuyter-Inflamed_20170710mouse2_Day1_Right_665']:
        #     ac_threshold = .95

        preds_class_dict = {}
        if fname in combined_pred_dict_cell:
            preds_class_dict['cell'] = combined_pred_dict_cell[fname]
        if num_classes>1 and fname in combined_pred_dict_med:
            preds_class_dict['cell_medium'] = combined_pred_dict_med[fname]

        true_centroids = true_dict[fname]
        # true_centroids = []
        out_dict = visualise_preds_minus_chamber(fname, png_orig_scale=raw_images[f_idx, ],
                                                 png_scaled=converted_imgs[f_idx, ], chamber_preds=img_preds[f_idx, ],
                                                 file_dict=preds_class_dict, true_centroids=true_centroids,
                                                 orig_scale=orig_scale, conservative=conservative, no_mid=no_mid,
                                                 ac_threshold=ac_threshold, show_medium=show_medium, show_all=show_all,
                                                 save_dir=save_dir)
        valid_coords_dict.update(out_dict)

    # pred closeness based on predicted and actual
    # precision/sensitivity measures overall
    # precision/sensitivity measures within seg chamber
    with open('test_{}_o{}_c{}_m{}_t{}_sm{}_s{}.json'.format(suffix, int(orig_scale), conservative, no_mid, ac_threshold,
                                                             int(show_medium), int(show_all)), 'w') as fout:
        json.dump(valid_coords_dict, fout)
    fout.close()

    for key in sorted(valid_coords_dict.keys()):
        cells = valid_coords_dict[key]['cell'] if 'cell' in valid_coords_dict[key] else []
        cell_med = valid_coords_dict[key]['cell_medium'] if 'cell_medium' in valid_coords_dict[key] else []
        print(key, len(cells), len(cell_med), valid_coords_dict[key]['chamber_size'])

    # combined_dict on cell and cell medium
    combined_dict = {}
    for fname, fdict in valid_coords_dict.items():
        combined_dict[fname] = fdict['cell'] if 'cell' in fdict else [] + fdict['cell_medium'] if 'cell_medium' in fdict else []

    found_dict, missed_dict = check_missed_helper(true_dict, combined_dict, pixel_lim=3)
    total_predicted = [len(found_cells) for fname, found_cells in found_dict.items()]
    total_cells = np.sum([len(true_dict[fname]) for fname in found_dict.keys()])
    # total_cells = np.sum([len(f_coords) for fname, f_coords in true_dict.items()])
    print(total_cells, np.sum(total_predicted), np.sum(total_predicted) / total_cells)
    return


def combine_class_preds(pred_dict1, pred_dict2):
    combined_dict = pred_dict1.copy()
    # combined_dict.update(combined_pred_dict_med)  # this overrides for same key values
    for key, coords in pred_dict2.items():
        if key not in combined_dict:
            combined_dict[key] = coords
        else:
            combined_dict[key] += coords
    return combined_dict


def euclid_dist(coord_1, coord_2):
    x_1, y_1 = coord_1
    x_2, y_2 = coord_2

    euclidean_dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
    return euclidean_dist


# sensitivity - check for each labelled if it was predicted
def check_missed_helper(true_dict, combined_dict, pixel_lim=3, imgs=[]):
    found_dict = {}
    missed_dict = {}
    for img_name, img_coords in true_dict.items():  # name then simply labelled coords
        for jdx, img_coord in enumerate(img_coords):    # for each labelled cell in each image
            if len(img_coord) > 2:
                img_center = (img_coord[0] + img_coord[2]) / 2., (img_coord[1] + img_coord[3]) / 2.
            else:
                img_center = img_coord

            pred_coords = []
            img_key = img_name
            if img_key in combined_dict:
                pred_coords = combined_dict[img_key]    # combine into 1 dict for easy use

            matched_pred = False
            for pred_coord in pred_coords:  # check if this labelled cell (img_coord) was predicted
                if len(pred_coord)>4:
                    x1, y1, x2, y2, _ = pred_coord
                elif len(pred_coord)==4:
                    x1, y1, x2, y2 = pred_coord
                elif len(pred_coord)==2:
                    x1, y1 = pred_coord
                    x2, y2 = x1, y1
                pred_center = (x1+x2)/2., (y1+y2)/2.
                coord_dist = euclid_dist(img_center, pred_center)
                if coord_dist<=pixel_lim:
                    matched_pred = True
                    break   # already matched

            img_coord = img_coord.tolist() if isinstance(img_coord, np.ndarray) else list(img_coord)
            if matched_pred:    # labelled cell matched in pred
                if img_name in found_dict:
                    found_dict[img_name].append(img_coord)
                else:
                    found_dict[img_name] = [img_coord]
            else:
                if img_name in missed_dict:
                    missed_dict[img_name].append(img_coord)
                else:
                    missed_dict[img_name] = [img_coord]

    # # visualise missed preds
    # if imgs:
    #     out_folder = os.path.join('sensitivity_analysis', pred_type)
    #     if not os.path.isdir(out_folder):
    #         os.makedirs(out_folder)
    #     img_names = true_dict.keys()
    #     total_missed = 0
    #     for fname, missed_coords in missed_dict.items():
    #         img_index = img_names.index(fname.replace('.png', ''))
    #         img = imgs[img_index]
    #         plt.clf()
    #         plt.imshow(img)
    #         for missed_coord in missed_coords:
    #             total_missed += 1
    #             x1, y1, x2, y2 = missed_coord
    #             plt.scatter(x=[x1, x1, x2, x2], y=[y1, y2, y1, y2], c='lime', s=2)
    #         outname = os.path.join(out_folder, fname)
    #         plt.savefig(outname)
    return found_dict, missed_dict


def generate_img_html(folder, fbase):
    # img_files = sorted([x for x in os.listdir(folder) if x.endswith('png')])
    # for idx, img in enumerate(img_files):
    for idx in range(0, 1201):
        # img_name = os.path.join(folder, 'pred_{} ({}).png'.format(fbase, idx))
        img_name = 'pred_{} ({}).png'.format(fbase, idx)
        with open(os.path.join(folder, 'pred_images.html'), 'a') as fout:
            fout.write('<img src="{}"/>\n'.format(img_name))
        fout.close()
    return


def plot_cell(class_cell, cname, size=2, orig_scale=False, all_white=False, add_jitter=True,
              color_dict={'cell': 'red', 'cell_medium': 'yellow', 'cell_lite': 'white'}):
    x1, y1, x2, y2 = class_cell
    if all_white:
        # color ='white'
        color ='red'    # more obvious
    else:
        color = color_dict[cname]

    x = (x1+x2)/2.
    y = (y1+y2)/2.
    if not orig_scale:
        x = x /predict_image_cells.DOWNSAMPLE_RATIO
        y = y / predict_image_cells.DOWNSAMPLE_RATIO

    # add jitter in case overlaid
    if add_jitter:
        x += (np.random.rand(1) - 0.5) * 3
        y += (np.random.rand(1) - 0.5) * 3

    plt.scatter(x=x, y=y, c=color, s=size)
    return


def visualise_preds_minus_chamber(fname, png_orig_scale, png_scaled, chamber_preds, file_dict, true_centroids=[],
                                  orig_scale=False, conservative=5, no_mid=1, ac_threshold=0.9, show_medium=True,
                                  show_all=False, save_dir=None):
    if png_orig_scale is None or len(png_orig_scale)==0:
        fpath = os.path.join(orig_img_folder, fname + '.png')
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    else:
        img = png_orig_scale

    # plot original img
    plt.figure(1)
    plt.clf()
    if orig_scale:
        plt.subplot(131)
        plt.imshow(img)
        plt.axis('off')
    else:
        plt.subplot(131)
        plt.imshow(png_scaled)

    # treat chamber properly
    plt.subplot(132)
    plt.imshow(png_scaled)
    if orig_scale: plt.axis('off')  # to avoid showing sleight-of-hand change in scales

    mid_limits, mid_min, mid_max = predict_image_cells.any_middle_stripes(img, avg_period=10, intensity_threshold=180)
    chamber_limits, chamber_size, mean_x, mean_y = predict_image_cells.calc_img_chamber_size(chamber_preds, pred_threshold=ac_threshold)

    if no_mid and (mid_min is not None and mid_max is not None):
        chamber_limits_no_center = chamber_limits[(chamber_limits[:, 1]*predict_image_cells.DOWNSAMPLE_RATIO<mid_min) |
                                                  (chamber_limits[:, 1]*predict_image_cells.DOWNSAMPLE_RATIO>mid_max), ]
    else:
        chamber_limits_no_center = chamber_limits
    plt.scatter(x=chamber_limits_no_center[:, 1], y=chamber_limits_no_center[:, 0], c='yellow', s=1)
    # if save_dir is not None:
    #     plt.savefig(os.path.join(save_dir, '{}_chamber_seg.png'.format(fname)), bbox_inches='tight')

    # overlay red cells in segmented chamber
    if len(true_centroids) > 0:
        true_centroids = np.asarray(true_centroids)
        # plt.scatter(x=true_centroids[:, 0], y=true_centroids[:, 1], c='red', s=2)
        # since labelling on 1024,1000 - adjust appropriately
        plt.scatter(x=true_centroids[:, 0]/predict_image_cells.DOWNSAMPLE_RATIO, y=true_centroids[:, 1]/predict_image_cells.DOWNSAMPLE_RATIO, c='lime', s=2)

    # now figure out predicted cells in chamber and return these
    plt.subplot(133)
    if orig_scale:
        plt.imshow(png_orig_scale)
        plt.axis('off')
    else:
        plt.imshow(png_scaled)

    # overlay red cells in segmented chamber
    if len(true_centroids) > 0:
        true_centroids = np.asarray(true_centroids)
        plt.scatter(x=true_centroids[:, 0]/predict_image_cells.DOWNSAMPLE_RATIO, y=true_centroids[:, 1]/predict_image_cells.DOWNSAMPLE_RATIO, c='lime', s=2)

    combined_preds = {}  # initialize
    combined_preds[fname] = {}
    combined_preds[fname]['chamber_size'] = chamber_size

    pred_classes = file_dict.keys()
    for idx, cname in enumerate(pred_classes):
        if cname=='cell_lite':  # disregard as too much noise
            continue
        if not show_medium and cname=='cell_medium':
            continue
        combined_preds[fname][cname] = []   # init

        class_cells = file_dict[cname]
        for class_cell in class_cells:
            # plot_cell(class_cell, cname, orig_scale=orig_scale, add_jitter=False, all_white=True)    # and visualise
            if show_all or predict_image_cells.is_cell_in_ac(class_cell, chamber_limits, mean_x, mean_y,
                                                             conservative=conservative, img=img, no_mid=no_mid,
                                                             mid_min=mid_min, mid_max=mid_max):
                combined_preds[fname][cname].append(class_cell)    # if in chamber update count
                plot_cell(class_cell, cname, orig_scale=orig_scale, add_jitter=False, all_white=True)  # and visualise

    # save plot
    if save_dir is not None:
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir,
                                 '{}_o{}_c{}_m{}_t{}_sm{}_s{}.png'.format(fname, int(orig_scale), conservative,
                                                                          no_mid, ac_threshold, int(show_medium),
                                                                          int(show_all))),
                    bbox_inches='tight')

    return combined_preds   # predicted cells in chamber


def get_true_coords_by_class(folder, coord_file="training_coords.txt", path_prefix=''):
    true_dict_by_class = {}

    with open(os.path.join(folder, coord_file), 'r') as fin:
        for l in fin:
            arr = l.rstrip().split(",")  # faster-rcnn input format
            file_name = arr[0].replace(path_prefix, '')
            cur_coord = [tuple([int(a) for a in arr[1:-1]])]
            cur_class = arr[-1]

            # add to class dict
            if cur_class not in true_dict_by_class:
                true_dict_by_class[cur_class] = {}  # init
                true_dict_by_class[cur_class][file_name] = cur_coord
            else:
                if file_name in true_dict_by_class[cur_class]:
                    true_dict_by_class[cur_class][file_name] += cur_coord
                else:
                    true_dict_by_class[cur_class][file_name] = cur_coord

    # sanity check
    for cls, cls_dict in true_dict_by_class.items():
        print(cls, np.sum([len(vals) for vals in cls_dict.values()]))  # sanity check

    # FIXME - broken for some reason
    # # add to overall dict
    # true_dict = {}
    # # temp = true_dict_by_class.copy()
    # for cls, cls_dict in true_dict_by_class.items():
    #     print(cls, np.sum([len(val) for val in cls_dict.values()]))
    #     # f_cells = []
    #     for fname, f_cells in cls_dict.items():
    #         if fname in true_dict:
    #             true_dict[fname] += f_cells
    #         else:
    #             true_dict[fname] = f_cells
    # # np.sum([len(vals) for vals in true_dict.values()])  # sanity check
    return true_dict_by_class


def get_true_coords_file(folder, coord_file="training_coords.txt", path_prefix=''):
    true_dict = {}
    with open(os.path.join(folder, coord_file), 'r') as fin:
        for l in fin:
            arr = l.rstrip().split(",")  # faster-rcnn input format
            file_name = arr[0].replace(path_prefix, '')
            cur_coord = [tuple([int(a) for a in arr[1:-1]])]
            cur_class = arr[-1]

            # add to class dict
            if file_name in true_dict:
                true_dict[file_name] += cur_coord
            else:
                true_dict[file_name] = cur_coord
    print('total labelled', np.sum([len(val) for val in true_dict.values()]))
    return true_dict


def sensitivity_analysis(results_folder, coord_file='valid_coords.txt', suffix='', classes=['cell', 'cell_medium', 'cell_lite']):
    num_classes = len(classes)

    sep = '\\'
    result_folder_toks = results_folder.split(sep)
    true_file_path = sep.join(result_folder_toks[:-1])
    path_prefix = '/data/yue/pepple/accell/{}/{}/'.format(result_folder_toks[-2], result_folder_toks[-1])
    true_dict_by_class = get_true_coords_by_class(true_file_path, coord_file=coord_file, path_prefix=path_prefix)
    # check against number of lines found for each class
    true_dict = get_true_coords_file(true_file_path, coord_file=coord_file, path_prefix=path_prefix)
    print('check total labelled cells:', np.sum([len(val) for val in true_dict.values()]))  #number of all labelled cells - compare against file length

    # predicted cells by class
    # # check against vim count of ) in corresponding .txt
    cell_dict = get_predicted_coords(results_folder, file_name='{}_cell{}.txt'.format('coords', suffix))
    print('check predicted cell:', np.sum([len(val) for val in cell_dict.values()]))
    if num_classes>1:
        cell_med_dict = get_predicted_coords(results_folder, file_name='{}_{}{}.txt'.format('coords', classes[1], suffix))
        print('checked predicted cell_med', np.sum([len(val) for val in cell_med_dict.values()]))
    if num_classes>2:
        cell_lite_dict = get_predicted_coords(results_folder, file_name='{}_{}{}.txt'.format('coords', classes[2], suffix))
        print('checked predicted cell_lite', np.sum([len(val) for val in cell_lite_dict.values()]))

    # combine for overall value
    combined_dict = {}
    cell_keys = cell_dict.keys()
    all_keys = list(cell_keys)
    if num_classes>1:
        med_keys = cell_med_dict.keys()
        all_keys = list(set(list(cell_keys) + list(med_keys) ))
    if num_classes>2:
        lite_keys = cell_lite_dict.keys()
        all_keys = list( set( list(cell_keys) +list(med_keys) +list(lite_keys) ))

    for key in all_keys:
        # true_key = '{}.png'.format(key)
        true_key = key
        cell_vals = cell_dict[key] if key in cell_dict else []
        combined_dict[true_key] = cell_vals
        if num_classes>1:
            med_vals = cell_med_dict[key] if key in cell_med_dict else []
            combined_dict[true_key] = cell_vals + med_vals
        if num_classes > 2:
            lite_vals = cell_lite_dict[key] if key in cell_lite_dict else []
            combined_dict[true_key] = cell_vals + med_vals + lite_vals

    # sensitivity
    found_dict, missed_dict = check_missed_helper(true_dict, combined_dict, pixel_lim=3)
    print('total labelled=', np.sum([len(val) for val in true_dict.values()]),
          'found=', np.sum([len(val) for val in found_dict.values()]),
          'missed=', np.sum([len(val) for val in missed_dict.values()]))
    # found and missed should add together to total labelled for sensitivity
    total_predicted = [len(found_cells) for fname, found_cells in found_dict.items()]   # this is total_matched
    total_cells = np.sum([len(true_dict[fname]) for fname in true_dict.keys()])
    # total_cells = np.sum([len(f_coords) for fname, f_coords in true_dict.items()])
    print(suffix, 'all_preds', 'total_true=', total_cells, 'total_matched=', np.sum(total_predicted), 'sensitivity=', np.sum(total_predicted) / total_cells)

    # check by class
    # found overall should be sum of found in each class - NO could be mis-labelled in terms of class!!!
    for cls in classes:
        target_dict = cell_dict
        if cls =='cell_medium':
            target_dict = cell_med_dict
        elif cls=='cell_lite':
            target_dict = cell_lite_dict
        found_class_dict, missed_class_dict = check_missed_helper(true_dict_by_class[cls], target_dict, pixel_lim=3)
        # found for class should be sum of found and missed in class
        print('total labelled for ', cls, np.sum([len(val) for val in true_dict_by_class[cls].values()]),
              'found=', np.sum([len(val) for val in found_class_dict.values()]),
              'missed=', np.sum([len(val) for val in missed_class_dict.values()]))
        total_predicted = [len(found_cells) for fname, found_cells in found_class_dict.items()]
        total_cells = np.sum([len(true_dict_by_class[cls][fname]) for fname in true_dict_by_class[cls].keys()])
        print(suffix, cls, 'total_true=', total_cells, 'total_matched=', np.sum(total_predicted), 'sensitivity=', np.sum(total_predicted) / total_cells)

    # precision
    false_positive_dict, matched_dict = check_false_positives_helper(combined_dict, true_dict, pixel_lim=3)
    print('total predicted=', np.sum([len(val) for val in combined_dict.values()]),
          'matched=', np.sum([len(val) for val in matched_dict.values()]),
          'false_positives=', np.sum([len(val) for val in false_positive_dict.values()]))
    num_false_positives = [len(cells) for cells in false_positive_dict.values()]
    num_preds = [len(cells) for cells in combined_dict.values()]
    print(suffix, 'all_preds', 'total_fp=', np.sum(num_false_positives), 'total_preds=', np.sum(num_preds), 'fpr=', np.sum(num_false_positives) / np.sum(num_preds),
          'precision=', 1 - np.sum(num_false_positives) / np.sum(num_preds))

    for cls in classes:
        target_dict = cell_dict
        if cls=='cell_medium':
            target_dict = cell_med_dict
        elif cls=='cell_lite':
            target_dict = cell_lite_dict
        false_class_dict, matched_class_dict = check_false_positives_helper(target_dict, true_dict_by_class[cls], pixel_lim=3)
        print('total predicted for ', cls, np.sum([len(val) for val in target_dict.values()]),
              'matched=', np.sum([len(val) for val in matched_class_dict.values()]),
              'false_positives=', np.sum([len(val) for val in false_class_dict.values()]))
        num_false_positives = [len(cells) for cells in false_class_dict.values()]
        num_preds = [len(cells) for cells in target_dict.values()]
        print(suffix, cls, 'total_fp=', np.sum(num_false_positives), 'total_preds=', np.sum(num_preds), 'fpr=', np.sum(num_false_positives) / np.sum(num_preds),
              'precision=', 1 - np.sum(num_false_positives) / np.sum(num_preds))

    return


def check_false_positives_helper(combined_dict, true_dict, pixel_lim=3, imgs=[], out_folder=None):
    # flip dynamics vs sensitivity code - look at false positives for prediction
    img_names = true_dict.keys()

    false_positive_dict = {}
    found_dict = {}
    # for each file (fname),
    # then for each predicted cell for file see if false positive against true dict
    for fname, pred_cells in combined_dict.items():
        true_key = fname if fname in true_dict else fname.replace('.png', '')
        if true_key not in true_dict:   # if no true labelled then
            pred_cells = [x.tolist() if isinstance(x, np.ndarray) else list(x) for x in pred_cells]
            false_positive_dict[fname] = pred_cells
            continue
        true_cells = true_dict[true_key]    # true coords for file

        for pred_coord in pred_cells:   # for each predicted cell
            matched_true = False
            if len(pred_coord) > 4:
                x1, y1, x2, y2, _ = pred_coord
            elif len(pred_coord)==4:
                x1, y1, x2, y2 = pred_coord
            elif len(pred_coord)==2:
                x1, y1 = pred_coord
                x2, y2 = x1, y1
            pred_center = (x1 + x2) / 2., (y1 + y2) / 2.

            for true_cell in true_cells:
                if len(true_cell)>2:
                    true_center = (true_cell[0]+true_cell[2])/2., (true_cell[1]+true_cell[3])/2.
                else:
                    true_center = true_cell

                coord_dist = euclid_dist(true_center, pred_center)
                if coord_dist <= pixel_lim:
                    matched_true = True
                    break   # if matched

            pred_coord = pred_coord.tolist() if isinstance(pred_coord, np.ndarray) else list(pred_coord)
            if not matched_true:    # didnt match any true/labelled coords
                if fname in false_positive_dict:
                    false_positive_dict[fname].append(pred_coord)
                else:
                    false_positive_dict[fname] = [pred_coord]
            else:
                if fname in found_dict:
                    found_dict[fname].append(pred_coord)
                else:
                    found_dict[fname] = [pred_coord]

    return false_positive_dict, found_dict


if __name__ == '__main__':
    # get_traced_images(is_traced=True)

    # generate html page of results]
    # generate_img_html(os.path.join('acseg', 'Volume_converted', 'Uninflamed', '20170717mouse2_Day-7_Right', 'pred_figs_unique174'), '20170717mouse2_Day-7_Right')
    # generate_img_html(os.path.join('acseg', 'Volume_converted', 'Uninflamed', '20170710mouse1_Day-7_Left', 'pred_figs_unique174'), '20170710mouse1_Day-7_Left')

    ## for arvo abstract
    # count_num_volume(folder=orig_img_folder)
    # count_num_volume(folder=empty_img_folder)

    # create_pred_img_patches(is_traced=True)
    # get_ac_preds_for_images(is_traced=True)
    # review/visualise ac cell preds for images
    # combined_patch_preds()

    # check sensitivity on un-averaged analysis
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training3_32_32', 'valid'), suffix='_ac_training3_32_32')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training3_32_32', 'train'),
    #                      suffix='_ac_training3_32_32', coord_file='training_coords.txt')

    # sensitivity/precision analysis on validation data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avgNew_32_32', 'valid'), suffix='_weights_s128')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avgNew_32_32', 'valid'), suffix='_weights_s64')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avgNew_32_32', 'valid'), suffix='_weights_s32')
    # # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avgNew_32_32', 'train'), suffix='_weights_s128', coord_file='training_coords.txt')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avgNew_32_32', 'train'), suffix='_weights_s64', coord_file='training_coords.txt')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avgNew_32_32', 'train'), suffix='_weights_s32', coord_file='training_coords.txt')
    # see /pepple/accell/ac_training_avgNew_32_32/summary_results.txt
    # or https://docs.google.com/spreadsheets/d/1qB1ope_JWENw_YL7IaJqPgpsvqrCBnnp_V32wk0YKZ4/edit#gid=0

    # # evaulate on new averaged cell training and its preds
    # combined_patch_class_preds(results_folder='pepple_test_data_avg', suffix='_weights_s128')  # scale=[128, 256, 512]
    # combined_patch_class_preds(results_folder='pepple_test_data_avg', suffix='_weights_s64')  # scale=[64, 128, 256]
    # combined_patch_class_preds(results_folder='pepple_test_data_avg', suffix='_weights_s32')  # scale=[32, 64, 128]
    # combined_patch_class_preds(results_folder='pepple_test_data_avgNew', suffix='_weights_s128')  # scale=[128, 256, 512]
    # combined_patch_class_preds(results_folder='pepple_test_data_avgNew', suffix='_weights_s64')  # scale=[64, 128, 256]
    # combined_patch_class_preds(results_folder='pepple_test_data_avgNew', suffix='_weights_s32')  # scale=[32, 64, 128]

    # # in situ trained results
    # # sensitivity/precision analysis on validation data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'valid'), suffix='_weights_s128', classes=['cell'])
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'valid'), suffix='_weights_s64', classes=['cell'])
    # # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'train'), suffix='_weights_s128', coord_file='train_coords.txt', classes=['cell'])
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'train'), suffix='_weights_s64', coord_file='train_coords.txt', classes=['cell'])
    # # combined_patch_class_preds(results_folder='pepple_test_data_avgNew', suffix='_weights_s128')  # scale=[128, 256, 512]
    # # combined_patch_class_preds(results_folder='pepple_test_data_avgNew', suffix='_weights_s64')  # scale=[64, 128, 256]
    # # combined_patch_class_preds(results_folder='pepple_test_data_avgNew', suffix='_weights_s32')  # scale=[32, 64, 128]

    # # augmented results
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'valid'), suffix='_weights_s128_aug', classes=['cell'])
    # # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'train'), suffix='_weights_s128_aug', coord_file='train_coords.txt', classes=['cell'])
    # # NB s_64_aug should be broken
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'valid'), suffix='_weights_s64_aug', classes=['cell'])
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'train'), suffix='_weights_s64_aug', coord_file='train_coords.txt', classes=['cell'])

    # # insitu 3 classes
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'valid'), suffix='_weights_s128')  # validation
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'train'), suffix='_weights_s128', coord_file='train_coords.txt') # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'valid'), suffix='_weights_s64')  # validation
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'train'), suffix='_weights_s64', coord_file='train_coords.txt') # check fit of data
    # # combined_patch_class_preds(results_folder='pepple_test_data_avgNew', suffix='_weights_s128')  # scale=[128, 256, 512]
    # # combined_patch_class_preds(results_folder='pepple_test_data_avgNew', suffix='_weights_s64')  # scale=[64, 128, 256]
    # # augmented insitu 3 classes
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'valid'), suffix='_weights_s128_aug')  # validation
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'train'), suffix='_weights_s128_aug', coord_file='train_coords.txt') # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'valid'), suffix='_weights_s64_aug')  # validation
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'train'), suffix='_weights_s64_aug', coord_file='train_coords.txt') # check fit of data

    # nickhypo
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'nickHypo', 'valid'), suffix='_weights_s128')    # 70% kathryn on nick
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'nickHypo', 'valid'), suffix='_weights_s64')     # 70% kathryn on nick
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'valid'), suffix='_weights_s128')   # 70% kathryn on 30% kathryn
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'valid'), suffix='_weights_s64')    # 70% kathryn on 30% kathryn
    # # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'train'), suffix='_weights_s128', coord_file='train_coords.txt')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'train'), suffix='_weights_s64', coord_file='train_coords.txt')

    # # 2class
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'valid'), suffix='_weights_s128_2class', classes=['cell', 'cell_medium'])  # 70% kathryn on 30% kathryn
    # # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'train'), suffix='_weights_s128_2class', coord_file='train_coords.txt', classes=['cell', 'cell_medium'])

    # # longer runs
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'valid'), suffix='_weights_s128_aug', classes=['cell'])
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitu_32_32', 'train'), suffix='_weights_s128_aug', classes=['cell'], coord_file='train_coords.txt')  # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'valid'), suffix='_weights_s128_aug')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'train'), suffix='_weights_s128_aug', coord_file='train_coords.txt')  # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'valid'), suffix='_weights_s64_aug')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insituM_32_32', 'train'), suffix='_weights_s64_aug', coord_file='train_coords.txt')  # check fit of data

    # # less zoom
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'valid'), suffix='_weights_s128_2class_z256', classes=['cell', 'cell_medium'])
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'train'), suffix='_weights_s128_2class_z256', classes=['cell', 'cell_medium'], coord_file='train_coords.txt')  # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'valid'), suffix='_weights_s128_2class_z128', classes=['cell', 'cell_medium'])
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'train'), suffix='_weights_s128_2class_z128', classes=['cell', 'cell_medium'], coord_file='train_coords.txt')  # check fit of data

    # 2 class augmented with fp and 2 scales
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'valid'), suffix='_weights_s128_2class_aug', classes=['cell', 'cell_medium'])
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'train'), suffix='_weights_s128_2class_aug', classes=['cell', 'cell_medium'], coord_file='train_coords_class_aug.txt')  # check fit of data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'valid'), suffix='_weights_s128_2class_aug_s64', classes=['cell', 'cell_medium'])
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'train'), suffix='_weights_s128_2class_aug_s64', classes=['cell', 'cell_medium'], coord_file='train_coords_class_aug.txt')  # check fit of data

    # # blurred plop generated data
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_blurred_32_32', 'valid'), suffix='_weights_s128')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_blurred_32_32', 'train'), suffix='_weights_s128', coord_file='training_coords.txt')  # check fit of data
    #
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_blurred_32_32', 'valid'), suffix='_weights_s64')
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_blurred_32_32', 'train'), suffix='_weights_s64', coord_file='training_coords.txt')  # check fit of data

    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_blurred_32_32', 'valid'), suffix='_weights_s64_class', classes=['cell', 'cell_medium'])
    # sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_avg_blurred_32_32', 'train'), suffix='_weights_s64_class', classes=['cell', 'cell_medium'], coord_file='train_coords_class.txt')  # check fit of data

    # 1-scan insitu
    sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_insitu_nickHypo', 'valid'), suffix='_weights_s128')
    sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_insitu_nickHypo', 'train'), suffix='_weights_s128', coord_file='train_coords.txt')  # check fit of data

    sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_insitu_nickHypo', 'valid'), suffix='_weights_s64')
    sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_insitu_nickHypo', 'train'), suffix='_weights_s64', coord_file='train_coords.txt')  # check fit of data

    sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_insitu_nickHypo', 'valid'), suffix='_weights_s64_class', classes=['cell', 'cell_medium'])
    sensitivity_analysis(results_folder=os.path.join('accell', 'ac_training_insitu_nickHypo', 'train'), suffix='_weights_s64_class', classes=['cell', 'cell_medium'], coord_file='train_coords_class.txt')  # check fit of data


    ### review segmentation and understand differences
    # accell_imgs = get_file_names('./accell/segmentations')
    # acseg_images = get_file_names('./acseg/segmentations')
    # acseg_empty_images = get_file_names('./acseg/empty_segmentations')

    # compare resolutions and img intensity moment profiles
    # compare_img_conversions()
    # thought experiment on predicting whole image vs parts of image
    # pred_experiment()
    # review_preds()
    1
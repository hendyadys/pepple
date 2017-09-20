from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import load_model
from keras.optimizers import Adam
# from keras import backend as K

# import sys
# import argparse
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt

# from train import get_unet, dice_coef, K
from train import get_unet
from data import load_valid_data, load_train_data, load_sliced_data, slice_data

# # horizontal slice transform
results_folder = './runs/run2/weights'
results_figs = './runs/run2/figs'
results_base = './runs/run2'
# # results_base = '/home/yue/pepple/runs/2017-08-07-17-15-40'
# # results_folder = '/home/yue/pepple/runs/2017-08-07-17-15-40/weights'
# # results_figs = '/home/yue/pepple/runs/2017-08-07-17-15-40/figs'
test_weights = 'weights-improvement-049--0.88935424.hdf5'
#
# vertical slice transform
results_folder = './runs/runVertical/weights'
results_figs = './runs/runVertical/figs'
results_base = './runs/runVertical'
results_base = '/home/yue/pepple/runs/2017-08-09-10-20-24'
results_folder = '/home/yue/pepple/runs/2017-08-09-10-20-24/weights'
results_figs = '/home/yue/pepple/runs/2017-08-09-10-20-24/figs'
test_weights = 'weights-improvement-050--0.95407502.hdf5'

# # vertical slice transform longer version with lr=1e-7
# results_folder = './runs/runVertical2/weights'
# results_figs = './runs/runVertical2/figs'
# results_base = './runs/runVertical2'
# results_base = '/home/yue/pepple/runs/2017-08-09-19-02-27'
# results_folder = '/home/yue/pepple/runs/2017-08-09-19-02-27/weights'
# results_figs = '/home/yue/pepple/runs/2017-08-09-19-02-27/figs'
# test_weights = 'weights-improvement-378--0.91209759.hdf5'   # still improving when crashed

img_rows = 512
img_cols = 512
# cropped_rows = 128
# cropped_cols = 496
cropped_rows = 496
cropped_cols = 128
pixel_overlap = 8
num_aug=96
val_start = 75


def load_params():
    params = []
    with open("%s/params.txt" % results_base) as fin:
        for l in fin:
            arr = l.rstrip().split("\t")
            params.append(np.float32(arr[1]))
    return params


# center and scale data appropriately
def center_scale_imgs(imgs, masks):
    params = load_params()
    imgs = imgs - params[0]
    imgs = imgs/params[1]
    masks = masks/255.
    return imgs, masks, params


def load_validation_data():
    imgs_valid, imgs_mask_valid = load_sliced_data()
    imgs_valid, imgs_mask_valid, params = center_scale_imgs(imgs_valid, imgs_mask_valid)
    return imgs_valid, imgs_mask_valid, params


# def analyse():
#     plt.switch_backend('agg')
#     model = get_unet()
#     imgs_valid, imgs_mask_valid, params = load_validation_data()
#
#     pnames = sorted(list(glob.glob('%s/weights*.hdf5' % results_folder)))
#     print(results_folder, len(pnames))
#
#     img_num = len(imgs_valid) - 1
#     img_num = 9140  # somewhat more intuitive segment
#     # plt.imsave('%s/orignal_input.png' % results_figs, imgs_valid[img_num,:,:,0])
#     cur_valid = imgs_valid[img_num, :, :, 0]
#     plt.imshow(cur_valid)
#     plt.title('orignal input')
#     plt.savefig('%s/orignal_input.png' % results_figs, bbox_inches='tight')
#
#     cur_mask = imgs_mask_valid[img_num, :, :, 0]
#     plt.imshow(cur_mask)
#     plt.title('actual target')
#     plt.savefig('%s/target.png' % results_figs, bbox_inches='tight')
#
#     for pname in pnames:
#         # p_toks = pname.split('-')
#         p_toks = pname.split('--')
#         epoch_num = p_toks[0].split('-')[-1]
#         val_dice = float(p_toks[1].split('.hdf5')[0])
#         print('processing %s; epoch:%s; val_dice=%f' % (pname, epoch_num, val_dice))
#
#         model.load_weights(pname)
#         output = model.predict(cur_valid.reshape(1, cropped_rows, cropped_cols, 1), batch_size=1)  # inference step
#         plt.imshow(output[0,:,:,0])
#         plt.title(pname)
#         plt.savefig('%s/%s.png' % (results_figs, epoch_num), bbox_inches='tight')
#         print('after model predict:', output.shape)
#
#     return 1


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
    plt.ylabel('dice loss', fontsize=12)
    plt.show()
    return


# rotation test on strips
def rotation_test():
    model = get_unet()
    imgs_valid, imgs_mask_valid, params = load_validation_data()
    valid_shape = imgs_valid.shape
    # mask_shape = imgs_mask.shape
    val_img_start = 75 * 96
    imgs_valid = imgs_valid[val_img_start:,]
    imgs_mask_valid = imgs_mask_valid[val_img_start:,]

    # weight_file = '%s/%s' % (results_folder, test_weights)  # use 50th epoch for sanity
    # model.load_weights(weight_file)
    # from train import dice_coef_loss, dice_coef
    # model.compile(optimizer=Adam(lr=params[2]), loss=dice_coef_loss, metrics=[dice_coef])

    # visualize imgs
    img_num = 9140 - val_img_start  # somewhat more intuitive segment
    # plt.figure(1)
    # plt.imshow(imgs_valid[img_num, :, :, 0])
    # plt.title('original')

    # rotate up to 1-20 degrees
    from scipy.ndimage.interpolation import rotate
    val_dice_losses = {}
    for deg in range(-20, 20):
        print(deg)
        new_valid = rotate(imgs_valid, deg, axes=(2, 1))
        new_mask = rotate(imgs_mask_valid, deg, axes=(2, 1))

        # take middle of rotated images
        new_valid_shape = new_valid.shape   # always bigger to fit rotation
        rem1 = (new_valid_shape[1] - valid_shape[1]) % 2
        row_start = int((new_valid_shape[1]-valid_shape[1])/2)
        row_end = int(new_valid_shape[1]-row_start - rem1)
        rem2 = (new_valid_shape[2] - valid_shape[2]) % 2
        col_start = int((new_valid_shape[2] - valid_shape[2]) / 2)
        col_end = int(new_valid_shape[2] - col_start - rem2)
        new_valid2 = new_valid[:, row_start:row_end, col_start:col_end, :]
        new_mask2 = new_mask[:, row_start:row_end, col_start:col_end, :]

        # # visualize differences
        # plt.figure(2)
        # plt.imshow(new_valid[img_num,:,:,0])
        # plt.title('rotated %d' % deg)

        val_dice = model.evaluate(new_valid2, new_mask2,  batch_size=10, verbose=1)  # inference step
        val_dice_losses[deg] = val_dice
        print(val_dice)

    # write to file
    with open("%s/rotation_test.txt" % results_base, "w") as fout:
        for deg, val_dice in val_dice_losses.items():
            fout.write("rotation=%d degrees; val_dice=%f\n" % (deg, val_dice[-1]))

    return val_dice_losses


# translation test on strips
def translation_test(direction=0, extend=0):  # 0 for left, 1 for right, 2 for both
    imgs_valid, imgs_mask_valid, params = load_validation_data()
    val_img_start = 75 * 96
    imgs_valid = imgs_valid[val_img_start:, ]
    imgs_mask_valid = imgs_mask_valid[val_img_start:, ]
    valid_shape = imgs_valid.shape
    mask_shape = imgs_mask_valid.shape

    model = get_unet()
    weight_file = '%s/%s' % (results_folder, test_weights)  # use 50th epoch for sanity
    model.load_weights(weight_file)
    from train import dice_coef_loss, dice_coef
    model.compile(optimizer=Adam(lr=params[2]), loss=dice_coef_loss, metrics=[dice_coef])

    # rotate up to 1-20 degrees
    val_dice_losses = {}
    for move_amount in range(0, 21):
        new_valid2 = np.zeros(valid_shape) - params[0]  # center for data
        new_mask2 = np.zeros(mask_shape)
        if extend:  # this will extend imgs_valid rather than just shift
            new_valid2 = imgs_valid
            new_mask2 = imgs_mask_valid
        if direction==0:    # shift horizontally
            new_valid2[:,move_amount:,] = imgs_valid[:,:valid_shape[1]-move_amount,]
            new_mask2[:,move_amount:,] = imgs_mask_valid[:,:valid_shape[1]-move_amount,]
        elif direction==1:  # shift vertically
            new_valid2[:,:,move_amount:,] = imgs_valid[:,:,:valid_shape[2]-move_amount,]
            new_mask2[:,:,move_amount:,] = imgs_mask_valid[:,:,:valid_shape[2]-move_amount,]
        elif direction==2:  # shift both
            new_valid2[:,move_amount:,] = imgs_valid[:,:valid_shape[1]-move_amount,]
            new_mask2[:,move_amount:,] = imgs_mask_valid[:,:valid_shape[1]-move_amount,]
            new_valid2[:,:,move_amount:,] = imgs_valid[:,:,:valid_shape[2]-move_amount,]
            new_mask2[:,:,move_amount:,] = imgs_mask_valid[:,:,:valid_shape[2]-move_amount,]

        val_dice = model.evaluate(new_valid2, new_mask2, batch_size=10, verbose=1)  # inference step
        val_dice_losses[move_amount] = val_dice
        print(val_dice)

    # write to file
    file_name = 'translation_d%d_e%d.txt' % (direction, extend)
    with open("%s/%s" % (results_base, file_name), "w") as fout:
        for move_amount, val_dice in val_dice_losses.items():
            fout.write("move=%d; val_dice=%f\n" % (move_amount, val_dice[-1]))

    return val_dice_losses


# should just do this server side and bring back images
def compute_image_preds(img_num):
    model = get_unet()
    imgs_valid, imgs_mask_valid, params = load_validation_data()

    pnames = sorted(list(glob.glob('%s/weights*.hdf5' % results_folder)))
    print(results_folder, len(pnames))
    num_epochs = len(pnames)
    img_start = img_num*num_aug

    img_preds = np.zeros((num_epochs, num_aug, cropped_rows, cropped_cols, 1), dtype=np.float32)
    for idx, pname in enumerate(pnames):
        p_toks = pname.split('--')
        epoch_num = p_toks[0].split('-')[-1]
        val_dice = float(p_toks[1].split('.hdf5')[0])
        print('processing %s; epoch:%s; val_dice=%f' % (pname, epoch_num, val_dice))

        model.load_weights(pname)
        output = model.predict(imgs_valid[img_start:img_start+num_aug, ], batch_size=1)  # inference step
        img_preds[idx, :] = output

    np.save('%s/predicted_img_%d.npy' % (results_base, img_num), img_preds)
    return 1


def make_movie_images(img_num):
    plt.switch_backend('agg')
    # plot original image and mask
    imgs_train, imgs_mask_train = load_train_data()
    plt.imshow(imgs_train[img_num, :, :, 0])
    plt.title('orignal input')
    plt.savefig('%s/orignal_input.png' % results_figs, bbox_inches='tight')

    plt.imshow(imgs_mask_train [img_num, :, :, 0])
    plt.title('actual target')
    plt.savefig('%s/target.png' % results_figs, bbox_inches='tight')

    npy_path = '%s/predicted_img_%d.npy' % (results_base, img_num)
    img_preds = np.load(npy_path)
    pred_shape = img_preds.shape
    num_epochs = pred_shape[0]
    num_epochs = 200
    num_aug = pred_shape[1]

    aligned_img = np.ndarray((num_epochs, num_aug, img_rows, img_cols), dtype=np.float32)
    num_cols = int((img_cols-cropped_cols)/pixel_overlap)
    for t in range(0, num_epochs):
        for i in range(0, num_aug):
            row_start = i // num_cols * pixel_overlap
            col_start = i % num_cols * pixel_overlap
            aligned_img[t, i, row_start:row_start+cropped_rows, col_start:col_start+cropped_cols] = img_preds[t, i, :, :, 0]
    avg_preds = np.mean(aligned_img, axis=1)

    for idx, avg_pred in enumerate(avg_preds):
        plt.imshow(avg_pred)
        plt.title('epoch %d' % (idx+1))   # make 1-index
        plt.savefig('%s/avg_pred_i%d_e%d.png' % (results_figs, img_num, idx+1), bbox_inches='tight')
    return


def plot_translation_results():
    for i in range(0, 3):
        # for j in range(0, 2):
            # file_base = 'translation_d%d_e%d' % (i, j)
        for j in range(0, 1):
            file_base = 'translation2_d%d' % (i)
            file_name = '%s.txt' % file_base
            move_amt = []
            val_dice = []
            with open('%s/%s' % (results_base, file_name)) as fin:
                for l in fin:
                    arr = l.rstrip().split("=")
                    move_amt.append(int(arr[1].split(";")[0]))
                    val_dice.append(float(arr[-1]))
            plt.scatter(move_amt, val_dice)
            trans_direction = 'vertical'
            if i==0:
                trans_direction = 'vertical'    # shift in row indices is vertical
            elif i==1:
                trans_direction = 'horizontal'
            elif i==2:
                trans_direction = 'diagonal'
            plt.title("%s translation" % (trans_direction))
            plt.xlabel('translation amount', fontsize=12)
            plt.ylabel('validation dice', fontsize=12)
            plt.savefig('%s.png' % file_base, bbox_inches='tight')
            plt.close()
    return


def rotation_test2():
    model = get_unet()
    imgs_strips, imgs_mask_strips, params = load_validation_data()
    imgs_whole, imgs_mask_whole = load_train_data()
    imgs_whole, imgs_mask_whole, params = center_scale_imgs(imgs_whole, imgs_mask_whole)    # normalize

    # subset validation images
    val_img_start = 75
    imgs_whole = imgs_whole[val_img_start:,]
    imgs_mask_whole = imgs_mask_whole[val_img_start:,]
    whole_shape = imgs_whole.shape

    # use fixed weights file
    weight_file = '%s/%s' % (results_folder, test_weights)  # use 50th epoch for sanity
    model.load_weights(weight_file)
    from train import dice_coef_loss, dice_coef
    model.compile(optimizer=Adam(lr=params[2]), loss=dice_coef_loss, metrics=[dice_coef])

    # rotate up to 1-20 degrees
    val_dice_losses = {}
    from scipy.ndimage.interpolation import rotate
    for deg in range(-20, 20):
        imgs_rotated, imgs_rotated_big = rotate_img(imgs_whole, deg)
        masks_rotated, masks_rotated_big = rotate_img(imgs_mask_whole, deg)

        # # visualize rotated image
        # img_num = 10
        # plt.figure(1)
        # plt.imshow(imgs_whole[img_num,:,:,0])
        # plt.title('original')
        # plt.figure(2)
        # plt.imshow(imgs_rotated_big[img_num,:,:,0])
        # plt.title('rotated')
        # plt.figure(3)
        # plt.imshow(imgs_rotated[img_num,:,:,0])
        # plt.title('rotated middle bit')
        # plt.figure(4)
        # plt.imshow(masks_rotated_big[img_num,:,:,0])
        # plt.title('rotated mask')
        # plt.figure(5)
        # plt.imshow(masks_rotated[img_num,:,:,0])
        # plt.title('rotated mask middle bit')

        # slice up middle bit of rotated
        sliced_data, sliced_mask = slice_data(data=imgs_rotated, mask=masks_rotated, save_data=False)
        val_dice = model.evaluate(sliced_data, sliced_mask,  batch_size=10, verbose=1)  # inference step
        val_dice_losses[deg] = val_dice

    # write to file
    with open("%s/rotation_test2.txt" % results_base, "w") as fout:
        for deg, val_dice in val_dice_losses.items():
            fout.write("rotation=%d degrees; val_dice=%f\n" % (deg, val_dice[-1]))

    return val_dice_losses


def translation_test2(direction=0, max_displacement=21):  # 0 for left, 1 for right, 2 for bothdef rotation_test2():
    model = get_unet()
    imgs_strips, imgs_mask_strips, params = load_validation_data()
    imgs_whole, imgs_mask_whole = load_train_data()
    imgs_whole, imgs_mask_whole, params = center_scale_imgs(imgs_whole, imgs_mask_whole)    # normalize
    whole_shape = imgs_whole.shape

    # subset validation images
    val_img_start = 75
    imgs_whole = imgs_whole[val_img_start:,]
    imgs_mask_whole = imgs_mask_whole[val_img_start:,]
    whole_shape = imgs_whole.shape

    # use fixed weights file
    weight_file = '%s/%s' % (results_folder, test_weights)  # use 50th epoch for sanity
    model.load_weights(weight_file)
    from train import dice_coef_loss, dice_coef
    model.compile(optimizer=Adam(lr=params[2]), loss=dice_coef_loss, metrics=[dice_coef])

    # rotate up to 1-20 degrees
    val_dice_losses = {}
    for move_amount in range(0, max_displacement):
        translated_img = translate_img(imgs_whole, direction, move_amount, init_val=params[0])
        translated_mask = translate_img(imgs_mask_whole, direction, move_amount, init_val=params[0])

        # # visualize rotated image
        # img_num = 10
        # plt.figure(1)
        # plt.imshow(imgs_whole[img_num,:,:,0])
        # plt.title('original')
        # plt.figure(2)
        # plt.imshow(translated_img[img_num,:,:,0])
        # plt.title('translated')

        # slice up middle bit of rotated
        sliced_data, sliced_mask = slice_data(data=translated_img, mask=translated_mask, save_data=False)
        val_dice = model.evaluate(sliced_data, sliced_mask,  batch_size=10, verbose=1)  # inference step
        val_dice_losses[move_amount] = val_dice

    # write to file
    file_name = 'translation2_d%d.txt' % (direction)
    with open("%s/%s" % (results_base, file_name), "w") as fout:
        for move_amount, val_dice in val_dice_losses.items():
            fout.write("move=%d; val_dice=%f\n" % (move_amount, val_dice[-1]))
    return val_dice_losses


def rotate_img(imgs, deg):
    from scipy.ndimage.interpolation import rotate
    imgs_shape = imgs.shape   # should be [n, row, col, 1]
    imgs_rotated = rotate(imgs, deg, axes=(2, 1))
    print(deg, imgs_shape , imgs_rotated.shape)

    # take middle bit
    rotated_shape = imgs_rotated.shape  # always bigger to fit rotation
    rem1 = (rotated_shape[1] - imgs_shape [1]) % 2
    row_start = int((rotated_shape[1] - imgs_shape [1]) / 2)
    row_end = int(rotated_shape[1] - row_start - rem1)
    rem2 = (rotated_shape[2] - imgs_shape [2]) % 2
    col_start = int((rotated_shape[2] - imgs_shape [2]) / 2)
    col_end = int(rotated_shape[2] - col_start - rem2)
    imgs_rotated_resized = imgs_rotated[:, row_start:row_end, col_start:col_end, :]
    return imgs_rotated_resized, imgs_rotated


def translate_img(imgs, direction, move_amount, init_val=0):
    imgs_shape = imgs.shape   # should be [n, row, col, 1]
    imgs_translated = np.zeros(imgs_shape) - init_val  # center for data
    if direction == 0:  # shift horizontally
        imgs_translated[:, move_amount:, ] = imgs[:, :imgs_shape[1] - move_amount, ]
    elif direction == 1:  # shift vertically
        imgs_translated[:, :, move_amount:, ] = imgs[:, :, :imgs_shape[2] - move_amount, ]
    elif direction == 2:  # shift both
        imgs_translated[:, move_amount:, ] = imgs[:, :imgs_shape[1] - move_amount, ]
        imgs_translated[:, :, move_amount:, ] = imgs[:, :, :imgs_shape[2] - move_amount, ]
    return imgs_translated


# transform=0 (nothing); -1 (rotation); 2 (vertical translation)
def make_movie_images_new(img_num, transform=0, transform_amount=0):
    plt.switch_backend('agg')
    img_start = img_num*num_aug

    imgs_valid, imgs_mask_valid, params = load_validation_data()
    transform_action = ''
    transform_qualifier = ''
    if transform:
        imgs_orig, imgs_mask_orig = load_train_data()
        imgs_orig, imgs_mask_orig, params = center_scale_imgs(imgs_orig, imgs_mask_orig)  # normalize

        img_shape = imgs_orig[img_num,].shape
        cur_img = imgs_orig[img_num,].reshape([1] + list(img_shape))
        cur_mask = imgs_mask_orig[img_num,].reshape([1] + list(img_shape))
        if transform==-1:    # rotation
            transform_action='rotation'
            transform_qualifier = 'by %d degrees' % transform_amount
            img_data, img_data_big = rotate_img(cur_img, transform_amount)
            mask_data, mask_data_big = rotate_img(cur_mask, transform_amount)
        else:
            transform_action='translation'
            transform_qualifier = 'by %d pixels' % transform_amount
            img_data = translate_img(cur_img, direction=transform-1, move_amount=transform_amount, init_val=params[0])
            mask_data = translate_img(cur_mask, direction=transform-1, move_amount=transform_amount, init_val=params[0])

        img_data, mask_data = slice_data(img_data, mask_data, save_data=False)
    else:
        img_data = imgs_valid[img_start:img_start+num_aug,]
        mask_data = imgs_mask_valid[img_start:img_start+num_aug,]

    pnames = sorted(list(glob.glob('%s/weights*.hdf5' % results_folder)))
    print(results_folder, len(pnames))
    num_epochs = len(pnames)

    model = get_unet()
    for idx, pname in enumerate(pnames):
        p_toks = pname.split('--')
        epoch_num = p_toks[0].split('-')[-1]
        val_dice = float(p_toks[1].split('.hdf5')[0])
        print('processing %s; epoch:%s; val_dice=%f' % (pname, epoch_num, val_dice))

        model.load_weights(pname)
        output = model.predict(img_data, batch_size=1)  # inference step
        # output = np.zeros(img_data.shape, dtype=np.float32)

        # generate images and save
        avg_preds = combine_img(output, real_mean=True)
        plt.imshow(avg_preds)
        # transform_action = 'rotation'

        plt.title('%s %s; epoch %d' % (transform_action, transform_qualifier, idx + 1))  # make 1-index
        plt.savefig('%s/recombined_pred_i%d_e%d_%s_m%d.png' %
                    (results_figs, img_num, idx + 1, transform_action, transform_amount), bbox_inches='tight')
    return 1


def combine_img(strip_preds, real_mean=False):
    aligned_img = np.ndarray((num_aug, img_rows, img_cols), dtype=np.float32)
    num_cols = int((img_cols - cropped_cols) / pixel_overlap)
    for i in range(0, num_aug):
        row_start = i // num_cols * pixel_overlap
        col_start = i % num_cols * pixel_overlap
        aligned_img[i, row_start:row_start + cropped_rows, col_start:col_start + cropped_cols] = strip_preds[i, :, :, 0]

    if real_mean:
        pred_sum = aligned_img.sum(0)
        pred_nonzero = (aligned_img != 0).sum(0).astype(float)
        avg_preds = np.true_divide(pred_sum, pred_nonzero)
        avg_preds[pred_nonzero == 0] = 0
    else:
        avg_preds = np.mean(aligned_img, axis=1)
    return avg_preds


# whole image dice computation stuff
def combine_predicted_strips_into_image(img_preds, img_num, debug_mode=False):
    img_start = img_num*num_aug
    combined_img_pred = combine_img(img_preds[img_start:img_start+num_aug, ], real_mean=True)

    if debug_mode:
        plt.imsave('%s/recon_%d.png' % (results_figs, img_num), combined_img_pred)
    return combined_img_pred


def dice_coef_np(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def calculate_whole_img_dice(model, weights, imgs_whole, masks_whole, imgs_strips, masks_strips, debug_mode=False):
    imgs_shape = imgs_whole.shape

    # load and predict
    model.load_weights(weights)
    output = model.predict(imgs_strips, batch_size=10)  # inference step
    if debug_mode:
        np.save('%s/all_preds.npy' % (results_base), output)
    print('output shape:', output.shape)  # should be (28*96)*496*128*1

    # combine img for val_dice
    predicted_imgs = np.ndarray(imgs_shape, dtype=np.float32)   # for tensor dice calculation
    for j in range(0, imgs_shape[0]):  # for each image
        predicted_imgs[j, :, :, 0] = combine_predicted_strips_into_image(output, j, debug_mode=debug_mode)

    # compute dice on original image - padding
    masks_strips = masks_strips.astype(np.float32)
    masks_whole = masks_whole.astype(np.float32)
    # print(imgs_mask_valid.dtype, output.dtype, imgs_mask_orig.dtype, predicted_val_imgs.dtype)
    strip_dice = dice_coef_np(masks_strips, output)
    whole_dice = dice_coef_np(masks_whole, predicted_imgs)
    whole_dice_no_padding = dice_coef_np(masks_whole[:, :, :500, ], predicted_imgs[:, :, :500, :])
    return predicted_imgs, strip_dice, whole_dice, whole_dice_no_padding


def evaluate_whole_img_dice(debug_mode=False):
    plt.switch_backend('agg')

    model = get_unet()
    imgs_valid, imgs_mask_valid, params = load_validation_data()
    imgs_orig, imgs_mask_orig = load_train_data()
    imgs_orig, imgs_mask_orig, params = center_scale_imgs(imgs_orig, imgs_mask_orig)  # normalize

    pnames = sorted(list(glob.glob('%s/weights*.hdf5' % results_folder)))
    # pnames = ['%s/%s' % (results_folder, test_weights)]
    num_epochs = len(pnames)
    print(results_folder, num_epochs)

    # only look at validation data
    imgs_orig = imgs_orig[val_start:,]
    imgs_mask_orig = imgs_mask_orig[val_start:,]

    img_start = val_start*num_aug
    imgs_valid = imgs_valid[img_start:,]
    imgs_mask_valid = imgs_mask_valid[img_start:,]

    combined_dice_file = './combined_dice.txt'
    for idx, pname in enumerate(pnames):    # loop over epoch weights
        p_toks = pname.split('--')
        epoch_num = p_toks[0].split('-')[-1]
        val_dice = float(p_toks[1].split('.hdf5')[0])
        print('processing %s; epoch:%s; val_dice=%f' % (pname, epoch_num, val_dice))

        predicted_imgs, strip_dice, whole_dice, whole_dice_no_padding = \
            calculate_whole_img_dice(model, pname, imgs_orig, imgs_mask_orig, imgs_valid, imgs_mask_valid, debug_mode=debug_mode)

        if debug_mode:
            np.save('%s/predicted_imgs.npy' % (results_base), predicted_imgs)

        with open(combined_dice_file, 'a') as fout:
            # fout.write('%d\t%f\t%f\t%f\n' % (int(epoch_num), K.eval(strip_dice), K.eval(whole_dice), K.eval(whole_dice_no_padding)) )
            fout.write('%d\t%f\t%f\t%f\n' % (int(epoch_num), strip_dice, whole_dice, whole_dice_no_padding))
    return


def debug_whole_image_dice():
    # truth
    imgs_valid, imgs_mask_valid, params = load_validation_data()
    imgs_train, imgs_mask_train = load_train_data()
    imgs_train, imgs_mask_train, params = center_scale_imgs(imgs_train, imgs_mask_train )  # normalize

    imgs_train = imgs_train.astype(np.float32)
    imgs_mask_train = imgs_mask_train.astype(np.float32)

    all_preds = np.load('%s/all_preds.npy' % (results_base))
    predicted_imgs = np.load('%s/predicted_imgs.npy' % (results_base))

    img_num = 10
    plt.figure(1)
    plt.imshow(imgs_train[img_num+val_start,:,:,0])
    plt.title('original image')

    plt.figure(2)
    plt.imshow(imgs_mask_train[img_num+val_start,:,:,0])
    plt.title('original mask')

    plt.figure(3)
    plt.imshow(predicted_imgs[img_num,:,:,0])
    plt.title('combined predicted mask')

    plt.figure(4)
    recon_img = combine_predicted_strips_into_image(all_preds, img_num)
    plt.imshow(recon_img)

    predicted_imgs2 = np.ndarray((imgs_train.shape[0]-val_start, img_rows, img_cols, 1), dtype=np.float32)  # for tensor dice calculation
    for i in range(0,imgs_train.shape[0]-val_start):
        recon_img = combine_predicted_strips_into_image(all_preds, i)
        predicted_imgs2[i,:,:,0] = recon_img
    whole_dice = dice_coef_np(imgs_train[val_start:,], predicted_imgs2)
    whole_dice_no_padding = dice_coef_np(imgs_train[val_start:, :, :500, ], predicted_imgs2[:, :, :500, :])
    return


def predict_problem_img(file_name):
    import cv2, os
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # img = img.reshape((img_rows, 500, 1))
    img = img.reshape((1024, 1000, 1))
    zero_pad = np.zeros((1024, 24, 1), dtype=np.uint8)
    img = np.concatenate((img, zero_pad), axis=1)   # HACK - add padding at bottom of image

    # center and scale data appropriately
    params = load_params()
    img = img - params[0]
    img = img/params[1]

    # subsample for convenience and slice_data to work
    img = img[1::2, 1::2,]
    img = img.reshape((1, img_rows, img_cols, 1))
    # slice up
    sliced_img, sliced_mask = slice_data(img, img)
    # predict
    model = get_unet()
    weight_file = '%s/%s' % (results_folder, test_weights)  # use 50th epoch for sanity
    model.load_weights(weight_file)
    output = model.predict(sliced_img, batch_size=10)  # inference step
    combined_pred = combine_img(output, real_mean=True)

    # visualize
    # plt.switch_backend('agg')
    plt.figure(1)
    plt.imshow(img[0,:,:,0])
    plt.title('original test image')
    plt.figure(2)
    plt.imshow(combined_pred)
    plt.title('predicted mask')
    return


if __name__ == '__main__':
    # plot_loss('valid.txt')
    # plot_loss('train.txt')
    # analyse()

    # # translation_test
    # translation_test(0, extend=0)
    # translation_test(1, extend=0)
    # translation_test(2, extend=0)
    # # translation_test(0, extend=1) # these are smears - not useful
    # # translation_test(1, extend=1)
    # # translation_test(2, extend=1)
    # plot_translation_results()
    # translation_test2(0)
    # translation_test2(1)
    # translation_test2(2)
    # translation_test2(0,s max_displacement=496)

    rotation_test()
    rotation_test2()
    # #plot translation and rotation analysis
    # degs = []
    # deg_dice = []
    # with open("%s/rotation_test2.txt" % results_base) as fin:
    #     for l in fin:
    #         arr = l.rstrip().split("=")
    #         degs.append(int(arr[1].split()[0]))
    #         deg_dice.append(float(arr[-1]))
    # plt.scatter(degs, deg_dice)
    # plt.title('Rotation vs Validation Dice')
    # plt.grid(True)
    # plt.show()

    # plot_loss('combined_dice.txt')
    # evaluate_whole_img_dice()
    # evaluate_whole_img_dice(debug_mode=True)
    # debug_whole_image_dice()

    img_num = 100
    # compute_image_preds(img_num)
    # make_movie_images(img_num)
    make_movie_images_new(img_num, transform=-1, transform_amount=5)    # rotate 5 degrees
    make_movie_images_new(img_num, transform=-1, transform_amount=10)    # rotate 10 degrees
    make_movie_images_new(img_num, transform=-1, transform_amount=15)    # rotate 15 degrees
    make_movie_images_new(img_num, transform=-1, transform_amount=20)    # rotate 20 degrees

    make_movie_images_new(img_num, transform=1, transform_amount=20)    # vertical 20 pixels
    make_movie_images_new(img_num, transform=1, transform_amount=50)    # vertical 20 degrees
    make_movie_images_new(img_num, transform=1, transform_amount=100)  # vertical 100 degrees
    make_movie_images_new(img_num, transform=1, transform_amount=200)  # vertical 200 degrees
    make_movie_images_new(img_num, transform=1, transform_amount=300)  # vertical 300 degrees

    # # problematic images
    # predict_problem_img('./acseg/test/Inflamed_20170703mouse4_Day2_Right_591.png')
import os, random, cv2
import numpy as np
from matplotlib import pyplot as plt
from make_accell_data import prefix, img_folder, get_img_predictions, avg_images, get_coords, MULTI_CLASS, SCALE_FACTOR
INTENSITY_FACTOR=30


## helper functions
def pts2indices(pts):
    x, y = [], []
    for pt in pts:
        x.append(pt[0])
        y.append(pt[1])
    return (np.array(x), np.array(y))


def plot_img(img, fig_num=1):
    plt.figure(fig_num)
    plt.clf()
    plt.imshow(img)
    return


def plot_contours(img, contours, fig_num=2):
    img2 = np.zeros(img.shape)
    cv2.drawContours(img2, contours, -1, (255, 255, 255), 3)  # all contours on img

    plt.figure(fig_num)
    plt.clf()
    plt.imshow(img2)

    # cv2.drawContours(img, contours, 3, (255, 255, 255), 3)
    # cnt = contours[4]
    # cv2.drawContours(img, [cnt], 0, (255, 255, 255), 3)
    return


def plot_edges(edges, fig_num=2):
    plt.figure(fig_num)
    plt.clf()
    plt.imshow(edges, cmap='Greys')
    return


def find_obj_edges(mask, thresh1=0, visualise=False):
    mask = mask.astype(np.uint8)
    edges = cv2.Canny(mask, thresh1, 255)     # 0-255 for edge on mask
    # contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if visualise:
        plot_img(mask)
        plot_edges(edges)
    return edges


def find_obj_contours(mask, thresh1=127):
    ret, thresh = cv2.threshold(mask, thresh1, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_areas = [cv2.contourArea(x) for x in contours]
    return contours, cnt_areas


# create masks from labelled centroid
def create_accell_masks(folder=img_folder, do_avg=False, visualise=False):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder)  # raw_images->converted(scale 50%)->predicted
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)
    num_imgs = raw_images.shape[0]

    if do_avg:
        recentered_json_folder = os.path.join('accell', 'jsons_recentered')
    else:
        # recentered_json_folder = os.path.join('accell', 'jsons_recentered_1scan')
        recentered_json_folder = os.path.join('accell', 'jsons_recentered_1scan_3by3')

    cell_dict = {}
    for idx, img_name in enumerate(img_names):
        cur_coords = get_coords(os.path.join(recentered_json_folder, img_name.replace('.png', '.json')))
        cell_dict[img_name] = cur_coords

    # output folders
    out_folder_name = 'ac_seg_masks'
    base_folder = os.path.join(prefix, 'accell', format(out_folder_name))
    mask_folder = os.path.join(base_folder, 'mask_v1')
    mask_folder2 = os.path.join(base_folder, 'mask_v2')
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        os.makedirs(mask_folder)
        os.makedirs(mask_folder2)

    for idx, img_name in enumerate(img_names):
        print('processing {}/{}; {}'.format(str(idx), str(len(img_names)), img_name))
        img = raw_images[idx]
        mask1, mask2 = make_seg_mask(img_name, img, cell_dict)
        mask_path1 = os.path.join(mask_folder, '{}_mask.png'.format(img_name.replace('.png', '')))
        cv2.imwrite(mask_path1, mask1*INTENSITY_FACTOR)
        img_overlaid = overlay_edge_on_img(img, mask1)
        overlaid_path1 = os.path.join(mask_folder, '{}_overlaid.png'.format(img_name.replace('.png', '')))
        cv2.imwrite(overlaid_path1, img_overlaid)
        # plot_img(img_overlaid, fig_num=10)
        mask_path2 = os.path.join(mask_folder2, '{}_mask.png'.format(img_name.replace('.png', '')))
        cv2.imwrite(mask_path2, mask2*INTENSITY_FACTOR)
        img_overlaid2 = overlay_edge_on_img(img, mask2)
        overlaid_path2 = os.path.join(mask_folder2, '{}_overlaid.png'.format(img_name.replace('.png', '')))
        cv2.imwrite(overlaid_path2, img_overlaid2)
        # plot_img(img_overlaid2, fig_num=10)
    return


def overlay_edge_on_img(img, mask, visualise=False):    # visualise mask
    img_overlaid= img.copy()
    edge = find_obj_edges(mask*127)
    img_overlaid[edge!=0] = 127

    if visualise:   # stack to see better
        img_overlaid2 = np.repeat(np.expand_dims(img_overlaid, axis=2), 3, axis=2)
        plot_img(img_overlaid2)
        # cv2.imwrite('baskd.png', img_overlaid2)
    return img_overlaid


def make_seg_mask(img_name, img, cell_dict, visualise=False):
    cells = cell_dict[img_name]
    mask_all_1 = np.zeros(img.shape)
    mask_all_2 = np.zeros(img.shape)
    for cell_center in cells:
        mask1, mask2, mask_pts1, mask_pts2 = make_cell_mask(img, cell_center, cls_val=1)
        mask_all_1 = np.maximum(mask_all_1, mask1)
        mask_all_2 = np.maximum(mask_all_2, mask2)

    if visualise:
        plot_img(img, fig_num=1)
        for cell in cells:
            x, y = cell
            plt.scatter(x=x, y=y, color='red')
        plot_img(mask_all_1, fig_num=2)
        plot_img(mask_all_2, fig_num=3)
    return mask_all_1, mask_all_2


def make_cell_mask(img, cell_center, cls_val=1):
    mask_pts1, mask1 = expand_recurse(img, cell_center, cls_val, ratio_thresh=2)
    mask_pts2, mask2 = expand_recurse_pointwise(img, cell_center, cls_val, ratio_thresh=2)
    return mask1, mask2, mask_pts1, mask_pts2


def expand_recurse_pointwise(img, cell_center, cls_val, ratio_thresh=2, visualise=False):
    nrows, ncols = img.shape
    mask = np.zeros((nrows, ncols))

    x, y = cell_center
    center_val = img[y, x]  # NB - x (cols), y (rows)

    # ratio_thresh for big gradient/change in edge intensities
    n_by_m = make_n_by_m(x,y, n=1, m=1)  # start with 3*3;
    cur_candidates = n_by_m.copy()
    all_candidates = n_by_m.copy()
    mask_pts = [cell_center]  # start with 1*1;
    counter =0
    while len(cur_candidates) > 0:
        # print(counter)
        counter+=1

        cur_pt = cur_candidates[0]
        cur_candidates.pop(0)
        if cur_pt in mask_pts:  # already processed
            continue
        cur_x, cur_y = cur_pt
        pt_val = img[cur_y, cur_x]  # NB - x (cols), y (rows)
        if center_val/pt_val < ratio_thresh:
            mask_pts.append(cur_pt)
            new_candidates = make_n_by_m(cur_x, cur_y, n=1, m=1)
            for new_c in new_candidates:
                euclid_dist = np.linalg.norm(np.array(cell_center)-np.array(new_c))
                if euclid_dist < 3 and new_c not in all_candidates:     # not prev checked
                    all_candidates.append(new_c)
                    cur_candidates.append(new_c)
                    # cur_candidates = np.unique(cur_candidates, axis=0)
                    # cur_candidates = list(cur_candidates)
            # print(1)

    mask_pts0 = np.array(mask_pts)
    mask_pts2 = np.zeros(mask_pts0.shape, dtype=np.int)  # flip mask_pts back to x,y format as in cell_center
    mask_pts2[:, 0] = mask_pts0[:, 1]
    mask_pts2[:, 1] = mask_pts0[:, 0]
    mask[pts2indices(list(mask_pts2))] = cls_val

    if visualise:
        plot_img(img, fig_num=1)
        plt.scatter(x=x, y=y, color='red')
        plot_img(mask, fig_num=2)
    return mask_pts, mask


def expand_recurse(img, cell_center, cls_val, ratio_thresh=2, visualise=False):
    nrows, ncols = img.shape
    mask = np.zeros((nrows, ncols))

    x, y = cell_center
    center_val = img[y, x]  # NB - x (cols), y (rows)
    n_by_m = make_n_by_m(y, x, n=1, m=1)  # start with 3*3;
    mask[pts2indices(n_by_m)] = cls_val  # expand outwards

    # ratio_thresh for big gradient/change in edge intensities
    mask_pts = n_by_m   # start with 3*3;
    directions = ['up', 'down', 'left', 'right']
    while len(directions)>0:
        cur_dir = directions[0]
        cur_mask_pts = np.argwhere(mask!=0)     # N*2
        new_pts, intensity_ratio = get_expanded_pts(img, cur_mask_pts, cur_dir)
        new_vals = img[pts2indices(new_pts)]
        if max(np.linalg.norm(np.array(new_pts)-np.array([y,x]), axis=1)) <3 and \
                                center_val/np.mean(new_vals)<ratio_thresh:    # relative to center brightness
            directions.append(cur_dir)  # keep expanding in dir
            mask_pts += new_pts
        popped_element = directions.pop(0)
        mask[pts2indices(mask_pts)] = cls_val

    mask_pts = np.array(mask_pts)
    mask_pts2 = np.zeros(mask_pts.shape)    # flip mask_pts back to x,y format as in cell_center
    mask_pts2[:,0] = mask_pts[:,1]
    mask_pts2[:,1] = mask_pts[:,0]
    if visualise:
        plot_img(img, fig_num=1)
        plt.scatter(x=x, y=y, color='red')
        plot_img(mask, fig_num=2)

    return list(mask_pts2), mask


def get_expanded_pts(img, cur_pts, dir):
    nrows, ncols = img.shape
    y_idx = 0   # maps to rows
    x_idx = 1   # maps to cols
    if dir in ['up', 'down']:
        if dir =='up':
            cur_y = min(cur_pts[:, y_idx])
            new_y = max(0, cur_y - 1)
            y_range = range(new_y, new_y+1)
        elif dir=='down':
            cur_y = max(cur_pts[:, y_idx])
            new_y = min(nrows, cur_y + 1)
            y_range = range(new_y, new_y+1)
        edge_pts = cur_pts[cur_pts[:, y_idx] == cur_y, ]
        min_x = min(edge_pts[:, x_idx])
        max_x = max(edge_pts[:, x_idx])
        x_range = range(min_x, max_x + 1)  # inclusive
        # if dir=='up':
        #     x_range = range(max(0, min_x-1), max_x +1)   # new_top_left_corner_x -> max_x
        # elif dir=='down':
        #     x_range = range(min_x, min(ncols, max_x+1 +1))  # min_x -> new_bot_right_corner_x
    else:
        if dir == 'left':
            cur_x = min(cur_pts[:, x_idx])
            new_x = max(0, cur_x - 1)
            x_range = range(new_x, new_x+1)
        elif dir == 'right':
            cur_x = max(cur_pts[:, x_idx])
            new_x = min(ncols, cur_x + 1)
            x_range = range(new_x, new_x+1)
        edge_pts = cur_pts[cur_pts[:, 1] == cur_x, ]
        min_y = min(edge_pts[:, y_idx])
        max_y = max(edge_pts[:, y_idx])
        y_range = range(min_y, max_y + 1)  # inclusive
        # if dir=='left':
        #     y_range = range(min_y, min(nrows, max_y+1 +1))   # new_bot_left_corner_y -> max_y
        # elif dir=='right':
        #     y_range = range(max(0, min_y-1), max_y +1)  # new_top_right_corner_y -> max_y

    new_pts = [(y, x) for x in x_range for y in y_range]
    new_vals = img[pts2indices(new_pts)]
    cur_vals = img[pts2indices(edge_pts)]
    intensity_ratio = np.mean(cur_vals)/np.mean(new_vals)
    return new_pts, intensity_ratio


def make_n_by_m(x, y, n=1, m=1):
    n_by_m = []
    for x_prime in range(x-n, x+n+1):   # inclusive
        for y_prime in range(y-n, y+m+1):   # inclusive
            n_by_m.append((x_prime, y_prime))
    return n_by_m


def check_cell_stats(f1='mask_v1', f2='mask_v2'):
    base_folder = os.path.join(prefix, 'accell', 'ac_seg_masks')
    f1 = os.path.join(base_folder, f1)
    f2 = os.path.join(base_folder, f2)
    img_names = os.listdir(f1)
    mask_names = [x for x in sorted(img_names) if 'mask' in x]

    contourDict = {}
    for idx, mask_name in enumerate(mask_names):
        path_f1 = os.path.join(f1, mask_name)
        mask1 = cv2.imread(path_f1, cv2.IMREAD_GRAYSCALE)
        path_f2 = os.path.join(f2, mask_name)
        mask2 = cv2.imread(path_f2, cv2.IMREAD_GRAYSCALE)
        # visualise
        plot_img(mask1, fig_num=1)
        plot_img(mask2, fig_num=2)

        path_overlaid_f1 = os.path.join(f1, mask_name.replace('mask', 'overlaid'))
        overlaid1 = cv2.imread(path_overlaid_f1)
        path_overlaid_f2 = os.path.join(f2, mask_name.replace('mask', 'overlaid'))
        overlaid2 = cv2.imread(path_overlaid_f2)
        # visualise
        plot_img(overlaid1, fig_num=3)
        plot_img(overlaid2, fig_num=4)

        # count contours and contourAreas
        cnt1, cnt_areas1 = find_obj_contours(mask1, 0)
        cnt2, cnt_areas2 = find_obj_contours(mask2, 0)
        # record
        contourDict[mask_name] = {'edgeMask':cnt_areas1, 'ptMask':cnt_areas2}
    return


## create patched datasets for PSPNet
def make_train_valid_data(mask_folder, folder=img_folder, patch_rows=512, patch_cols=512, num_samples=5, visualise=0):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder)  # raw_images->converted(scale 50%)->predicted

    # split training and validation
    kathryn_leslie_img_names = [x for x in sorted(img_names) if 'Kathryn' in x or 'Leslie' in x]
    num_kathryn_leslie = len(kathryn_leslie_img_names)
    train_split = 0.85
    train_end = int(np.floor(num_kathryn_leslie*train_split))
    train_names = kathryn_leslie_img_names[:train_end]
    valid_names = kathryn_leslie_img_names[train_end:]

    train_images, valid_images = [], []
    train_pred_images, valid_pred_images = [], []
    for img_name in train_names:
        train_idx = img_names.index(img_name)
        train_images.append(raw_images[train_idx,])
        train_pred_images.append(img_preds[train_idx,])

    for img_name in valid_names:
        valid_idx = img_names.index(img_name)
        valid_images.append(raw_images[valid_idx, ])
        valid_pred_images.append(img_preds[valid_idx,])

    train_folder = os.path.join(prefix, 'accell', 'ac_seg_masks', 'train')
    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)
    valid_folder = os.path.join(prefix, 'accell', 'ac_seg_masks', 'valid')
    if not os.path.isdir(valid_folder):
        os.makedirs(valid_folder)

    sample_patches(train_names, train_images, train_pred_images, mask_folder, train_folder, patch_rows=patch_rows,
                   patch_cols=patch_cols, num_samples=num_samples, visualise=visualise)
    sample_patches(valid_names, valid_images, valid_pred_images, mask_folder, valid_folder, patch_rows=patch_rows,
                   patch_cols=patch_cols, num_samples=num_samples, visualise=visualise)
    return


def sample_patches(img_names, images, img_preds, mask_folder, out_folder, patch_rows=512, patch_cols=512, num_samples=5, visualise=0):
    for idx, img_name in enumerate(img_names):
        print('processing {}/{}; {}'.format(idx, len(img_names), img_name))

        img = images[idx]
        nrows, ncols = img.shape
        mask_path = os.path.join(mask_folder, img_name.replace('.png', '_mask.png'))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        pad_size =50
        img_c, mask_c, (y1, y2), (x1, x2) = get_img_center(img, mask, img_preds[idx], pad_size=pad_size)
        img_rows, img_cols = img_c.shape
        for jdx in range(num_samples):
            if (x2-x1+1)<patch_cols or (y2-y1+1)<patch_rows:    # AC too small
                if (x2 - x1 + 1) < patch_cols:
                    diff_col = patch_cols-(x2-x1+1) + pad_size
                    x1 = max(0, x1 - int(diff_col/2))
                    x2 = min(ncols, x2 + int(diff_col/2))
                if (y2-y1+1)<patch_rows:
                    diff_row = patch_rows - (y2 - y1+ 1) + pad_size
                    y1 = max(0, y1 - int(diff_row/2))
                    y2 = min(nrows, y2 + int(diff_row/2))
                img_c = img[y1:y2+1, x1:x2+1]
                mask_c = mask[y1:y2+1, x1:x2+1]
                img_rows, img_cols = img_c.shape
            y_sample_range = range(0, img_rows - patch_rows)
            x_sample_range = range(0, img_cols - patch_cols)
            if len(y_sample_range)==0 or len(x_sample_range)==0:
                print(idx, img_name, jdx, len(y_sample_range), len(x_sample_range))

            sampled_y = random.sample(y_sample_range, 1)[0]
            sampled_x = random.sample(x_sample_range, 1)[0]
            sampled_img = img_c[sampled_y:sampled_y + patch_rows, sampled_x:sampled_x + patch_cols]
            sampled_mask = mask_c[sampled_y:sampled_y + patch_rows, sampled_x:sampled_x + patch_cols]
            if visualise:
                plot_img(img_c, fig_num=10)
                plot_img(mask_c, fig_num=11)
                plot_img(sampled_img, fig_num=1)
                plot_img(sampled_mask, fig_num=2)

            save_name = '{}_r{}_c{}.png'.format(img_name.replace('.png', ''), y1+sampled_y, x1+sampled_x)
            mask_name = save_name.replace('.png', '_mask.png')
            sampled_img_path = os.path.join(out_folder, save_name)
            cv2.imwrite(sampled_img_path, sampled_img)
            sampled_mask_path = os.path.join(out_folder, mask_name)
            cv2.imwrite(sampled_mask_path, sampled_mask/INTENSITY_FACTOR)
    return


def get_img_center(img, mask, img_preds, pad_size=50, visualise=False):
    nrows, ncols = img.shape

    row_maxes = np.max(img_preds, axis=1)
    chamber_rows = np.where(row_maxes>.2)
    first_row = int(max(chamber_rows[0][0]/SCALE_FACTOR - pad_size, 0))
    last_row = int(min(chamber_rows[0][-1]/SCALE_FACTOR + pad_size, nrows-1))
    col_maxes = np.max(img_preds, axis=0)
    chamber_cols = np.where(col_maxes>.2)
    first_col = int(max(chamber_cols[0][0]/SCALE_FACTOR - pad_size, 0))
    last_col = int(min(chamber_cols[0][-1]/SCALE_FACTOR + pad_size, ncols-1))
    img_c = img[first_row:last_row+1, first_col:last_col+1]     # include last row and col
    mask_c = mask[first_row:last_row+1, first_col:last_col+1]    # include last row and col

    if visualise:
        plot_img(img_preds, fig_num=2)
        plot_img(img, fig_num=1)
        plt.axvline(first_col, color='red')
        plt.axvline(last_col, color='red')
        plt.axhline(first_row, color='lime')
        plt.axhline(last_row, color='lime')
        plot_img(img_c, fig_num=3)
        plot_img(mask_c, fig_num=4)
    return img_c, mask_c, (first_row, last_row), (first_col, last_col)


def fit_object_ellipse(img, cnt, visualise=False):
    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    ellipse = [(x, y), (MA, ma), angle]

    if visualise:
        img2 = img.copy()
        thickness =5
        cv2.ellipse(img2, (int(x),int(y)), (int(MA), int(ma)), angle, 0, 360, (255, 255, 255), thickness=thickness)  # draw ellipse
        plot_img(img2, fig_num=3)
    return ellipse


if __name__ == '__main__':
    # create_accell_masks(folder=img_folder, visualise=False, do_avg=False)

    # ## cell stats to check 2 masks
    # check_cell_stats()
    # NB - pointwise mask seems to leave unclosed singletons in contours

    ## make train/valid data for PSPnet
    mask_folder= os.path.join(prefix, 'accell', 'ac_seg_masks', 'mask_v1')  # edge recursive masks; v2 point-recursive
    make_train_valid_data(mask_folder=mask_folder, folder=img_folder)

    # ## demographics of dataset
    # train_folder = os.path.join(prefix, 'accell', 'ac_seg_masks', 'train')
    # train_masks = [x for x in os.listdir(train_folder) if 'mask' in x]
    # total_px, mask_px = 0, 0
    # for idx, mask_name in enumerate(train_masks):
    #     mask_path = os.path.join(train_folder, mask_name)
    #     mask= cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #     cur_mask_px = np.sum(mask!=0)
    #     mask_px += cur_mask_px
    #     total_px += np.prod(mask.shape)
    # print(total_px, mask_px, mask_px/total_px)  # 170131456 89531 0.0005262460106142864
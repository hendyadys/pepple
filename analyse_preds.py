import numpy as np
import random, os, subprocess, cv2, json
from sys import platform
import time

from matplotlib import pyplot as plt
from matplotlib import patches

# from data import slice_data
# from analyser import combine_img, center_scale_imgs
# from make_accell_data import convert_raw_imgs, convert_image, get_img_predictions, DOWNSAMPLE_RATIO, get_raw_imgs, find_chamber_center
# from analyseCellPreds import recombine_predictions, visualise_pred_vs_truth

patch_rows = 32
patch_cols = 32
prob = [0, 5, 10, 25, 50, 75, 90, 95, 100]
prob = [0, 10, 50, 90, 100]


def parse_pred_file(folder, conservative=False, full_vol=False, no_mid=0, ac_threshold=.9, old_nomen=True):
    fname = '{}_preds'.format(folder.split('\\')[-1])
    if old_nomen:
        if conservative:
            fname = '{}{}'.format(fname, '_conservative')
        if full_vol:
            fname = '{}{}'.format(fname, '_full_vol')
        fname = '{}.csv'.format(fname)
    else:
        fname = '{}_c{}_m{}_t{}.csv'.format(fname, conservative, no_mid, ac_threshold)

    summary_file = os.path.join(folder, fname)
    if not os.path.isfile(summary_file):
        print('summary file not found:', summary_file)
        return [], []

    img_names = []
    img_stats = []
    with open(summary_file, 'r') as fin:
        # lines are not ordered
        counter = 0
        for l in fin.readlines():
            if counter==0:  # ignore header
                counter+=1
                continue

            l_toks = l.rstrip().split(',')
            img_name = l_toks[0]
            seg_area = int(l_toks[1])
            num_cell = int(l_toks[2])
            area_cell = int(l_toks[3])
            num_med = int(l_toks[4])
            area_med = int(l_toks[5])
            img_names.append(img_name)
            img_stats.append([seg_area, num_cell, area_cell, num_med, area_med])
    fin.close()
    return img_names, np.asarray(img_stats)


# plot ac areas vs img_names
def get_lim_val_index(img, do_min=False):
    # cur_coord = img.argmax(axis=0)
    img_shape = img.shape

    if do_min:
        val_index = img.argmin()
    else:
        val_index = img.argmax()

    if len(img_shape) > 1:
        i, j = np.unravel_index(val_index, img.shape)   # for higher d
        lim_val = img[i, j]
    else:
        i, j = np.unravel_index(val_index, [len(img), 1])
        lim_val = img[i]
    return lim_val, i, j


def get_pred_stats(folder, conservative=0, full_vol=False, no_mid=1, ac_threshold=.95, old_nomen=True):
    # visualise from outputted imgs for speed and less data loading
    if old_nomen:
        pred_figs = os.path.join(folder, 'pred_figs') if not conservative else os.path.join(folder, 'pred_figs_conservative')
    else:
        pred_figs = os.path.join(folder, 'pred_figs_c{}_m{}_t{}'.format(conservative, no_mid, ac_threshold))

    img_names, img_stats = parse_pred_file(folder, conservative=conservative, full_vol=full_vol, no_mid=no_mid,
                                           ac_threshold=ac_threshold, old_nomen=old_nomen)
    if len(img_stats)==0:   # missing file
        return None, None

    # # visualise min and max ac seg
    # max_ac_area, max_ac_index_i, max_ac_index_j = get_lim_val_index(img_stats[:, 0], do_min=False)
    # visualise_img(pred_figs, img_names[max_ac_index_i], title='max ac area')
    #
    # min_ac_area, min_ac_index_i, min_ac_index_j = get_lim_val_index(img_stats[:, 0], do_min=True)
    # visualise_img(pred_figs, img_names[min_ac_index_i], title='min ac area')

    # # ac cells vs cell areas
    # plt.figure(3)
    # plt.scatter(img_stats[:, 1], img_stats[:, 2])
    # plt.title('num_cells vs area_cells')
    # plt.figure(4)
    # plt.scatter(img_stats[:, 3], img_stats[:, 4])
    # plt.title('num_cell_medium vs area_cell_medium')

    # find high and low values
    # find corresponding image and review
    max_acs, max_ac_index_i, max_ac_index_j = get_lim_val_index(img_stats[:, 1], do_min=False)
    # visualise_img(pred_figs, img_names[max_ac_index_i], title='max ac count')
    min_acs, min_ac_index_i, min_ac_index_j = get_lim_val_index(img_stats[:, 1], do_min=True)
    # visualise_img(pred_figs, img_names[min_ac_index_i], title='min ac count')

    max_acs_all, max_ac_all_index_i, max_ac_all_index_j = get_lim_val_index(img_stats[:, 1]+img_stats[:, 3], do_min=False)
    # visualise_img(pred_figs, img_names[max_ac_all_index_i], title='max ac count')
    min_acs_all, min_ac_all_index_i, min_ac_all_index_j = get_lim_val_index(img_stats[:, 1]+img_stats[:, 3], do_min=True)
    # visualise_img(pred_figs, img_names[min_ac_all_index_i], title='min ac count')

    # now look at distribution of cells
    cell_perc = np.percentile(img_stats[:, 1], prob)  # by cell
    cell_all_perc = np.percentile(img_stats[:, 1]+img_stats[:, 3], prob)  # by cell+cell_med
    # cell_stats = np.append(cell_perc, cell_all_perc)     # min, med, max for cells and cell_med
    cell_stats = np.append(cell_perc, cell_all_perc)
    cell_info = [img_names[min_ac_index_i], img_names[img_stats[:, 1].tolist().index(cell_perc[1])], img_names[max_ac_index_i],
                 img_names[min_ac_all_index_i], img_names[(img_stats[:, 1]+img_stats[:, 3]).tolist().index(cell_all_perc[1])], img_names[max_ac_all_index_i]]     # min, med, max for cells and cell_med

    if cell_perc[0]!=min_acs or cell_perc[-1]!=max_acs:
        print('check ac count for ', folder)
    if cell_all_perc[0]!=min_acs_all or cell_all_perc[-1]!=max_acs_all:
        print('check all ac count for ', folder)

    return cell_stats, cell_info


def visualise_img(img_folder, img_name, title=''):
    plt.figure()
    img_path = os.path.join(img_folder, '{}.png'.format(img_name))
    img = cv2.imread(img_path)
    plt.imshow(img)
    if title:
        plt.title('{} - {}'.format(title, img_name))
    return


def get_pepple_classes(file_base='peppleScores'):
    right_dict = get_pepple_classes_helper(file_base, is_left=False)
    left_dict = get_pepple_classes_helper(file_base, is_left=True)
    right_dict.update(left_dict)    # merges in place
    return right_dict


def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_pepple_classes_helper(file_base='peppleScores', is_left=False):
    file = '{}_{}.csv'.format(file_base, 'Left' if is_left else 'Right')
    file_class_dict = {}
    with open(file, 'r') as fin:
        for idx, l in enumerate(fin.readlines()):
            if idx==0: continue     # skip header   - Animal/ Eye,Sac Date OCT Score,R Day-7,R Day1,R Day2
            l_toks = l.rstrip().split(',')
            fname = l_toks[0]
            score_dminus7 = l_toks[2].replace('*', '')
            score_d1 = l_toks[3].replace('*', '')
            score_d2 = l_toks[-1].replace('*', '')

            # file_class_dict[fname] = {'fname': fname, 'score_d1':score_d1, 'score_d2':score_d2, 'score_dminus7':score_dminus7}
            fname_d1 = '{}_Day1_{}'.format(fname, 'Left' if is_left else 'Right')
            fname_d2 = '{}_Day2_{}'.format(fname, 'Left' if is_left else 'Right')
            fname_dminus7 = '{}_Day-7_{}'.format(fname, 'Left' if is_left else 'Right')
            file_class_dict[fname_d1] = float(score_d1) if isFloat(score_d1) else -1
            file_class_dict[fname_d2] = float(score_d2) if isFloat(score_d2) else -1
            file_class_dict[fname_dminus7] = float(score_dminus7) if isFloat(score_dminus7) else -1

    fin.close()
    return file_class_dict


# scale is not linear
def class2count(pepple_class):
    if pepple_class==0:
        count = 0
    elif pepple_class==0.5:
        count = 5/2
    elif pepple_class==1:
        count = (5+15)/2
    elif pepple_class==2:
        count = (16+24)/2
    elif pepple_class==3:
        count = 35
    elif pepple_class==4:   # hypopion - this might get segmented out
        count = 50
    else:   # shouldn't happen
        print('invalid class:', pepple_class)
        count = 0
    return count


def count2class(cell_count):
    if cell_count==0:
        pepple_class = 0
    elif cell_count < 5:
        pepple_class = 0.5
    elif cell_count < 16:
        pepple_class = 1
    elif cell_count < 25:
        pepple_class = 2
    elif cell_count < 51:
        pepple_class = 3
    elif cell_count > 50:   # hypopion - this might get segmented out
        pepple_class = 4
    return pepple_class


def visualise_count_vs_scores(stats, comp_col, folder_info, title='', ignore_unknowns=True, add_jitter=True):
    if ignore_unknowns:  # -1
        stats = stats[stats[:, 0] > -1, ]

    num_vols, num_stats = stats[:, 1:].shape
    no_4s = len(stats[stats[:, 0] < 4])
    if add_jitter:
        # stats[:, 1:] = stats[:, 1:] + (np.random.rand(num_vols, num_stats) - 0.5)/10   # jitter cell counts
        stats[:, 1:] += (np.random.rand(num_vols, num_stats) - 0.5)/5   # jitter cell counts

    plt.figure()
    plt.clf()
    plt.scatter(stats[:, 0], stats[:, comp_col])     # cell median
    plt.title(title)
    plt.xlabel('expert scores')
    plt.ylabel('cell counts {}'.format('with jitter' if add_jitter else ''))

    # kappa = agree/total
    agreement_strict = 0
    agreement_1 = 0
    agreement_strict_no4s = 0
    agreement_1_no4s = 0
    for idx in range(len(all_folder_stats[:, 2])):
        pepple_class_equiv = count2class(all_folder_stats[idx, comp_col])
        pepple_class = all_folder_stats[idx, 0]

        if pepple_class_equiv==pepple_class:
            agreement_strict += 1
            agreement_1 += 1
        elif abs(pepple_class_equiv-pepple_class)<=1:
            agreement_1 += 1
        if pepple_class!=4:
            if pepple_class_equiv == pepple_class:
                agreement_strict_no4s += 1
                agreement_1_no4s += 1
            elif abs(pepple_class_equiv - pepple_class) <= 1:
                agreement_1_no4s += 1

        if abs(pepple_class_equiv-all_folder_stats[idx, 0]) > 1:  # only label things way off (2 classes)
            label = folder_info[idx][comp_col-1]  # since this is list of lists
            label = '{}_m{}_vs_p{}'.format(label, pepple_class_equiv, pepple_class)
            plt.annotate(label, xy=(all_folder_stats[idx, 0], all_folder_stats[idx, comp_col]), xytext=(-20, 20),
                         textcoords='offset points', ha='right', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    kappa = agreement_strict/float(num_vols)
    kappa1 = agreement_1/float(num_vols)
    print('comp_col={}; kappa={}/{}={}; kappa1={}'.format(comp_col, agreement_strict, num_vols, kappa, kappa1))
    kappa_no4s = agreement_strict_no4s/float(no_4s)
    kappa1_no4s = agreement_1_no4s/float(no_4s)
    print('comp_col={}; kappa_no4s={}/{}={}; kappa1__no4s={}'.format(comp_col, agreement_strict_no4s, no_4s, kappa_no4s, kappa1_no4s))
    return


def get_longitudinal_folders(base_folder):
    long_folders = {}

    folders = [x for x in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, x)) and 'right' in x.lower()]  # only want infected mice
    for idx, folder in enumerate(folders):
        # f_base = folder.replace('Right', '').rstrip()
        # if f_base[-1]=='_':
        #     f_base = f_base[:-1]

        f_base = folder.split('_')[0]
        f_d1 = '{}_Day1_Right'.format(f_base)
        f_d2 = '{}_Day2_Right'.format(f_base)
        f_dminus = '{}_Day-7_Right'.format(f_base)
        if f_d1 in folders and f_d2 in folders and f_dminus in folders: # 3 day points for mouse
            if f_base in long_folders:
                long_folders[f_base].append(folder)
            else:
                long_folders[f_base] = [folder]
    return long_folders


def get_longitudinal_cells(base_folder):
    long_base_dict = get_longitudinal_folders(base_folder)
    mice_exp_names = long_base_dict.keys()
    num_mice = len(mice_exp_names)
    long_stats = np.zeros(shape=(num_mice, 3, 10))
    all_cell_info = []
    for jdx, f_base in enumerate(mice_exp_names):
        folders = long_base_dict[f_base]
        ordered_folders = sorted(folders)   # Day-7, Day1, Day2
        temp_info = []
        for idx, folder in enumerate(ordered_folders):
            cell_stats, cell_info = get_pred_stats(os.path.join(base_folder, folder), conservative=5, full_vol=False,
                                                   no_mid=1, ac_threshold=.95, old_nomen=False)
            long_stats[jdx, idx, ] = cell_stats
            temp_info.append(cell_info)
        all_cell_info.append(temp_info)

    # visualise - add a little jitter
    jittered_stats = long_stats + (np.random.rand(num_mice, 3, len(prob)*2) - 0.5) / 5
    d_periods =[-7, 1, 2]
    if len(prob)==3:
        comp_index = 2  # 100% (max)
    elif len(prob)==5:
        comp_index = 3  # 90%
    plt.figure(1)
    plt.clf()
    plt.plot(np.transpose(np.tile(d_periods, [num_mice, 1])), np.transpose(jittered_stats[:, :, comp_index]))
    # plt.figure(2)
    # plt.clf()
    # # plt.scatter(np.tile(d_periods, [num_mice, 1]), jittered_stats[:, :, comp_index])
    # # add labels
    # plt.ylabel('number of cells')
    # plt.xlabel('days since injection')
    # for jdx, f_base in enumerate(mice_exp_names):
    #     plt.scatter(x=np.asarray(d_periods), y=jittered_stats[jdx, :, 2])
    #     folders = long_base_dict[f_base]
    #     ordered_folders = sorted(folders)   # Day-7, Day1, Day2
    #     # label = f_base
    #     for idx, folder in enumerate(folders):
    #         label = all_cell_info[jdx][idx][comp_index]
    #         plt.annotate(label, xy=(d_periods[idx], jittered_stats[jdx, idx, comp_index]), xytext=(-20, 20),
    #                      textcoords='offset points', ha='right', va='bottom',
    #                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    return


if __name__ == '__main__':
    get_longitudinal_cells(os.path.join('volume_data', 'vol_to_analyse'))

    # folder = os.path.join('volume_data', '20170703mouse1_Day1_Right')  # 2
    # folder = os.path.join('volume_data', '20170703mouse2_Day1_Right')  # .5
    # folder = os.path.join('volume_data', '20170703mouse3_Day1_Right')  # 2
    # folder = os.path.join('volume_data', '20170703mouse4_Day1_Right')  # .5
    # folder = os.path.join('volume_data', '20170703mouse5_Day1_Right')  # 1
    # folder = os.path.join('volume_data', '20170703mouse6_Day1_Right')  # 4
    # folder = os.path.join('volume_data', '20170703mouse7_Day1_Right')  # 4
    # folder = os.path.join('volume_data', '20170703mouse8_Day1_Right')  # .5

    # pepple_label_dict = {'20170703mouse1_Day1_Right':2, '20170703mouse2_Day1_Right':.5, '20170703mouse3_Day1_Right':2,
    #                      '20170703mouse4_Day1_Right':.5, '20170703mouse5_Day1_Right':1,
    #                '20170703mouse6_Day1_Right':4, '20170703mouse7_Day1_Right':4, '20170703mouse8_Day1_Right':.5}
    pepple_label_dict = get_pepple_classes(file_base='peppleScores')
    all_folder_stats = []
    all_folder_info = []
    # folders = ['20170703mouse1_Day1_Right', '20170703mouse2_Day1_Right', '20170703mouse3_Day1_Right',
    #                '20170703mouse4_Day1_Right', '20170703mouse5_Day1_Right',
    #                '20170703mouse6_Day1_Right', '20170703mouse7_Day1_Right']
    folder_path = os.path.join('volume_data', 'vol_to_analyse')
    folders = [x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))]
    for folder in folders:
        # cell_stats = get_pred_stats(folder, conservative=True, full_vol=False)
        cell_stats, cell_info = get_pred_stats(os.path.join(folder_path, folder), conservative=5, full_vol=False, no_mid=1, ac_threshold=.95, old_nomen=False)
        if folder in pepple_label_dict:
            pepple_class = pepple_label_dict[folder]
        else:
            if 'Left' in folder:
                print('invalid key for left', folder)
                pepple_class = -1
            else:
                pepple_class = pepple_label_dict[folder.replace(' Right', '_Right')]

        if cell_stats is not None:
            all_folder_stats.append(np.append(pepple_class, cell_stats))
            all_folder_info.append(cell_info)
    # visualise
    all_folder_stats = np.asarray(all_folder_stats)

    # visualise_count_vs_scores(all_folder_stats, comp_col=2, folder_info=all_folder_info, title='class vs median(num_cells)')
    visualise_count_vs_scores(all_folder_stats, comp_col=3, folder_info=all_folder_info, title='class vs max(num_cells)')
    # visualise_count_vs_scores(all_folder_stats, comp_col=5, folder_info=all_folder_info, title='class vs median(all_cells)')
    visualise_count_vs_scores(all_folder_stats, comp_col=6, folder_info=all_folder_info, title='class vs max(all_cells)')
    print('done')
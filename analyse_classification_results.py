import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score

CLASSIFICATION_FOLDER = 'classification'


def parse_preds_file(file):
    preds = np.loadtxt(file, delimiter=',')
    return preds


def analyse_preds(region='AC', dset='test', is_weighted=0):
    # file = os.path.join(CLASSIFICATION_FOLDER, 'preds_{}_{}_w{}.csv'.format(region, dset, is_weighted))
    file = os.path.join(CLASSIFICATION_FOLDER, 'preds_{}_{}_w{}.csv'.format(region, dset, is_weighted))
    # file = os.path.join(CLASSIFICATION_FOLDER, 'figures_new_1024', 'preds_{}_{}_w{}.csv'.format(region, dset, is_weighted))
    # file = os.path.join(CLASSIFICATION_FOLDER, 'figures_new_512', 'preds_{}_{}_w{}.csv'.format(region, dset, is_weighted))
    preds = parse_preds_file(file)
    # print(preds)
    preds_t = np.argmax(preds[:, :-1], axis=1)
    targets = preds[:,-1]
    num_test = len(preds)
    num_correct = np.sum(preds_t==targets)
    acc = num_correct/num_test

    # class_labels = [0,1,2,3,4,5]
    # c = confusion_matrix(targets.astype(np.int), preds_t.astype(np.int), labels=class_labels)
    # cd = ConfusionMatrixDisplay(c, display_labels=class_labels)
    # cd.plot()
    # plt.show()

    # # relabel
    targets_c = targets.copy()
    targets_c[targets_c==5] = 0.5
    preds_tc = preds_t.astype(np.float).copy()
    preds_tc[preds_tc==5] = 0.5
    targets_c = targets_c.astype(str)
    preds_tc = preds_tc.astype(str)

    f = os.path.join('Z:/yue/pepple/classification/data/2021.07.07 New TIFFs/{}_imgs_r512_c512/'.format(region))
    f = os.path.join('Z:/yue/pepple/classification/data/2021.07.07 New TIFFs/{}_imgs_r512_c1024/'.format(region))
    f = os.path.join('Z:/yue/pepple/data_2024/{}_imgs_r512_c512/'.format(region))
    if dset=='valid':
        file = os.path.join(f, 'validation_img_labels.csv')
    elif dset=='test':
        file = os.path.join(f, 'test_img_labels.csv')
    labels = []
    with open(file, 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(',')
            labels.append(l_toks)
    fin.close()
    labels = np.array(labels)
    # sanity check
    print(len(targets_c), len(labels), np.sum(labels[:, 0].astype(np.float) == targets_c.astype(np.float)))

    # output preds and labels for pepple
    with open('labels_preds_{}_{}.csv'.format(region, dset), 'w') as fout:
        for idx, x in enumerate(labels):
            fname = x[-1].split('/')[-1]
            if fname[0]=='S':
                fname_toks = fname.split(' ')
                fname = ' '.join(fname_toks[1:])
            vals = [fname, x[0], preds_tc[idx]]
            fout.write('{}\n'.format(','.join(vals)))
    fout.close()

    # # output blank + images for pepple
    # out_folder = os.path.join(CLASSIFICATION_FOLDER, 'pred_label_diff_imgs')
    # if not os.path.isdir(out_folder):
    #     os.makedirs(out_folder)
    # import cv2
    # for idx, x in enumerate(labels):
    #     fpath = x[-1].replace('/data/yue/pepple', 'z:/yue/pepple')
    #     fname = x[-1].split('/')[-1]
    #     if fname[0]=='S':
    #         fname_toks = fname.split(' ')
    #         fname = ' '.join(fname_toks[1:])
    #     if float(x[0]) != float(preds_tc[idx]):
    #         img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    #         fpath_out = os.path.join(out_folder, fname)
    #         cv2.imwrite(fpath_out, img)

    k = cohen_kappa_score(preds_tc, targets_c)
    k_lin = cohen_kappa_score(preds_tc, targets_c, weights='linear')
    k_quad = cohen_kappa_score(preds_tc, targets_c, weights='quadratic')
    print(region, dset, is_weighted, acc, k, k_lin, k_quad)

    class_labels_c = list(np.array([0,0.5,1,2,3,4]).astype(str))
    c_c = confusion_matrix(targets_c, preds_tc, labels=class_labels_c)
    cd_c = ConfusionMatrixDisplay(c_c, display_labels=class_labels_c)
    cd_c.plot()
    # plt.show()
    plt.savefig(os.path.join(CLASSIFICATION_FOLDER, 'confusion_matrix_{}_{}_w{}.png'.format(region, dset, is_weighted)),
                bbox='tight')

    return


def analyze_human_agreement():
    from explore_labels_files_2024 import read_final_score_2025
    label_dict_final, bad_files = read_final_score_2025()
    ac_labels = []
    for img_key, score_data in label_dict_final["AC"].items():
        kc_score = parse_score(score_data["KC score"])
        kp_score = parse_score(score_data["KP score"])
        lw_score = parse_score(score_data["LW score"])
        xp_score = parse_score(score_data["XP score"])
        ac_labels.append([kc_score, kp_score, lw_score, xp_score])
    ac_labels_array = np.array(ac_labels).astype(float)

    k1_a, k_lin1_a, k_quad1_a = compute_kappa(ac_labels_array[:,1], ac_labels_array[:,2])   # pepple vs wilson
    k2_a, k_lin2_a, k_quad2_a = compute_kappa(ac_labels_array[:, 2], ac_labels_array[:, 0])  # wilson vs costello
    k3_a, k_lin3_a, k_quad3_a = compute_kappa(ac_labels_array[:, 2], ac_labels_array[:, 3])  # wilson vs xu

    pc_labels = []
    for img_key, score_data in label_dict_final["PC"].items():
        kc_score = parse_score(score_data["KC score"])
        kp_score = parse_score(score_data["KP score"])
        lw_score = parse_score(score_data["LW score"])
        xp_score = parse_score(score_data["XP score"])
        pc_labels.append([kc_score, kp_score, lw_score, xp_score])
    pc_labels_array = np.array(pc_labels).astype(float)

    k1_p, k_lin1_p, k_quad1_p = compute_kappa(pc_labels_array[:,1], pc_labels_array[:,2])   # pepple vs wilson
    k2_p, k_lin2_p, k_quad2_p = compute_kappa(pc_labels_array[:, 2], pc_labels_array[:, 0])  # wilson vs costello
    k3_p, k_lin3_p, k_quad3_p = compute_kappa(pc_labels_array[:, 2], pc_labels_array[:, 3])  # wilson vs xu
    return


def compute_kappa(arr1, arr2):
    arr1_indices = np.nonzero(~np.isnan(arr1))
    arr2_indices = np.nonzero(~np.isnan(arr2))
    common_indices = set(list(arr1_indices[0])).intersection(list(arr2_indices[0]))
    arr1_common = arr1[list(common_indices)]
    arr2_common = arr2[list(common_indices)]

    k = cohen_kappa_score(arr1_common, arr2_common)
    k_lin = cohen_kappa_score(arr1_common, arr2_common, weights='linear')
    k_quad = cohen_kappa_score(arr1_common, arr2_common, weights='quadratic')
    print(len(arr1_indices), len(arr2_indices), len(common_indices), k, k_lin, k_quad)
    return k, k_lin, k_quad


def parse_score(score_str):
    if score_str not in ["0", ".5", "1", "2", "3", "4"]:
        return float("nan")
    return float(score_str)


if __name__ == "__main__":
    # analyze_human_agreement()

    region = 'AC'
    for region in ['AC', 'PC']:
        for dset in ['valid', 'test']:
            # for is_weighted in [0, 1]:
            for is_weighted in [0]:
                analyse_preds(region, dset, is_weighted)
import os
import numpy as np
from analyseCellPreds import check_false_positives_helper, check_missed_helper, sensitivity_analysis, \
    get_predicted_coords, get_true_coords_by_class, get_true_coords_file


def write_fp_coords(results_folder, true_folder, true_file='valid_coords.txt', path_prefix='', suffix='', classes=['cell', 'cell_medium', 'cell_lite']):
    num_classes = len(classes)

    # true_dict_by_class = get_true_coords_by_class(true_folder, coord_file=true_file, path_prefix=path_prefix)
    # check against number of lines found for each class
    true_dict = get_true_coords_file(true_folder, coord_file=true_file, path_prefix=path_prefix)
    print('check total labelled cells:', np.sum([len(val) for val in true_dict.values()]))  #number of all labelled cells - compare against file length

    cell_dict = get_predicted_coords(results_folder, file_name='{}_{}{}.txt'.format('coords', classes[0], suffix))
    # combined_dict = cell_dict
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
    if num_classes > 1:
        med_keys = cell_med_dict.keys()
        all_keys = list(set(list(cell_keys) + list(med_keys) ))
    if num_classes>2:
        lite_keys = cell_lite_dict.keys()
        all_keys = list(set(list(cell_keys) + list(med_keys) + list(lite_keys)))

    for key in all_keys:
        # true_key = '{}.png'.format(key)
        true_key = key
        cell_vals = cell_dict[key] if key in cell_dict else []
        combined_dict[true_key] = cell_vals
        if num_classes > 1:
            med_vals = cell_med_dict[key] if key in cell_med_dict else []
            combined_dict[true_key] = cell_vals + med_vals
        if num_classes > 2:
            lite_vals = cell_lite_dict[key] if key in cell_lite_dict else []
            combined_dict[true_key] = cell_vals + med_vals + lite_vals

    false_positive_dict, matched_dict = check_false_positives_helper(combined_dict, true_dict, pixel_lim=3)
    print('total predicted=', np.sum([len(val) for val in combined_dict.values()]),
          'matched=', np.sum([len(val) for val in matched_dict.values()]),
          'false_positives=', np.sum([len(val) for val in false_positive_dict.values()]))
    num_false_positives = [len(cells) for cells in false_positive_dict.values()]
    num_preds = [len(cells) for cells in combined_dict.values()]
    print(suffix, 'all_preds', 'total_fp=', np.sum(num_false_positives), 'total_preds=', np.sum(num_preds), 'fpr=', np.sum(num_false_positives) / np.sum(num_preds),
          'precision=', 1 - np.sum(num_false_positives) / np.sum(num_preds))

    out_folder = results_folder
    with open(os.path.join(out_folder, 'fp{}.txt'.format(suffix.split('_')[-1].replace('s', ''))), 'w') as fout:
        for fname, fcells in false_positive_dict.items():
            for cell in fcells:
                vals = [fname] + [str(x) for x in cell[:-1]] + ['not_cell']
                fout.write('{}\n'.format(','.join(vals)))
    fout.close()
    return


if __name__ == '__main__':
    # # allow overlap
    # r_type = 'train'
    # sub_folder = 'ac_training_avg_insitu_32_32'
    # true_folder = os.path.join('accell', sub_folder, r_type)
    # true_file = '{}_coords.txt'.format(r_type)
    # results_folder = os.path.join('accell', sub_folder, r_type)
    # path_prefix = '/data/yue/pepple/accell/{}/{}/'.format(sub_folder, r_type)
    #
    # # suffix = '_weights_s128'
    # # write_fp_coords(results_folder, true_folder=true_folder, true_file=true_file, path_prefix=path_prefix, suffix=suffix)
    #
    # suffix = '_weights_s64'
    # write_fp_coords(results_folder, true_folder=true_folder, true_file=true_file, path_prefix=path_prefix, suffix=suffix)

    # allow overlap
    r_type = 'train'
    sub_folder = 'ac_training_avg_insituM_32_32'
    sub_folder = 'ac_training_avg_insitunickHypo_32_32'
    true_folder = os.path.join('accell', sub_folder)
    true_file = '{}_coords.txt'.format(r_type)
    results_folder = os.path.join('accell', sub_folder, r_type)
    path_prefix = '/data/yue/pepple/accell/{}/{}/'.format(sub_folder, r_type)
    # suffix = '_weights_s64'
    suffix = '_weights_s128'
    suffix = '_weights_s128_2class'
    write_fp_coords(results_folder, true_folder=true_folder, true_file=true_file, path_prefix=path_prefix, suffix=suffix, classes=['cell', 'cell_medium'])
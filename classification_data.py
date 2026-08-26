import os, json, cv2, glob, sys, random, math
import numpy as np
from sys import platform
from matplotlib import pyplot as plt

from sys import platform
if platform == "linux" or platform == "linux2":
    base_folder = '/data/yue/pepple'
    sep = '/'
elif platform == "win32":
    base_folder = 'z:/yue/pepple'
    # base_folder = '.'
    # sep = '\\'
img_folder = os.path.join(base_folder, 'classification', 'data')

DICT_KEY_SCORE = 'score'
DICT_KEY_LOC = 'img_loc'


def read_labels(label_file=os.path.join(img_folder, '2020.12.22 by Jessica UPDATED listing of all TIFF images (contains duplicates).csv')):
    # all_tiffs = glob.glob('{}/**/*.TIFF'.format(img_folder), recursive=True)
    label_json = os.path.join(img_folder, 'label_img_path.json')
    if os.path.isfile(label_json):
        fin = open(label_json).read()
        label_dict = json.loads(fin)
        return label_dict

    label_dict = {'AC':{}, 'PC':{}}
    with open(label_file, 'r') as fin:
        lines = fin.readlines()
        for idx, l in enumerate(lines):
            l_toks = l.rstrip().split(',')
            class_level, region, img_name, img_url = l_toks
            region = region.replace(' ', '')
            img_loc = glob.glob('{}/**/*{}'.format(img_folder, img_name), recursive=True)
            if len(img_loc)!=1:
                print(idx, ',', img_name, ',', img_loc)
            if len(img_loc)>0:
                if img_name not in label_dict[region]:
                    label_dict[region][img_name] = []
                label_dict[region][img_name].append({DICT_KEY_SCORE:class_level, DICT_KEY_LOC:img_loc[0]})
    fin.close()

    with open(label_json, 'w') as fout:
        json.dump(label_dict, fout)
    fout.close()
    return label_dict


def get_png_from_tiff(tiff_file, nrow=512, ncol=500):
    img = cv2.imread(tiff_file, cv2.IMREAD_GRAYSCALE)   # TIFFs are 1024*1000
    # img_scaled = cv2.resize(img, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
    img_scaled = cv2.resize(img[:1000,:1000], (ncol, nrow), interpolation=cv2.INTER_CUBIC)
    return img_scaled


def parse_img_name(img_name):
    if 'ouse' in img_name:
        m_index = img_name.index('ouse')
    elif 'ousse' in img_name:
        m_index = img_name.index('ousse')
    else:
        # print(img_name)
        m_index = img_name.index('m')
    substring = img_name[m_index:]
    sub_toks = substring.split('_')
    mouse_num, which_eye = sub_toks[0], sub_toks[1]
    return mouse_num, which_eye


def check_images(f=img_folder, eye_level=False):
    # label_file = os.path.join(f, '2020.12.22 by Jessica UPDATED listing of all TIFF images (contains duplicates).csv'),
    # label_dict = read_labels(label_file)
    for region in ['PC']:
        region_dict = read_labels_new(region=region)
        # compute_kappa(region_dict)

        mouse_demo = {}
        level_demo = {}  # mouse and level demographics
        for img_name, img_dict in region_dict.items():
            mouse_num, which_eye = parse_img_name(img_name)
            if eye_level:
                mouse_key = '{}_{}'.format(mouse_num, which_eye)
            else:
                mouse_key = mouse_num
            img_level = img_dict[DICT_KEY_SCORE]
            if mouse_key not in mouse_demo:
                mouse_demo[mouse_key] = []
            mouse_demo[mouse_key].append(img_level)

            if img_level not in level_demo:
                level_demo[img_level ] = []
            level_demo[img_level].append(mouse_key)

        print(region, 'level_demo:', [(x, len(x_vals)) for x, x_vals in level_demo.items()])
        print(region, 'mouse_demo:', [(x, len(x_vals)) for x, x_vals in mouse_demo.items()])
        with open(os.path.join(f, '{}_mouse_{}.json'.format(region, int(eye_level))), 'w') as fout:
            json.dumps(mouse_demo, fout)
        fout.close()
        with open(os.path.join(f, '{}_level_{}.json'.format(region, int(eye_level))), 'w') as fout:
            json.dumps(level_demo, fout)
        fout.close()

        # # output to csv for training/test split
        # for mouse_key in sorted(mouse_demo.keys()):
        #     x_vals = mouse_demo[mouse_key]
        #     for y in [0, 0.5, 1, 2, 3, 4]:
        #         print(region, mouse_key, y, np.sum(np.array(x_vals)==str(y)))

        train_test_split_file = os.path.join(f, 'train_test_split_{}.csv'.format(region))
        with open(train_test_split_file, 'w') as fout:
            fout.write('mouse, score \n')
            for mouse_key, mouse_data in mouse_demo.items():
                for d in mouse_data:
                    fout.write('{},{}\n'.format(mouse_key, d))
        fout.close()
    return


def make_training_test_split_data(f=img_folder, eye_level=False, nrow=512, ncol=500):
    split_dict = {"AC":
                      {"training":["ouse1", "ouse2", "ouse4", "ouse5", "ouse6", "ouse7", "ouse8", "ouse9", "ouse11",
                                   "ouse12", "ouse19", "ouse14", "ouse15", "ouse17", "ouse18", "ouse20", "ouse21",
                                   "ouse22", "ouse2F"],
                       "validation":["ouse16", "ouse10", "ousse10"],
                       "test":["ouse3", "ouse13"]},
                  'PC':{"training":["ouse1", "ouse2", "ouse4", "ouse5", "ouse6", "ouse7", "ouse8", "ouse9", "ouse11",
                                   "ouse12", "ouse19", "ouse14", "ouse15", "ouse17", "ouse18", "ouse20", "ouse21",
                                   "ouse22", "ouse2F"],
                       "validation":["ouse16", "ouse10", "ousse10"],
                       "test":["ouse3", "ouse13"]},
                  }
    for region in ['AC', 'PC']:
        # region_dict = read_labels_new(region=region)
        region_dict = read_labels_20210217(region=region, f=f)

        combined_img_folder = os.path.join(f, '{}_imgs_r{}_c{}'.format(region, nrow, ncol))
        if not os.path.isdir(combined_img_folder):
            os.makedirs(combined_img_folder)

        for img_name, img_dict in region_dict.items():
            mouse_num, which_eye = parse_img_name(img_name)
            mouse_key = mouse_num

            file_split = None
            if mouse_key in split_dict[region]['training']:
                file_split = 'training'
            elif mouse_key in split_dict[region]['validation']:
                file_split = 'validation'
            elif mouse_key in split_dict[region]['test']:
                file_split = 'test'
            else:
                file_split = 'training'
                # continue

            tiff_file = img_dict[DICT_KEY_LOC]
            img_score = img_dict[DICT_KEY_SCORE]
            png_path = os.path.join(combined_img_folder, '{}.png'.format(img_name.replace('.TIFF', '')))
            if nrow==ncol:
                img = get_png_from_tiff(tiff_file, nrow=nrow, ncol=ncol)
                cv2.imwrite(png_path, img)  # not cropped for 512*512
            else:
                img = get_png_from_tiff(tiff_file, nrow=1024, ncol=1024)
                ignore_top = 100 if region=="AC" else 80
                q = 60 if region=='PC' else 70
                img2 = crop_image(img, ignore_top=ignore_top, q=q)
                print(img_name, tiff_file, img.shape, img2.shape)
                cv2.imwrite(png_path, img2)

            split_file = os.path.join(combined_img_folder, '{}_img_labels.csv'.format(file_split))
            with open(split_file, 'a') as fout:
                fout.write('{},{}\n'.format(img_score, png_path))
            fout.close()
    return


def read_labels_new(region='AC', f=img_folder):
    label_json = os.path.join(img_folder, 'label_paths_{}.json'.format(region))
    if os.path.isfile(label_json):
        fin = open(label_json).read()
        label_dict = json.loads(fin)
        return label_dict

    label_file = os.path.join(f, 'labels_{}_latest.csv'.format(region))
    label_dict = {}
    with open(label_file, 'r') as fin:
        lines = fin.readlines()
        header = lines[0].rstrip().split(',')
        header[0] = DICT_KEY_SCORE
        header[2] = 'img_name'
        for idx, l in enumerate(lines[1:]):
            l_toks = l.rstrip().split(',')
            # class_level, region, img_name, kp_level, lw_level, sj_level = l_toks
            cur_dict = dict(zip(header, l_toks))
            img_name = cur_dict[header[2]]
            # img_name_toks = img_name.split()
            # img_name_core = [x for x in img_name_toks if 'ouse' in x or 'ousse' in x][0]
            img_loc = glob.glob('{}/**/*{}*'.format(img_folder, img_name), recursive=True)
            if len(img_loc)==0:
                print(idx, ',', img_name, ',', img_loc)

                # try 0001 fix
                index_0001 = img_name.index('_0001') + 5
                if img_name[index_0001]!=' ':
                    img_name = '{} {}'.format(img_name[:index_0001], img_name[index_0001:])
                else:
                    img_name = '{}{}'.format(img_name[:index_0001], img_name[index_0001:])
                img_loc = glob.glob('{}/**/*{}*'.format(img_folder, img_name), recursive=True)

                if len(img_loc)==0:
                    print('missing 2', idx, ',', img_name, ',', img_loc)
                    # try ra fix
                    r_suffix = 'RA' if region=='AC' else 'RR'
                    index_ra = img_name.upper().index(r_suffix)+2
                    if img_name[index_ra]!='':  # add space
                        img_name = '{} {}'.format(img_name[:index_ra], img_name[index_ra:])
                    else:   # remove space
                        img_name = '{}{}'.format(img_name[:index_ra], img_name[index_ra:])
                    img_loc = glob.glob('{}/**/*{}*'.format(img_folder, img_name), recursive=True)
                    if len(img_loc)==0:
                        print('missing 3', idx, ',', img_name, ',', img_loc)

            if len(img_loc)>0:
                cur_dict[DICT_KEY_LOC] = img_loc[0]
                if img_name not in label_dict:
                    label_dict[img_name]= cur_dict
                else:
                    print('duplicate img_name:', img_name, cur_dict==label_dict[img_name])
    fin.close()

    with open(label_json, 'w') as fout:
        json.dump(label_dict, fout)
    fout.close()
    return label_dict


def compute_kappa(region_dict):
    img_names, img_names_train, img_names_valid, img_names_test = [],[],[],[]
    scores = []
    for img_name, img_dict in region_dict.items():
        img_names.append(img_name)
        if 'ouse3' in img_name or 'ouse13' in img_name:
            img_names_test.append(img_name)
        elif 'ouse10' in img_name or 'ousse10' in img_name or 'ouse16' in img_name:
            img_names_valid.append(img_name)
        else:
            img_names_train.append(img_name)
        cur_vals = (img_dict['KP'], img_dict['LW'], img_dict['SJ'])
        scores.append(cur_vals)
    scores = np.array(scores)
    img_names = np.array(img_names)
    img_names_train = np.array(img_names_train)
    img_names_valid = np.array(img_names_valid)
    img_names_test = np.array(img_names_test)

    from sklearn.metrics import cohen_kappa_score
    observer = ['KP', 'LW', 'SJ']
    img_names_dict = {'all':img_names, 'train':img_names_train, 'valid':img_names_valid, 'test':img_names_test}
    for d_name in ['train', 'valid', 'test']:
        img_names_subset = img_names_dict[d_name]
        img_names_common, comm1, comm2 = np.intersect1d(img_names, img_names_subset, return_indices=True)
        scores_sub = scores[comm1, :]

        for kappa_type in [None, 'linear', 'quadratic']:
            for idx in range(2):
                for jdx in range(idx+1,3):
                    if kappa_type is not None:
                        k = cohen_kappa_score(scores_sub[:, idx], scores_sub[:, jdx], weights=kappa_type)
                    else:
                        k = cohen_kappa_score(scores_sub[:, idx], scores_sub[:, jdx])
                    print(d_name, kappa_type, observer[idx], observer[jdx], k)
    return


def generate_csv_from_json(region='AC', f=img_folder):
    out_csv = os.path.join(f, 'processed_data_{}.csv'.format(region))
    label_json = os.path.join(f, 'label_paths_{}.json'.format(region))
    fin = open(label_json).read()
    label_dict = json.loads(fin)

    if f==img_folder or region=='PC':
        col_names = ['img_name', 'img_loc', 'score', 'KP', 'SJ', 'LW']
    else:
        col_names = ['img_name', 'img_loc', 'score', 'KLP score', 'SJ score', 'LW score']
    with open(out_csv, 'w') as fout:
        fout.write('{}\n'.format(','.join(col_names)))
        for key, key_dict in label_dict.items():
            cur_vals = [key_dict[x] for x in col_names]
            fout.write('{}\n'.format(','.join([str(x) for x in cur_vals])))
    fout.close()

    return


def read_labels_20210217(region='AC', f=img_folder):
    label_json = os.path.join(f, 'label_paths_{}.json'.format(region))
    if os.path.isfile(label_json):
        fin = open(label_json).read()
        label_dict = json.loads(fin)
        return label_dict

    label_file = os.path.join(f, 'labels_{}_latest.csv'.format(region))
    label_dict = {}
    with open(label_file, 'r') as fin:
        lines = fin.readlines()
        header = lines[0].rstrip().split(',')
        header[0] = 'img_name'
        header[1] = DICT_KEY_SCORE
        for idx, l in enumerate(lines[1:]):
            l_toks = l.rstrip().split(',')
            # class_level, region, img_name, kp_level, lw_level, sj_level = l_toks
            cur_dict = dict(zip(header, l_toks))
            img_name = cur_dict['img_name']
            img_loc = glob.glob('{}/**/*{}*'.format(f, img_name), recursive=True)
            if len(img_loc)==0:
                print(idx, ',', img_name, ',', img_loc)

            if len(img_loc)>0:
                cur_dict[DICT_KEY_LOC] = img_loc[0]
                if img_name not in label_dict:
                    label_dict[img_name]= cur_dict
                else:
                    print('duplicate img_name:', img_name, cur_dict==label_dict[img_name])
    fin.close()

    with open(label_json, 'w') as fout:
        json.dump(label_dict, fout)
    fout.close()
    return label_dict


def resize_image(image, min_value, max_value, diff_value, max_rows=512):
    extra = max_rows - diff_value
    min_value = int(min_value - math.ceil(extra / 3.0))
    max_value = int(max_value + math.floor(extra / 3.0*2))
    if max_value>1024:
        max_diff = max_value-1024
        image = image[min_value-max_diff:max_value-max_diff, :]
    elif min_value<0:
        image = image[0:max_value-min_value, :]
    else:
        image = image[min_value:max_value, :]
    return image


def crop_image(img, q=70, ignore_top=50):
    target_size = (512, 512)
    img_shape = img.shape
    row_means = np.nanmean(img, axis=1)
    # ignore top 50 rows
    row_means = row_means[ignore_top:]

    threshold = np.percentile(row_means, q=[q])    # want 216 from 768 about 72%
    valid_row_indices = np.nonzero(row_means>threshold)[0]
    min_row_index = min(valid_row_indices) + ignore_top
    max_row_index = max(valid_row_indices) + ignore_top

    diff_image = max_row_index-min_row_index
    resized_image = resize_image(img, min_row_index, max_row_index, diff_image, max_rows=target_size[0])

    # plt.figure(1)
    # plt.clf()
    # plt.imshow(img)
    # plt.axhline(min_row_index)
    # plt.axhline(max_row_index)
    # plt.figure(2)
    # plt.clf()
    # plt.imshow(resized_image)
    return resized_image


def scrambled_split(folder):
    train_img_file = os.path.join(folder, 'training_img_labels_all.csv')
    train_imgs = []
    with open(train_img_file, 'r') as fin:
        for l in fin.readlines():
            train_imgs.append(l.rstrip())
    fin.close()

    valid_img_file = os.path.join(folder, 'validation_img_labels_all.csv')
    valid_imgs = []
    with open(valid_img_file, 'r') as fin:
        for l in fin.readlines():
            valid_imgs.append(l.rstrip())
    fin.close()

    test_img_file = os.path.join(folder, 'test_img_labels_all.csv')
    test_imgs = []
    with open(valid_img_file, 'r') as fin:
        for l in fin.readlines():
            test_imgs.append(l.rstrip())
    fin.close()

    all_imgs = train_imgs + valid_imgs + test_imgs
    dup_imgs = []
    unique_imgs = []
    for x in all_imgs:
        if x not in unique_imgs:
            unique_imgs.append(x)
        else:
            dup_imgs.append(x)
    with open('duplicate_imgs.csv', 'w') as fout:
        for x in dup_imgs:
            fout.write('{}\n'.format(x))
    fout.close()

    all_imgs = np.unique(all_imgs)  # remove duplicates! there are duplicates in train_imgs and (valid_imgs + test_imgs)
    all_imgs = list(all_imgs)
    num_imgs = len(all_imgs)
    num_train = math.ceil(num_imgs * 0.8)
    num_valid = math.floor(num_imgs * 0.1)
    num_test = num_imgs - num_train - num_valid
    random.shuffle(all_imgs)

    train_imgs_shuffled = all_imgs[:num_train]
    valid_imgs_shuffled = all_imgs[num_train:num_train+num_valid]
    test_imgs_shuffled = all_imgs[num_train+num_valid:]

    cls_breakdown(all_imgs)
    cls_breakdown(train_imgs_shuffled)
    cls_breakdown(valid_imgs_shuffled)
    cls_breakdown(test_imgs_shuffled)

    with open(train_img_file.replace('_all', ''), 'w') as fout:
        for t in train_imgs_shuffled:
            fout.write('{}\n'.format(t))
    fout.close()
    with open(valid_img_file.replace('_all', ''), 'w') as fout:
        for t in valid_imgs_shuffled:
            fout.write('{}\n'.format(t))
    fout.close()
    with open(test_img_file.replace('_all', ''), 'w') as fout:
        for t in test_imgs_shuffled:
            fout.write('{}\n'.format(t))
    fout.close()
    return


def cls_breakdown(img_list):
    cls_dict = {}
    for img in img_list:
        img_toks = img.split(',')
        if img_toks[0] not in cls_dict:
            cls_dict[img_toks[0]] = []
        cls_dict[img_toks[0]].append(img_toks[1])

    for key in cls_dict.keys():
        print(key, len(cls_dict[key]))
    return cls_dict


if __name__ == '__main__':
    # check_images()
    # make_training_test_split_data(f=img_folder, eye_level=False, nrow=512, ncol=512)
    # generate_csv_from_json(region='AC')
    # generate_csv_from_json(region='PC')

    # # ## new 20210217 data
    # f= os.path.join(img_folder, 'Images for AI analysis')
    # read_labels_20210217('AC', f)
    # read_labels_20210217('PC', f)
    # # generate_csv_from_json(region='AC', f=f)
    # # generate_csv_from_json(region='PC', f=f)
    #
    # # make_training_test_split_data(f=f, eye_level=False, nrow=1024, ncol=1024)
    # make_training_test_split_data(f=f, eye_level=False, nrow=512, ncol=512)
    # # make_training_test_split_data(f=f, eye_level=False, nrow=512, ncol=1024)

    # # ## 2021-07-07 data
    # img_folder2 = os.path.join(base_folder, 'classification', 'data', '2021.07.07 New TIFFs')
    # read_labels_20210217('AC', img_folder2)
    # read_labels_20210217('PC', img_folder2)
    # # # # generate_csv_from_json(region='AC', f=img_folder2)
    # # # # generate_csv_from_json(region='PC', f=img_folder2)
    # # # make_training_test_split_data(f=img_folder2, eye_level=False, nrow=512, ncol=512)
    # make_training_test_split_data(f=img_folder2, eye_level=False, nrow=512, ncol=1024)

    folder = os.path.join(base_folder, 'classification', 'data', '2021.07.07 New TIFFs', 'AC_imgs_r512_c512')
    # folder = os.path.join(base_folder, 'classification', 'data', '2021.07.07 New TIFFs', 'PC_imgs_r512_c512')
    # folder = os.path.join(base_folder, 'classification', 'data', '2021.07.07 New TIFFs', 'AC_imgs_r512_c1024')
    # folder = os.path.join(base_folder, 'classification', 'data', '2021.07.07 New TIFFs', 'PC_imgs_r512_c1024')
    scrambled_split(folder)
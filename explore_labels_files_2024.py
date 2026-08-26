import os, cv2, json, csv, random
import numpy as np

from classification_data import base_folder, get_png_from_tiff, crop_image
VALID_LABELS = ["0", "0.5", "1", "2", "3", "4"]
VALID_LABELS_NUMERIC = [float(x) for x in VALID_LABELS]


def check_images(img_folder, ext):
    img_out_file = "{}_imgs.txt".format(img_folder)
    imgs_all = os.listdir(img_folder)
    imgs_ext = [x for x in imgs_all if ext in x]
    imgs_dict = {}

    with open(img_out_file, "w") as fout:
        for x in imgs_ext:
            imgs_dict[x.replace(ext, "")] = x
            fout.write("{}\n".format(x))
        fout.close()

    return imgs_dict


def keys_in_common(data_tiff_dict, data_ac_dict, data_pc_dict=None):
    data_ac_keys = set(data_ac_dict.keys())

    data_tiff_keys_raw = list(data_tiff_dict.keys())
    if data_pc_dict is not None:
        data_tiff_keys = []
        for x in data_tiff_keys_raw:
            temp_toks = x.split(" ")
            data_tiff_keys.append(" ".join(temp_toks[1:]))
    else:
        data_tiff_keys = data_tiff_keys_raw

    common_keys_ac = data_ac_keys.intersection(data_tiff_keys)
    if data_pc_dict is not None:
        data_pc_keys = set(data_pc_dict.keys())
        common_keys_pc = data_pc_keys.intersection(data_tiff_keys)
        unused_keys = set(data_tiff_keys).difference(data_ac_keys).difference(data_pc_keys)
    else:
        common_keys_pc = []
        unused_keys = set(data_tiff_keys).difference(data_ac_keys)
    print(len(common_keys_ac), len(common_keys_pc), len(unused_keys))
    return common_keys_ac, common_keys_pc, unused_keys


def labels_vs_images():
    # # data_tiff_dict = check_images(os.path.join(base_folder, "classification", "data", "Images for AI analysis", "TIFFs for analysis"), ext=".TIFF")
    # data_ac_dict = check_images(os.path.join(base_folder, "classification", "data", "Images for AI analysis", "AC_imgs_r512_c512"), ext=".png")
    # data_pc_dict = check_images(os.path.join(base_folder, "classification", "data", "Images for AI analysis", "PC_imgs_r512_c512"), ext=".png")
    # # files in common
    # common_keys_ac, common_keys_pc, unused_keys = keys_in_common(data_tiff_dict, data_ac_dict, data_pc_dict)
    # # 867 805 0 - checks out!

    ## these seem incomplete
    # data_tiff_2021_dict = check_images(os.path.join(base_folder, "classification", "data", "2021.07.07 New TIFFs"), ext=".TIFF")
    # data_ac_2021_dict = check_images(os.path.join(base_folder, "classification", "data", "2021.07.07 New TIFFs", "AC_imgs_r512_c512"), ext=".png")
    # data_pc_2021_dict = check_images(os.path.join(base_folder, "classification", "data", "2021.07.07 New TIFFs", "PC_imgs_r512_c512"), ext=".png")

    # ## check against this instead
    # data_ac_2021_dict = check_images(os.path.join(base_folder, "classification", "data", "Images for AI analysis", "labels_20210217", "AC_imgs_r512_c512"), ext=".png")   # png count matches labels_AC_latest.csv
    # data_pc_2021_dict = check_images(os.path.join(base_folder, "classification", "data", "Images for AI analysis", "labels_20210217", "PC_imgs_r512_c512"), ext=".png")   # png count matches labels_PC_latest.csv
    # # files in common
    # common_keys_ac_2021, _, unused_keys_2021_ac = keys_in_common(data_ac_2021_dict, data_ac_dict)
    # # 870 0 0 - all accounted for
    # common_keys_pc_2021, _, unused_keys_2021_pc = keys_in_common(data_pc_2021_dict, data_pc_dict)
    # # 828 0 0 - all accounted for

    # train and test label split
    ac_file_2024 = os.path.join(base_folder, "data_2024", "corrections_AC_new.csv")     # made 2025-03-20
    pc_file_2024 = os.path.join(base_folder, "data_2024", "corrections_PC_new.csv")     # made 2025-03-20
    # parse into 2 dictionaries
    labels_dict_ac = read_label_files(ac_file_2024)     # duplicates are same - 865 vs 870
    labels_dict_pc = read_label_files(pc_file_2024)     # duplicates are same - 827 vs 828
    # check missingness - losing a couple
    # compare against old labels - ok

    # new labels and images
    new_image_files = np.loadtxt("all_new_images.txt", dtype=str, delimiter=",")
    label_dict_final, bad_files = read_final_score_2025()

    # what is available and what is missing vs new_image_files and labels_dict_ac and labels_dict_pc
    labels_final_all_raw = list(label_dict_final["AC"].keys()) + list(label_dict_final["PC"].keys())
    labels_final_all = [x.replace(".TIFF", "") for x in labels_final_all_raw]
    new_image_files_set = set([x.replace(".tiff","") for x in new_image_files])
    images_found_2024 = new_image_files_set.intersection(labels_final_all)
    print(len(new_image_files_set), len(labels_final_all), len(images_found_2024))  # all images found in labels! - 2147 new and 2147 found
    missing_images = set(labels_final_all).difference(new_image_files_set)
    np.savetxt("images_no_access.txt", list(missing_images), fmt="%s", delimiter=",")

    # total 3971 labels
    labels_all_prev = list(labels_dict_ac.keys()) + list(labels_dict_pc.keys())
    images_prev = set(labels_all_prev).intersection(labels_final_all)
    print(len(labels_all_prev), len(labels_final_all), len(images_prev))  # no overlap

    # check with double loop with mouse up to laterality
    mouse_dict_prev = {}
    for ldx, label_name_raw in enumerate(labels_all_prev):
        label_toks = label_name_raw.split("_")
        label_name = "_".join(label_toks[:2])

        if label_name not in mouse_dict_prev:
            mouse_dict_prev[label_name] = []
        mouse_dict_prev[label_name].append(label_name_raw)

    mouse_dict_final = {}
    for ldx2, label_name_raw in enumerate(labels_final_all):
        mouse_name = get_mouse_name(label_name_raw)
        if mouse_name not in mouse_dict_final:
            mouse_dict_final[mouse_name] = []
        mouse_dict_final[mouse_name].append(label_name_raw)

    mice_final = list(mouse_dict_final.keys())
    mice_prev = list(mouse_dict_prev.keys())
    common_mice = set(mice_final).intersection(mice_prev)
    print(len(mice_final), len(mice_prev), len(common_mice))  # no overlap

    # 30 min
    # create resized images 30min
    # split train-valid-test by mouse name up to structure
    return label_dict_final, mice_final, mice_prev


def get_mouse_name(label_name):
    label_toks = label_name.split("_")
    mouse_name = "_".join(label_toks[:2])
    return mouse_name


def create_images_for_DL(label_dict_final, nrow=512, ncol=500):
    new_image_files = np.loadtxt("all_new_images.txt", dtype=str, delimiter=",")
    split_dict = create_training_split(label_dict_final)

    split_files_dict = {"AC":{}, "PC":{}}
    img_folder = os.path.join(base_folder, "data_2024", "img_folder")
    for idx, img_name in enumerate(new_image_files):
        tiff_file = os.path.join(img_folder, img_name)
        if ".tiff" in img_name:
            img_key = img_name.replace(".tiff", "")
        elif ".TIFF" in img_name:
            img_key = img_name.replace(".tiff", "")
        else:
            img_key = img_name.lower().replace(".tiff", "")

        region = None
        if img_key in label_dict_final["AC"]:
            region = "AC"
        elif img_key in label_dict_final["PC"]:
            region = "PC"
        else:
            print(img_key, img_name)
            continue

        out_folder = os.path.join(base_folder, "data_2024", "{}_imgs_r{}_c{}".format(region, nrow, ncol))
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        img_out_path = os.path.join(out_folder, "{}.png".format(img_key))
        if not os.path.isfile(img_out_path):
            img_out = create_image_for_DL(tiff_file, region=region, nrow=nrow, ncol=ncol)
            cv2.imwrite(img_out_path, img_out)

        # split by mouse
        mouse_name = get_mouse_name(img_key)
        img_split = split_dict[region][mouse_name]
        if img_split not in split_files_dict[region]:
            split_files_dict[region][img_split] = []
        split_files_dict[region][img_split].append([img_key, img_name, label_dict_final[region][img_key]["Final Score"], img_out_path])

    # write split files
    for region, region_data in split_files_dict.items():
        for split, region_files_data in region_data.items():
            region_folder = os.path.join(base_folder, "data_2024", "{}_imgs_r{}_c{}".format(region, nrow, ncol))
            split_path = os.path.join(region_folder, "{}_img_labels.csv".format(split))
            with open(split_path, "w") as fout:
                for rdx, rdata in enumerate(region_files_data):
                    print([region, split, rdx, rdata[-2], rdata[-1]])
                    #if rdata[-2]=='100' or len(rdata[-2])==0:   # bad labels? -> bad_labels.txt
                    if rdata[-2] not in VALID_LABELS:
                        print("invalid label:", rdata[-2], rdata[-1])
                        continue
                    fout.write("{},{}\n".format(rdata[-2], rdata[-1]))
            fout.close()
    return split_files_dict


def create_image_for_DL(tiff_file, region="AC", nrow=512, ncol=500):
    if nrow==ncol:
        img_out = get_png_from_tiff(tiff_file, nrow=nrow, ncol=ncol)    # not cropped for 512*512
    else:
        img = get_png_from_tiff(tiff_file, nrow=1024, ncol=1024)
        ignore_top = 100 if region=="AC" else 80
        q = 60 if region=='PC' else 70
        img_out = crop_image(img, ignore_top=ignore_top, q=q)

    return img_out


def create_training_split(label_dict_final, split_folder=os.path.join(base_folder, "data_2024")):
    split_ratio = [.8, .1, .1]
    # path_train = os.path.join(split_folder, "training_img_labels.csv")
    # path_valid = os.path.join(split_folder, "validation_img_labels.csv")
    # path_test = os.path.join(split_folder, "test_img_labels.csv")

    random.seed(6)  # fix seed
    # get previous splits - doesnt seem to matter as new mice!

    split_dict = {}
    mice_region_dict = {}
    for region in ["AC", "PC"]:
        mice_region_dict[region] ={}
        split_dict[region] = {}
        region_data = label_dict_final[region]
        for idx, img_key in enumerate(region_data.keys()):
            mouse_name = get_mouse_name(img_key)
            if img_key not in mice_region_dict[region]:
                mice_region_dict[region][mouse_name] = []
            mice_region_dict[region][mouse_name].append(img_key)

        region_mice = list(mice_region_dict[region])
        num_mice_region = len(region_mice)
        random.shuffle(region_mice)
        num_train = round(num_mice_region*split_ratio[0])
        num_valid = round(num_mice_region*split_ratio[1])
        num_test = num_mice_region - num_train - num_valid
        mice_train = region_mice[:num_train]
        mice_valid = region_mice[num_train:(num_train+num_valid)]
        mice_test = region_mice[(num_train+num_valid):]
        for mdx, mice_name in enumerate(mice_train):
            split_dict[region][mice_name] = "train"
        for mdx, mice_name in enumerate(mice_valid):
            split_dict[region][mice_name] = "validation"
        for mdx, mice_name in enumerate(mice_test):
            split_dict[region][mice_name] = "test"

    return split_dict


def read_label_files(fpath):    # 15min
    counter = 0

    label_dict = {}
    with open(fpath, "r") as fin:
        freader = csv.reader(fin, delimiter=",", quotechar='"')
        for row in freader:
            counter += 1
            if counter == 1:
                header = row
            elif counter > 1:
                cur_dict = dict(zip(header, row))
                img_name = cur_dict["Tiff"]
                img_key = img_name.replace(".TIFF", "")
                if img_key not in label_dict:
                    label_dict[img_key] = cur_dict
                else:
                    print(cur_dict, label_dict[img_key])    # duplicates are same

    print(counter-1, len(label_dict))     # sanity check - should be same length
    return label_dict


def read_final_score_2025():
    fpath = os.path.join(base_folder, "data_2024", "final_score_textOnly_new.csv")
    label_dict = {"AC":{}, "PC":{}}
    bad_files = []

    counter = 0
    with open(fpath, "r") as fin:
        freader = csv.reader(fin, delimiter=",", quotechar='"')
        for row in freader:
            counter += 1
            if counter == 1:
                header = row
            elif counter > 1:
                cur_dict = dict(zip(header, row))
                img_name = cur_dict["TIFF Name"]
                # img_key = img_name.replace(".TIFF", "")
                img_key = img_name
                # final_score = cur_dict["Final Score"]

                # # AC vs PC: if R_2 = PC, if R_4 = AC
                if "R_2" in img_name or "L_2" in img_name:
                    img_type = "PC"
                elif "R_4" in img_name or "L_4" in img_name:
                    img_type = "AC"
                else:
                    bad_files.append(img_name)

                if img_key not in label_dict[img_type]:
                    label_dict[img_type][img_key] = cur_dict
                else:
                    print(cur_dict, label_dict[img_type][img_key])  # duplicates are same
    return label_dict, bad_files


def get_new_image_names():
    new_image_folder = os.path.join(base_folder, "data_2024", "img_folder")
    all_files = os.listdir(new_image_folder)
    np.savetxt("all_new_images.txt", all_files, fmt="%s")
    return


def fix_2021_labels(region):
    new_folder = os.path.join(base_folder, "data_2024", "{}_imgs_r512_c512".format(region))
    correction_file_2024 = os.path.join(base_folder, "data_2024", "corrections_{}_new.csv".format(region))     # made 2025-03-20
    labels_dict = read_label_files(correction_file_2024)

    old_folder = os.path.join(base_folder, "classification", "data", "Images for AI analysis", "{}_imgs_r512_c512".format(region))
    for split in ["training", "validation", "test"]:
        old_file_path = os.path.join(old_folder, "{}_img_labels.csv".format(split))
        old_data = np.loadtxt(old_file_path, dtype=str, delimiter=",")
        old_paths = old_data[:,1]

        new_file_path = os.path.join(new_folder, "{}_img_labels_2021.csv".format(split))
        with open(new_file_path, "w") as fout:
            for odx, path_name in enumerate(old_paths):
                old_toks = path_name.split("/")
                old_name = old_toks[-1]
                old_key = old_name.replace(".png", "")

                if old_key in labels_dict:
                    new_val = labels_dict[old_key]
                    new_val_score = new_val["Score"]
                    if new_val_score in VALID_LABELS:
                        fout.write("{},{}\n".format(new_val_score, path_name))
                    else:
                        print(new_val_score, new_val, path_name)
        fout.close()
    return



if __name__ == '__main__':
    get_new_image_names()
    # labels_vs_images()

    # # crop/resize new images
    label_dict_final, bad_files = read_final_score_2025()
    # np.savetxt("uncertain_ac_pc_files.txt", bad_files, fmt="%s", delimiter=",")     # blank - good
    create_images_for_DL(label_dict_final, nrow=512, ncol=512)

    # fix_2021_labels(region="AC")
    # fix_2021_labels(region="PC")
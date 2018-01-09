# plot the log file for better understanding
import glob, os, cv2
import numpy as np
import matplotlib.pyplot as plt
from chamber_tracer import load_img, load_data, \
    FILE_BASE, LOG_FOLDER, FRAME_HEIGHT, FRAME_WIDTH, NUM_ACTIONS, IMG_DIR, JSON_DIR

ADD_LAST = 1
CHEAT_PEEK = 0
ADJUST_LOSS = 0
REWARD_ONLY_NEXT_TRUE = 1
REWARD_STATIC_PENALTY = 1

LOG_FOLDER_ALL_IMAGES = './dqn_log/logs_all_imgs'


def review_all_logs(pop_truth=0):
    for frame_height in [64, 128]:
        # for add_last in [0, 1]:
        for add_last in [1]:
            # for reward_next in [0, 1]:
            for reward_next in [0]:
                for pop_truth in [0, 1]:
                    frame_width = frame_height
                    num_actions = 8
                    cheat_peek = 0
                    adjust_loss = 0
                    static_penalty = 0
                    log_pattern = 'h{}_w{}_n{}_l{}_c{}_a{}_r{}_s{}'.format(frame_height, frame_width, num_actions,
                                                                           add_last, cheat_peek, adjust_loss,
                                                                           reward_next, static_penalty)
                    experiment_logs = sorted(list(glob.glob('{}/{}*.txt'.format(LOG_FOLDER, log_pattern))))
                    for experiment_log in experiment_logs:
                        experiment_log = experiment_log.replace(LOG_FOLDER+'\\', '')
                        episode_num = int(experiment_log.split('_')[0].replace('e', ''))
                        review_log(episode_num, frame_height=frame_height, frame_width=frame_width,
                                   num_actions=num_actions, add_last=add_last, cheat_peek=cheat_peek,
                                   adj_loss=adjust_loss, next_true=reward_next, static_penalty=static_penalty,
                                   pop_truth=pop_truth, better_coords=True, save_imgs=True)
    return


def review_log(episode_num, frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH, num_actions=NUM_ACTIONS,
               add_last=ADD_LAST, cheat_peek=CHEAT_PEEK, adj_loss=ADJUST_LOSS, next_true=REWARD_ONLY_NEXT_TRUE,
               static_penalty=REWARD_STATIC_PENALTY, pop_truth=0, better_coords=True, save_imgs=False):
    img, coords = load_img(better_coords=better_coords)

    # episode_num = 143
    if pop_truth:
        log_file = '{}/h{}_w{}_l{}_c{}_r{}_p{}_e{}.txt'.format(LOG_FOLDER, frame_height, frame_width, add_last,
                                                               cheat_peek, next_true, pop_truth, episode_num)
    else:
        log_file = '{}/e{}_h{}_w{}_n{}_l{}_c{}_a{}_r{}_s{}.txt'.format(LOG_FOLDER, episode_num, frame_height,
                                                                       frame_width, num_actions, add_last, cheat_peek,
                                                                       adj_loss, next_true, static_penalty)
    if save_imgs:
        plt.switch_backend('agg')
        if pop_truth:
            img_out_folder = '{}/figs/h{}_w{}_l{}_r{}_p{}_e{}'.format(LOG_FOLDER, frame_height, frame_width, add_last,
                                                                      next_true, pop_truth, episode_num)
        else:
            img_out_folder = '{}/figs/h{}_w{}_l{}_r{}_e{}'.format(LOG_FOLDER, frame_height, frame_width, add_last,
                                                                  next_true, episode_num)
        if not os.path.exists(img_out_folder):
            os.makedirs(img_out_folder)

    lines = []
    with open(log_file, 'r') as fin:
        for l in fin:
            cur_toks = l.rstrip().split(",")
            lines.append([float(x) for x in cur_toks])

    log_data = np.asarray(lines)
    # check total reward
    total_reward = np.sum(log_data[:, 1])
    avg_reward = total_reward/len(log_data)

    for idx in range(log_data.shape[0]):
        prev_data = log_data[:idx, ]
        iter_data = log_data[idx, ]
        if idx % 500 == 0 and idx < 100000:  # and idx < 25000:
            if save_imgs:
                if pop_truth:
                    save_path = '{}/h{}_w{}_l{}_c{}_r{}_p{}_e{}_i{}.png'\
                        .format(img_out_folder, frame_height, frame_width, add_last, cheat_peek, next_true, pop_truth,
                                episode_num, idx)
                else:
                    save_path = '{}/h{}_w{}_n{}_l{}_c{}_a{}_r{}_s{}_e{}_i{}.png'\
                        .format(img_out_folder, frame_height, frame_width, num_actions, add_last, cheat_peek, adj_loss,
                                next_true, static_penalty, episode_num, idx)
                display_iter(iter_data, prev_data, img, coords, episode_num, save_path=save_path)
    return


def display_iter(cur_iter_data, prev_data, img, coords, episode_num, save_path=None):
    num_points = prev_data.shape[0]
    plt.imshow(img)
    plt.scatter(x=prev_data[:, 2], y=prev_data[:, 3], c='red', s=2)
    plt.scatter(x=cur_iter_data[2], y=cur_iter_data[3], c='blue', s=3)
    cum_reward = np.sum(prev_data[:, 1])
    # plt.title('idx={}; cum_reward={:.2f}; action={}; reward={:.2f}'.format(num_points, cum_reward,
    #                                                                          int(cur_iter_data[0]), cur_iter_data[1]))
    plt.title('idx={}; cum_reward={:.2f};'.format(num_points, cum_reward))
    if save_path:
        plt.savefig(save_path)
    return


def check_all_logs():
    # fixed variables
    f_height = FRAME_HEIGHT
    f_width = FRAME_WIDTH
    adjust_loss = 0
    num_actions = NUM_ACTIONS

    # variables under consideration
    add_last_options = [0, 1]
    cheat_peek_options = [0, 1]
    reward_next_options = [0, 1]
    static_penalty_options = [0, 1]

    all_log_files = []
    avg_rewards = []
    for add_last in add_last_options:
        for cheat_peek in cheat_peek_options:
            for reward_next in cheat_peek_options:
                for static_penalty in static_penalty_options:
                    log_pattern = 'h{}_w{}_n{}_l{}_c{}_a{}_r{}_s{}'.format(f_height, f_width, num_actions,
                                                                                 add_last, cheat_peek, adjust_loss,
                                                                                 reward_next, static_penalty)
                    pnames = sorted(list(glob.glob('{}/*_{}.txt'.format(LOG_FOLDER, log_pattern))))
                    for pname in pnames:
                        lines = []
                        print('processing {}'.format(pname))
                        with open(pname, 'r') as fin:
                            for l in fin:
                                cur_toks = l.rstrip().split(",")
                                lines.append([float(x) for x in cur_toks])

                            log_data = np.asarray(lines)
                            num_points = len(log_data)
                            # check total reward
                            if num_points < 500:    # didnt do anything
                                continue
                            if num_points > 15000:
                                num_points = 15000

                            total_reward = np.sum(log_data[:num_points, 1])
                            avg_rewards.append(total_reward/num_points)
                            all_log_files.append(pname)
    # sort by avg_reward and see what files pop out
    s_list = sorted(enumerate(avg_rewards), key=lambda x:x[1], reverse=True)
    s_files = [all_log_files[x[0]] for x in s_list]
    for idx, x in enumerate(s_list):
        print(idx, avg_rewards[x[0]], all_log_files[x[0]])
    return all_log_files, avg_rewards


def make_movie(image_folder):
    video_name = '{}/{}_video.avi'.format('dqn_log', image_folder)

    images = [img for img in os.listdir(os.path.join('dqn_log', 'figs', image_folder)) if img.endswith(".png")]
    images = sorted(images, key=lambda img: int(img.split('_')[-1].replace('.png', '').replace('i', '')))
    frame = cv2.imread(os.path.join('dqn_log', 'figs', image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join('dqn_log', 'figs', image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    return


def make_movie_ffmpeg(folder):
    'h64_w64_l0_r0_e298'
    # 'h64_w64_n8_l0_c0_a0_r0_s0_e298_i500'
    folder_toks = folder.split('_')
    file_pattern = '{}_{}_{}_{}_{}_{}_{}_{}'.format(folder_toks[0], folder_toks[1], 'n8', folder_toks[2], 'c0_a0', folder_toks[3], 's0', folder_toks[-1])
    # os.system("ffmpeg -r 1 -i img%01d.png -vcodec mpeg4 -y movie.mp4")
    # FIXME - doesnt seem to recognize %01d.png in windows
    os.system("ffmpeg -r 1 -i {}_i%01d.png -vcodec mpeg4 -y {}/{}_movie.mp4".format(file_pattern, folder, folder))
    return


# general log image - need to grab correct image and coords
def review_general_logs():
    for frame_height in [64]:
        for add_last in [1]:
            frame_width = frame_height
            pop_truth = 0
            log_pattern = 'h{}_w{}_l{}_p{}_e'.format(frame_height, frame_width, add_last, pop_truth)
            experiment_logs = sorted(list(glob.glob('{}/{}*.txt'.format(LOG_FOLDER_ALL_IMAGES, log_pattern))))
            for experiment_log in experiment_logs:
                experiment_log = experiment_log.replace(LOG_FOLDER+'\\', '')
                episode_num = int(experiment_log.split('_')[-2].replace('e', ''))
                if episode_num > 600:
                    img_num = int(experiment_log.split('_')[-1].replace('.txt', '').replace('i', ''))
                    review_general_log(episode_num, img_num, frame_height=frame_height, frame_width=frame_width,
                                       add_last=add_last, pop_truth=pop_truth, better_coords=True, save_imgs=True)
    return


def review_general_log(episode_num, img_num, frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH, add_last=ADD_LAST,
                   pop_truth=0, better_coords=True, save_imgs=False):
    # img, coords = load_img(better_coords=better_coords)
    imgs, img_names, coords = load_data(img_folder=IMG_DIR, json_folder=JSON_DIR)
    cur_img = imgs[img_num, ]

    # reformat coords
    cur_coords = coords[img_num, ]
    xs = cur_coords[0]
    ys = cur_coords[1]
    coords_np = np.ndarray(shape=(len(xs), 2), dtype=np.int64)
    coords_np[:, 0] = xs
    coords_np[:, 1] = ys

    log_file = '{}/h{}_w{}_l{}_p{}_e{}_i{}.txt'.format(LOG_FOLDER_ALL_IMAGES, frame_height, frame_width, add_last,
                                                       pop_truth, episode_num, img_num)
    if save_imgs:
        plt.switch_backend('agg')
        img_out_folder = '{}/figs/h{}_w{}_l{}_p{}_e{}'.format(LOG_FOLDER, frame_height, frame_width,
                                                              add_last, pop_truth, episode_num)
        if not os.path.exists(img_out_folder):
            os.makedirs(img_out_folder)

    lines = []
    with open(log_file, 'r') as fin:
        for l in fin:
            cur_toks = l.rstrip().split(",")
            lines.append([float(x) for x in cur_toks])

    log_data = np.asarray(lines)
    # check total reward
    total_reward = np.sum(log_data[:, 1])
    avg_reward = total_reward/len(log_data)

    for idx in range(log_data.shape[0]):
        prev_data = log_data[:idx, ]
        iter_data = log_data[idx, ]
        if idx % 500 == 0 and idx < 100000:  # and idx < 25000:
            if save_imgs:
                save_path = '{}/h{}_w{}_l{}_p{}_e{}_j{}.png'.format(img_out_folder, frame_height, frame_width,
                                                                    add_last, pop_truth, episode_num, idx)
                display_iter(iter_data, prev_data, cur_img, coords_np, episode_num, save_path=save_path)
    return


def review_logs_general(pattern_fun, folder='./dqn_log/logs_1img', is_prefix=False):
    all_patterns, episode_loc, img_loc = pattern_fun()
    for idx, pattern in enumerate(all_patterns):
        if is_prefix:
            experiment_logs = sorted(list(glob.glob('{}/*_{}.txt'.format(folder, pattern))))
        else:
            experiment_logs = sorted(list(glob.glob('{}/{}_*.txt'.format(folder, pattern))))
        for experiment_log in experiment_logs:
            base_name = experiment_log.replace('.txt', '').replace(folder+'\\', '')
            file_toks = base_name.split('_')
            episode_num = int(file_toks[episode_loc].replace('e', ''))
            img_num = None
            if img_loc:
                img_num = int(file_toks[img_loc].replace('i', ''))
            # img_out_folder = '{}/figs/{}_e{}'.format(folder, '_'.join([x for idx,x in enumerate(file_toks) if idx!=episode_loc]), episode_num)
            img_out_folder = '{}/figs/{}'.format(folder, base_name)

            if episode_num > 5400:
                review_log_general(experiment_log, episode_num, img_num, img_out_folder=img_out_folder, better_coords=True, save_imgs=True)
    return


def review_log_general(log_file, episode_num, img_num, img_out_folder, better_coords=True, save_imgs=False):
    # img, coords = load_img(better_coords=better_coords)
    if img_num:
        imgs, img_names, all_coords = load_data(img_folder=IMG_DIR, json_folder=JSON_DIR)
        img = imgs[img_num, ]

        # reformat coords
        cur_coords = all_coords[img_num, ]
        xs = cur_coords[0]
        ys = cur_coords[1]
        coords = np.ndarray(shape=(len(xs), 2), dtype=np.int64)
        coords[:, 0] = xs
        coords[:, 1] = ys
    else:
        img, coords = load_img(better_coords=better_coords)

    if save_imgs:
        plt.switch_backend('agg')
        if not os.path.exists(img_out_folder):
            os.makedirs(img_out_folder)

    lines = []
    with open(log_file, 'r') as fin:
        for l in fin:
            cur_toks = l.rstrip().split(",")
            lines.append([float(x) for x in cur_toks])

    log_data = np.asarray(lines)
    # check total reward
    total_reward = np.sum(log_data[:, 1])
    avg_reward = total_reward/len(log_data)

    for idx in range(log_data.shape[0]):
        prev_data = log_data[:idx, ]
        iter_data = log_data[idx, ]
        if idx % 500 == 0 and idx < 100000:  # and idx < 25000:
            save_base = log_file.replace('.txt', '').replace(LOG_FOLDER+'\\', '')
            save_path = '{}/{}_e{}_i{}.png'.format(img_out_folder, save_base, episode_num, idx)
            if save_imgs:
                display_iter(iter_data, prev_data, img, coords, episode_num, save_path=save_path)
    return


def patterns_for_single_image():
    log_patterns = []
    for frame_height in [64, 128]:
        for add_last in [1]:
            for reward_next in [0]:
                for pop_truth in [0, 1]:
                    frame_width = frame_height
                    num_actions = 8
                    cheat_peek = 0
                    adjust_loss = 0
                    static_penalty = 0
                    log_pattern = 'h{}_w{}_n{}_l{}_c{}_a{}_r{}_s{}'.format(frame_height, frame_width, num_actions,
                                                                           add_last, cheat_peek, adjust_loss,
                                                                           reward_next, static_penalty)
                    log_patterns.append(log_pattern)
    return log_patterns, 0, None


def patterns_for_edge(random_data=True):
    log_patterns = []
    for frame_height in [64]:
        for edge_check in [0, 1]:
        # for edge_check in [1]:
            for edge_penalty in [0]:
            # for edge_penalty in [0, 1]:
            #     for episode_limit in [50000]:
                for episode_limit in [10000, 50000]:
                    for intensity_multipler in [0, 1]:
                        frame_width = frame_height
                        log_pattern = 'h{}_w{}_ec{}_ep{}_el{}_im{}'.format(frame_height, frame_width, edge_check,
                                                                           edge_penalty, episode_limit, intensity_multipler)
                        log_patterns.append(log_pattern)
    return log_patterns, len(log_pattern.split('_')), len(log_pattern.split('_'))+1 if random_data else None


if __name__ == '__main__':
    # review_all_logs()
    # review_log(72, frame_height=64, frame_width=64, num_actions=8, add_last=1, cheat_peek=0, adj_loss=0, next_true=0,
    #            static_penalty=0, better_coords=True, save_imgs=True)
    # check_all_logs()

    # make movies
    # make_movie(image_folder='h64_w64_l1_r0_e67')
    # make_movie(image_folder='h64_w64_l1_r0_e70')
    # make_movie(image_folder='h64_w64_l1_r0_e73')
    # make_movie(image_folder='h64_w64_l1_r0_e74')
    # make_movie(image_folder='h64_w64_l1_r0_e88')
    # make_movie(image_folder='h64_w64_l1_r0_e101')
    # make_movie(image_folder='h128_w128_l1_r0_e44')
    # make_movie(image_folder='h128_w128_l1_r0_e50')

    # FIXME - defective for some unknown reason -> better after removing lots of files for smaller size
    # make_movie(image_folder='h64_w64_l1_r0_e102')

    # review_logs_general(pattern_fun=patterns_for_single_image)
    review_logs_general(pattern_fun=patterns_for_edge, folder=LOG_FOLDER)
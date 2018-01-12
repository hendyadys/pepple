import logging, math, random, os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches

from qlearning_data import get_coords, remove_duplicates

logger = logging.getLogger(__name__)

# current image - picked
FILE_BASE = 'DeRuyter-Inflamed_20170703mouse6_Day2_Right_160'
from sys import platform
if platform == "linux" or platform == "linux2":
    IMG_DIR = '/data/yue/pepple_dqn/acseg/segmentations'
    JSON_DIR = '/data/yue/pepple_dqn/acseg/jsons'
    LOG_FOLDER = '/data/yue/pepple_dqn/dqn_log'
    TEST_FOLDER = '/data/yue/pepple_dqn/dqn_log/test'
elif platform == "win32":
    IMG_DIR = './acseg/segmentations'
    JSON_DIR = './acseg/jsons'
    LOG_FOLDER = './dqn_log'
    TEST_FOLDER = './dqn_log/test'

# environment settings
RAW_HEIGHT = 512
RAW_WIDTH = 500
DO_CENTRALIZE = True
if DO_CENTRALIZE:
    # FRAME_HEIGHT = 128
    # FRAME_WIDTH = 128
    FRAME_HEIGHT = 64
    FRAME_WIDTH = 64
else:
    FRAME_HEIGHT = 512  # Resized frame width
    FRAME_WIDTH = 500  # Resized frame height
ADD_LAST = 1
CHEAT_PEEK = 0
# ADJUST_LOSS = 0
# 0 for original system of {1:any_true, 0:revisited, -x:divergent}
REWARD_ONLY_NEXT_TRUE = 0
# REWARD_STATIC_PENALTY = 0
POPULATE_TRUTH = 0  # populate replay memory with human trace

NO_EDGE = 0
EDGE_CHECK = 0
EDGE_PENALTY = 0
EPISODE_LIMIT = 50000
EPISODE_LIMIT = 10000
INTENSITY_MULTIPLIER = 0

# NUM_ACTIONS = 9     # directions - allows in place; this is inefficient
NUM_ACTIONS = 8     # directions - allows in place;

# new img/coord dataset per episode or not - for generalization
RANDOM_DATA = 0

# storage settings
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network


class ChamberTracer():
    def __init__(self, file_base=FILE_BASE):
        self.reset()

    def reset(self):  # reset original observation and starting coord
        # load random image for this episode
        if RANDOM_DATA:
            cur_img, img_name, coords, rand_idx = load_random_data(img_folder=IMG_DIR, json_folder=JSON_DIR)
            self.file_base = img_name
            self.coords = [make_coord_key(coord) for coord in coords]   # str for tracking if coord should be visited
            self.original_img = cur_img
            self.idx = rand_idx     # for recoving and plotting
        else:
            self.file_base = FILE_BASE
            observation, coords = load_img(FILE_BASE, better_coords=True)
            self.coords = [make_coord_key(coord) for coord in coords]   # str for tracking if coord should be visited
            self.original_img = observation     # unadulterated image
            self.idx = 0

        init_coords = [self.coords[0]]*STATE_LENGTH  # str - just first point to avoid confusing reward and action coupling
        self.last_true_coord = init_coords[-1]  # as target for getting back on track
        self.visited_coords = init_coords  # str for easy comparison
        self.actions = [0] * len(init_coords)
        self.rewards = [0] * len(init_coords)
        self.state = make_state(self.original_img, visited_coords=self.visited_coords,
                                last_true_coord=self.last_true_coord, true_coords=self.coords)
        return self.state[:, :, -1]

    def step(self, action, visualise=False):
        observation, reward, terminal = self.calc_observation_reward(action)
        # update self.state
        self.state = np.append(self.state[:, :, 1:], observation.reshape(FRAME_HEIGHT, FRAME_WIDTH, 1), axis=2)
        self.rewards.append(reward)
        self.actions.append(action)

        if visualise:
            visualise_state(self)

        return observation, reward, terminal, {}

    def _is_terminal(self):
        true_coords = set(self.coords)
        common_points = set(self.visited_coords).intersection(true_coords)
        if len(common_points)==len(true_coords) or (EPISODE_LIMIT and len(self.visited_coords) > EPISODE_LIMIT):    # early stopping condition
            return True
        else:
            return False

    def render(self, save_path=None):
        if save_path:
            plt.switch_backend('agg')

        visited_coords = np.asarray([parse_coord_str(x) for x in self.visited_coords])
        plt.figure(1)
        plt.clf()
        plt.imshow(self.original_img)  # since static is this and for new relative images coordinates are to this
        plt.scatter(y=visited_coords[:, 1], x=visited_coords[:, 0], c='red', s=2)
        # traced_coords = [parse_coord_str(x) for x in self.coords]
        # traced_coords_real = np.asarray(traced_coords)
        # plt.scatter(x=traced_coords_real[:,0], y=traced_coords_real[:,1], c='white', s=2)
        plt.show(block=False)
        if save_path:
            plt.savefig(save_path)

        visualise_state(self)
        return

    # class utils
    def calc_observation_reward(self, action):
        new_observation, new_coord, terminal, reward = self.action_to_coord(action)
        # reward = self.calc_reward(new_coord)
        return new_observation, reward, terminal

    # if it crosses all coords and has not been visited
    def calc_reward(self, new_coord):
        coord_key = make_coord_key(new_coord)
        if coord_key in self.visited_coords[:-1]:   # up to current coord
            reward = -0.1   # small penalty for faffing around
        else:
            # always get next true
            x, y = get_coord_to_add(self.coords, self.last_true_coord, next_true=(CHEAT_PEEK or REWARD_ONLY_NEXT_TRUE))
            target_dist = np.sqrt(np.sum((np.asarray([x, y]) - np.asarray(new_coord)) ** 2))
            reward = -target_dist   # reward is divergence from target
            # if ADJUST_LOSS:  # large losses if further away
            #     reward = reward / float(max(FRAME_HEIGHT, FRAME_WIDTH))

            if REWARD_ONLY_NEXT_TRUE:
                if reward==0:
                    # reward = 0
                    self.last_true_coord = coord_key  # for tracking purposes
            else:
                if coord_key in self.coords:
                    reward = 0
                    self.last_true_coord = coord_key    # for tracking purposes
        return reward

    def coord_min_dist(self, new_coord):
        min_dist, closest_coord = coord_min_dist(self.coords, new_coord)
        return min_dist

    def action_to_coord(self, action, visualise=False):  # 1- hot vector to direction
        cur_state = self.state[:, :, -1]    # last state is current state
        cur_coord_str = self.visited_coords[-1]
        cur_coord = parse_coord_str(cur_coord_str)    # last coord is current coord

        box = action_to_direction_box(action)   # map action to coord system
        new_point = np.transpose(np.nonzero(box))
        w_diff = new_point[0][1] - 1    # col diff is x/width change
        h_diff = new_point[0][0] - 1    # row diff is y/height change

        new_x = cur_coord[0] + w_diff
        new_y = cur_coord[1] + h_diff
        new_coord = (new_x, new_y)
        str_new_coord = make_coord_key(new_coord)

        is_bounded = within_img_bounds(new_coord)
        terminal = False
        if is_bounded:  # if move is valid (stays within img bounds)
            new_observation = get_img_and_loc(self.original_img, self.visited_coords, self.last_true_coord, self.coords)
            terminal = self._is_terminal()  # check all labeled visited
            self.visited_coords.append(str_new_coord)   # needs new coord for making state images
        else:
            new_observation = np.copy(cur_state)  # in case pointer problems
            terminal = True     # also terminal if out of bounds
            if EDGE_CHECK:
                terminal = False  # keep playing but dont actually take step - artificially extend game
                new_coord = cur_coord
                self.visited_coords.append(cur_coord_str)  # needs new coord for making state images
            else:
                self.visited_coords.append(str_new_coord)
                if NO_EDGE:  # edge cliff penalty - probably too sharp
                    reward = -10 ** 10
        reward = self.calc_reward(new_coord)

        # adjust for different edge cases
        edge_factor = 1
        if EDGE_PENALTY:    # don't use edge check with this
            h_margin = int(FRAME_HEIGHT / 2.)
            w_margin = int(FRAME_WIDTH / 2.)
            # 1.2 for more curvature; 2 gets quite flat. 1 is too linear
            temp = min(abs(new_x - w_margin + .1), abs(RAW_WIDTH - new_x + .1))**1.2 \
                   + min(abs(new_y - h_margin + .1), abs(RAW_HEIGHT-new_y + .1))**1.2
            edge_factor += 1/temp
            reward *= edge_factor

        intensity_factor = 1
        if INTENSITY_MULTIPLIER:
            cur_intensity_mean = np.mean(cur_state)
            cur_intensity_std = np.std(cur_state)
            all_mean = np.mean(self.original_img)
            all_std = np.std(self.original_img)
            # intensity_factor = (cur_intensity_mean/cur_intensity_std)/(all_mean/all_std)
            intensity_factor = (cur_intensity_std/all_std)
            reward /= intensity_factor  # more different is good

        if visualise:
            box_orig = np.copy(box)
            box_orig[1, 1] = 1  # center point
            plt.imshow(box_orig)  # careful of meaning of x,y
            plt.scatter(y=h_diff + 1, x=w_diff + 1, c='blue', s=10)
        return new_observation, new_coord, terminal, reward

    def populate_replay_memory(self, agent, mode=1, visualise=False):  # mode=1 for true human path
        terminal = False

        counter = 0
        total_reward = 0
        while not terminal:
            action = get_action(mode, self)
            state = np.copy(self.state)
            observation, reward, terminal, _ = self.step(action)
            # next_state = np.append(self.state[:, :, 1:], observation.reshape(FRAME_HEIGHT, FRAME_WIDTH, 1), axis=2)
            next_state = self.state
            agent.replay_memory.append((state, action, reward, next_state, terminal))
            total_reward += reward
            counter += 1

        if visualise:
            visited_coords = np.asarray([parse_coord_str(x) for x in self.visited_coords])
            plt.imshow(self.original_img)
            plt.scatter(x=visited_coords[:, 0], y=visited_coords[:, 1], c='red', s=1)
        return agent

### general utils
def make_coord_key(coord):
    key_str = '{}_{}'.format(coord[0], coord[1])
    return key_str


def parse_coord_str(coord_str):
    x, y = coord_str.split('_')
    return int(x), int(y)


def get_img_and_loc(img, visited_coords, last_true_coord, true_coords, include_history=True, visualise=False):
    img = make_coord_img(img, visited_coords=visited_coords, last_true_coord=last_true_coord, true_coords=true_coords,
                         include_history=include_history)
    if visualise:
        plt.figure(1)
        plt.imshow(img)
        coord = visited_coords[0]   # only use current coord for now
        plt.scatter(x=coord[0], y=coord[1], c='red', s=10)
    return img


def make_coord_img(img, visited_coords, last_true_coord, true_coords, add_last_true=ADD_LAST, include_history=False, visualise=False):
    # Idea trace path of visited coords and link to rewards for that path
    num_mem = STATE_LENGTH if include_history else 1

    adjust_factor = 2 if add_last_true else 1
    new_img = np.copy(img)
    for idx, coord in enumerate(visited_coords[-1:-num_mem-1:-1]):   # from last (most recent) to first
        x, y = parse_coord_str(coord)   # careful about meaning of x, y
        # max intensity and linear scale downwards (128-255)
        new_img[y, x] = (1-idx/(STATE_LENGTH+1.)) * 127/adjust_factor + 128

    if add_last_true:
        x, y = get_coord_to_add(true_coords, last_true_coord)
        new_img[y, x] = 255

    # centralize
    if DO_CENTRALIZE:
        cur_coord = parse_coord_str(visited_coords[-1])     # last visited coord
        x, y = cur_coord
        lower_y = int(y-FRAME_HEIGHT/2.)
        upper_y = int(y+FRAME_HEIGHT/2.)
        lower_x = int(x-FRAME_WIDTH/2.)
        upper_x = int(x+FRAME_WIDTH/2.)
        new_img = new_img[lower_y:upper_y, lower_x:upper_x]

    if visualise:
        changed_coords = np.asarray([parse_coord_str(x) for x in visited_coords[-1:-num_mem-1:-1]])
        if DO_CENTRALIZE:
            plt.figure(1); plt.clf()
            plt.imshow(new_img)
            plt.scatter(x=changed_coords[:, 0]-lower_x, y=changed_coords[:, 1]-lower_y, c='red', s=1)
            fig1, ax1 = plt.subplots(1)
            ax1.imshow(img)
            ax1.add_patch(patches.Rectangle((lower_x, lower_y), upper_x - lower_x, upper_y - lower_y, fill=False, color='white', linewidth=1))
        else:
            plt.figure(1); plt.clf()
            plt.imshow(new_img)
            plt.scatter(x=changed_coords[:, 0], y=changed_coords[:, 1], c='red', s=1)

    return new_img


def get_coord_to_add(true_coords, last_true_coord, next_true=CHEAT_PEEK):
    if next_true:
        last_index = true_coords.index(last_true_coord)  # finds first case of true coord assuming ordering consistency
        if last_index < len(true_coords):
            peek_coord = true_coords[last_index + 1]
        else:
            peek_coord = true_coords[0]
        x, y = parse_coord_str(peek_coord)
    else:
        x, y = parse_coord_str(last_true_coord)
    return x, y


def load_random_data(img_folder, json_folder, nick_only=True, visualise=False):
    imgs, img_names, coords = load_data(img_folder, json_folder)

    if nick_only:
        nick_idx = [idx for idx, img_name in enumerate(img_names) if 'DeRuyter' in img_name]
        rand_idx = random.sample(nick_idx, 1)[0]
    else:
        rand_idx = random.randint(0, len(img_names) - 1)  # random img and trace

    # coords_np = np.transpose(np.asarray())
    cur_coords = coords[rand_idx]
    xs = cur_coords[0]
    ys = cur_coords[1]
    coords_np = np.ndarray(shape=(len(xs), 2), dtype=np.int64)
    coords_np[:, 0] = xs
    coords_np[:, 1] = ys

    cur_img = imgs[rand_idx, ]
    cur_img_name = img_names[rand_idx]

    if visualise:
        plt.imshow(cur_img)
        plt.title(cur_img_name)
        plt.scatter(x=coords_np[:, 0], y=coords_np[:, 1], c='red', s=2)
    return cur_img, cur_img_name, coords_np, rand_idx


def load_data(img_folder, json_folder):
    imgs, img_names = load_imgs(img_folder)
    coords = load_coords(json_folder, img_names)
    return imgs, img_names, coords


def load_coords(json_folder, img_names):
    coords_npy = '{}/acseg_coords_ql.npy'.format(LOG_FOLDER)

    if not os.path.isfile(coords_npy):
        coords = [] # variable length array
        for idx, img_name in enumerate(img_names):
            coord_path = '{}/{}'.format(JSON_DIR, img_name.replace('.png', '.json'))
            fin = open(coord_path).read()
            json_data = json.loads(fin)
            all_xs, all_ys = get_coords(json_data)
            xs, ys = remove_duplicates(all_xs, all_ys)
            coords.append((xs, ys))
        np.save(coords_npy, np.asarray(coords))
    else:
        coords = np.load(coords_npy)
    return coords


def load_imgs(img_folder):
    img_names = [x for x in sorted(os.listdir(img_folder)) if '.png' in x.lower() and 'mask' not in x.lower()]

    imgs_npy = '{}/acseg_ql.npy'.format(LOG_FOLDER)
    if not os.path.isfile(imgs_npy):
        imgs = np.ndarray((len(img_names), RAW_HEIGHT, RAW_WIDTH), dtype=np.float32)
        for idx, img_name in enumerate(img_names):
            img_path = os.path.join(img_folder, img_name)
            cur_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            cur_img = cur_img / 255 * 127  # cap 0-127 for img and reserve 128-255 for state
            imgs[idx, ] = cur_img
        np.save(imgs_npy, imgs)
    else:
        imgs = np.load(imgs_npy)

    return imgs, img_names


def load_img(file_base=FILE_BASE, better_coords=False, visualise=False):
    image_path = '{}/{}.png'.format(IMG_DIR, file_base)
    coord_path = '{}/{}.json'.format(JSON_DIR, file_base)

    image_npy = '{}/acseg_ql_{}.npy'.format(LOG_FOLDER, file_base)
    if not os.path.isfile(image_npy):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img/255 * 127     # cap 0-127 for img and reserve 128-255 for state
        np.save(image_npy, img)
    else:
        img = np.load(image_npy)

    coords_npy = '{}/acseg_coords_ql_{}.npy'.format(LOG_FOLDER, file_base)
    if not os.path.isfile(coords_npy):
        fin = open(coord_path).read()
        json_data = json.loads(fin)
        all_xs, all_ys = get_coords(json_data)
        xs, ys = remove_duplicates(all_xs, all_ys)
        coords = np.ndarray(shape=(len(xs), 2), dtype=np.int64)
        coords[:, 0] = xs
        coords[:, 1] = ys
        np.save(coords_npy, coords)
    else:
        coords = np.load(coords_npy)

    if better_coords:
        # # test points for get_contrasted_coord
        # get_contrasted_coord(img, (250, 249), [])
        # get_contrasted_coord(img, (276, 243), [])
        # get_contrasted_coord(img, (277, 243), [])
        # get_contrasted_coord(img, (278, 243), [])

        coords_npy = '{}/acseg_coords_ql_{}_clean.npy'.format(LOG_FOLDER, file_base)
        if os.path.isfile(coords_npy):
            coords = np.load(coords_npy)
        else:
            # coords = contrasted_chamber_coords(img, coords)
            new_coords = []
            for idx, coord in enumerate(coords):
                new_coord = get_contrasted_coord(img, coord, new_coords, max_steps=3, do_max=True)
                new_coords.append(new_coord)
            orig_coords = coords
            coords = remove_dupe_coords(np.asarray(new_coords))
            # np.save(coords_npy, coords)
            contrast_file = '{}_tighter.txt'.format(file_base)
            np.savetxt(contrast_file, coords, '%d')

            # manually remove some coords
            manual_file = '{}_tighter_manual.txt'.format(file_base)
            coords2 = np.loadtxt(manual_file, dtype=np.int16)

        # now interpolate coords
        p_coords = coords
        if 'coords2' in locals():
            p_coords = coords2
        new_coords = interpolate_coords(img, p_coords, visualise=visualise)

        if visualise:
            plt.imshow(img)
            plt.scatter(x=orig_coords[:, 0], y=orig_coords[:, 1], c='red', s=3)
            plt.scatter(x=p_coords[:, 0], y=p_coords[:, 1], c='blue', s=2)
            plt.scatter(x=new_coords[:, 0], y=new_coords[:, 1], c='green', s=1)
        coords = new_coords

    return img, coords


def contrasted_chamber_coords(img, coords, visualise=False):
    new_coords = [(0, 0)]
    # for each coord - get local high contrast coords
    for idx, coord in enumerate(coords):
        cur_new = local_contrast(img, coord, new_coords)
        new_coords += cur_new

    if visualise:
        plt.imshow(img)
        plt.scatter(x=coords[:, 0], y=coords[:, 1], c='red', s=1)
        new_coords2 = set(new_coords)
        new_coords = np.asarray(new_coords)
        plt.scatter(x=new_coords[:, 0], y=new_coords[:, 1], c='blue', s=2)
    return new_coords[1:, ]


# maintain uniqueness using prev_coords
def local_contrast(img, coord, prev_coords, visualise=False):
    x, y = coord
    max_contrast = 0
    max_contrast_coord = coord
    all_local_coords = []

    local_size = 1
    counter = 0
    for dy in range(-local_size, local_size, 1):
        for dx in range(-local_size, local_size, 1):
            cur_x = x + dx
            cur_y = y + dy
            cur_point = img[cur_y, cur_x]
            cur_patch = img[(cur_y-1):(cur_y+2), (cur_x-1):(cur_x+2)]
            cur_diffs = abs(cur_patch/cur_point - 1)
            i, j = get_coord_from_img(cur_diffs)
            local_max_contrast = cur_diffs[i, j]

            if local_max_contrast>max_contrast:
                max_contrast = local_max_contrast
                max_contrast_coord = cur_x+j-1, cur_y+i-1
                all_local_coords.append(max_contrast_coord)

    # if max_contrast_coord in prev_coords:   # avoid repetition
    #     max_contrast_coord = []

    if visualise:
        plt.imshow(img)
        plt.scatter(x=x, y=y, c='red', s=2)
        # plt.scatter(x=cur_x, y=cur_y, c='blue', s=2)
        plt.scatter(x=max_contrast_coord[0], y=max_contrast_coord[1], c='blue', s=2)
        all_local_coords = np.asarray(all_local_coords)
        plt.scatter(x=all_local_coords[:, 0], y=all_local_coords[:, 1], c='green', s=2)
    return [max_contrast_coord]


def _compute_coord_contrast(img, coord1, coord2, is_abs=True):
    x1, y1 = coord1
    x2, y2 = coord2
    cur_intensity = img[y1, x1]
    new_intensity = img[y2, x2]
    perc_diff = cur_intensity / new_intensity - 1
    if is_abs:
        perc_diff = abs(perc_diff)
    return perc_diff


# assumes completed boundaries that can only be tightened
def _contrasted_coord_recursive(img, start_coord, previous_coords, direction, cur_coord, step_num=0, best_contrast=0,
                                best_coord=None, max_steps=3):
    x, y = cur_coord
    dx = 0
    dy = 0
    if direction==0:
        dy = -1
    elif direction==1:
        dy = 1
    elif direction==2:
        dx = -1
    elif direction==3:
        dx = 1
    new_x = x + dx
    new_y = y + dy
    new_coord = np.asarray([new_x, new_y])  # enforce ndarray for stopping condition calculation purposes

    if step_num==0:     # get reverse direction contrast
        prev_x = x - dx
        prev_y = y - dy
        best_contrast = _compute_coord_contrast(img, cur_coord, (prev_x, prev_y), is_abs=True)  # want less intense

    # intuition get maximum absolute contrast where new_coord is less intense than original/start coord
    new_contrast = _compute_coord_contrast(img, cur_coord, new_coord, is_abs=True)
    # if new_contrast > best_contrast:
    if new_contrast > best_contrast and img[start_coord[1], start_coord[0]] > img[cur_coord[1], cur_coord[0]]:
        best_contrast = new_contrast
        best_coord = cur_coord    # want the edge coord instead going over
    else:
        if step_num==0:
            best_coord = cur_coord
        else:   # keep previous best coord
            1

    step_num +=1
    if _stop_condition(img, new_coord, previous_coords, step_num, max_steps, do_smooth=False):   # 3 steps in each direction
        return _contrasted_coord_recursive(img, start_coord, previous_coords, direction, new_coord, step_num, best_contrast,
                                           best_coord=best_coord, max_steps=max_steps)
    else:
        return best_coord, best_contrast


def _stop_condition(img, new_coord, previous_coords, step_num, max_steps=3, do_smooth=True):
    smooth_condition = True
    if do_smooth:
        cur_coord_dist = 0
        if len(previous_coords) > 0:
            last_coord = previous_coords[-1]
            cur_coord_dist = _dist(new_coord, last_coord)
        if cur_coord_dist <= np.sqrt(2) or np.any(abs(new_coord - last_coord) > 1):
            smooth_condition = True
        else:
            smooth_condition = False

    if step_num < max_steps and smooth_condition:
        return True
    else:
        return False


def _dist(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def remove_dupe_coords(all_coords):
    unique_coords = []
    for idx, coord in enumerate(all_coords):
        if tuple(coord) not in unique_coords:
            unique_coords.append(tuple(coord))
    return np.asarray(unique_coords)


def interpolate_coords(img, coords, visualise=False):
    new_coords = []
    coord_dict = {}
    for idx, cur_coord in enumerate(coords):
        cur_coord = tuple(cur_coord)
        if len(new_coords) > 0:
            prev_coord = new_coords[-1]
        else:
            prev_coord = cur_coord
        cur_coord_dist = _dist(cur_coord, prev_coord)

        if cur_coord_dist > np.sqrt(2):  # or np.any(abs(np.asarray(cur_coord) - np.asarray(prev_coord)) > 1):     # missing coord
            cur_x, cur_y = cur_coord
            init_coord = prev_coord
            while prev_coord!= cur_coord:
                prev_x, prev_y = prev_coord
                dy = np.sign(cur_y - prev_y)
                dx = np.sign(cur_x - prev_x)

                # start with prev_coord and move to cur_coord
                coord1 = prev_x, prev_y+dy
                coord2 = prev_x+dx, prev_y
                coord3 = prev_x+dx, prev_y+dy
                candidate_coords = [coord1, coord2, coord3]
                if cur_coord in candidate_coords:
                    break   # just add candidate coords

                valid_coords = []
                coord_dists = []
                intensities = []
                for candidate_coord in candidate_coords:
                    if candidate_coord != prev_coord:   # and candidate_coord not in new_coords:  # avoid duplicate paths
                        valid_coords.append(candidate_coord)
                        coord_dists.append(_dist(candidate_coord, cur_coord))
                        cur_contrast = _compute_coord_contrast(img, prev_coord, candidate_coord, is_abs=True)
                        intensities.append(cur_contrast)
                # enforce shortest distance and then intensity
                min_dist_idx = np.where(coord_dists==np.min(coord_dists))
                if len(min_dist_idx)==1:
                    prev_coord = valid_coords[min_dist_idx[0][0]]
                else:
                    dir_idx = np.argmin(intensities)    # follow coords most like prev_coord
                    prev_coord = valid_coords[dir_idx]
                new_coords.append(prev_coord)

                coord_str = make_coord_key(cur_coord)
                if coord_str in coord_dict:
                    coord_dict[coord_str].append(prev_coord)
                else:
                    coord_dict[coord_str] = [init_coord, prev_coord]
        if cur_coord not in new_coords:
            new_coords.append(cur_coord)

    new_coords = np.asarray(new_coords)
    if visualise:
        plt.imshow(img)
        plt.scatter(x=new_coords[:, 0], y=new_coords[:, 1], c='blue', s=1)
        coords = np.asarray(coords)
        plt.scatter(x=coords[:, 0], y=coords[:, 1], c='red', s=1)
    return new_coords


def get_contrasted_coord(img, start_coord, prev_coords, max_steps=3, do_max=True):
    dir_coords = []
    dir_contrasts = []

    for direction in range(4):
        dir_coord, dir_contrast = _contrasted_coord_recursive(img, start_coord, prev_coords, direction,
                                                              cur_coord=start_coord, step_num=0, max_steps=max_steps)
        dir_coords.append(dir_coord)
        dir_contrasts.append(dir_contrast)
    if do_max:
        max_idx = np.argmax(dir_contrasts)
    else:
        max_idx = np.argmin(dir_contrasts)
    return dir_coords[max_idx]


def naive_tracer(img, start_coord, visualise=False):
    all_coords = []
    new_coord = start_coord
    counter = 0
    while new_coord not in all_coords:
        new_coord = get_contrasted_coord(img, new_coord)    # adjust for intensity difference
        all_coords.append(new_coord)
        # direction with least constrast
        new_coord = get_contrasted_coord(img, start_coord, max_steps=1, do_max=False)
        counter+=1
        if visualise and counter % 10:
            plt.imshow(img)
            traced_coords = np.asarray(all_coords)
            plt.scatter(x=all_coords[:, 0], y=all_coords[:, 1], c='red', s=1)
    return all_coords


# def get_coord_from_mask(mask):
#     cur_coord = np.transpose(np.nonzero(mask))
#     return cur_coord
# obsolete
def get_coord_from_img(img):
    # cur_coord = img.argmax(axis=0)
    i, j = np.unravel_index(img.argmax(), img.shape)
    return i, j


def within_img_bounds(coord):
    x, y = coord
    if DO_CENTRALIZE:
        h_margin = int(FRAME_HEIGHT/2.)
        w_margin = int(FRAME_WIDTH/2.)
    if (x < 0+w_margin or x >= RAW_WIDTH-w_margin) or (y < 0+h_margin or y >= RAW_HEIGHT-h_margin):
        return False
    else:
        return True


def make_state(observation, visited_coords, last_true_coord, true_coords):
    processed_observation = get_img_and_loc(observation, visited_coords, last_true_coord, true_coords)
    state = [processed_observation for _ in range(STATE_LENGTH)]
    return np.stack(state, axis=2)


def visualise_state(env):
    state_shape = env.state.shape
    for idx in [STATE_LENGTH-1]:
        plt.figure(100+idx)
        plt.imshow(env.state[:, :, idx])

        plt.figure(1)
        plt.imshow(env.original_img)
        state_coord = parse_coord_str(env.visited_coords[-(STATE_LENGTH-idx)])
        plt.scatter(x=state_coord[0], y=state_coord[1], c='red', s=2)
    return


# 3 options for testing rewards and rendering
# 0 - random walk; 1 - truth; 2 - truth + noise; 3 - random direction
def playthrough(mode=0):
    env = ChamberTracer()
    terminal = False
    observation = env.reset()  # (single) observation includes location info
    state = env.state  # compartmentalize make_state (initial)

    counter = 0
    total_reward = 0
    while not terminal:
        action = get_action(mode, env)
        observation, reward, terminal, _ = env.step(action)
        total_reward += reward

        if counter % 100==0:
            env.render()
            print('t={}'.format(counter), '#visited={}'.format(len(env.visited_coords)),
                  'reward={}'.format(total_reward), 'terminal={}'.format(terminal))
        counter += 1
    return


def coord_min_dist(true_coords, new_coord):
    true_coords = np.asarray([parse_coord_str(x) for x in true_coords])
    new_coord = np.asarray(new_coord)
    all_dist = np.sum((true_coords - new_coord)**2, axis=1)
    min_dist = np.min(all_dist)
    return min_dist, true_coords[np.argmin(all_dist)]


def action_to_direction_box(action):
    pred = np.zeros(shape=(NUM_ACTIONS, 1), dtype=np.int64)
    pred[action] = 1
    if NUM_ACTIONS==9:  # allows 4 for no action
        1
    elif NUM_ACTIONS==8:   # no faffing allowed
        pred = np.insert(pred, 4, 0)  # insert no action at center
    box = pred.reshape((3, 3))
    return box


def coord_direction(coord_start, coord_end):
    (x_start, y_start) = coord_start
    (x_end, y_end) = coord_end

    y_diff = y_end - y_start
    x_diff = x_end - x_start
    row_ind = np.sign(y_diff) + 1
    col_ind = np.sign(x_diff) + 1

    box = np.zeros((3, 3), dtype=np.uint8)
    box[row_ind, col_ind] = 1

    # account for 0 = no action
    flat_box = box.flatten()
    action = np.nonzero(flat_box)[0][0] # only non-zero element
    if NUM_ACTIONS==8:  # if no faffing allowed
        action = action if action < 4 else action-1     # 4 implicitly will not happen
    return action


def get_action(mode, env):
    if mode==0:
        action = random.randrange(NUM_ACTIONS)
    elif mode==1:
        true_unvisited = list(set(env.coords) - set(env.visited_coords))
        min_dist, closest_unvisited = coord_min_dist(true_unvisited, parse_coord_str(env.last_true_coord))
        action = coord_direction(parse_coord_str(env.visited_coords[-1]), closest_unvisited)
    elif mode==2:
        k = 3   # 2 steps true - 1 step random
        r = random.randrange(k)
        if r < k-1:
            true_unvisited = list(set(env.coords) - set(env.visited_coords))
            min_dist, closest_unvisited = coord_min_dist(true_unvisited, parse_coord_str(env.last_true_coord))
            action = coord_direction(parse_coord_str(env.visited_coords[-1]), closest_unvisited)
        else:
            action = random.randrange(NUM_ACTIONS)
    elif mode==3:   # random direction
        action = 1  # fix direction for the moment

    return action


def generate_trace_path():
    img, coords = load_img()
    for idx in range(2, len(coords), 2):
        plt.imshow(img)
        plt.scatter(x=coords[:idx, 0], y=coords[:idx, 1], c='red', s=2)
        plt.savefig('{}_{}.png'.format(FILE_BASE, idx))
    return


if __name__ == '__main__':
    # playthrough(0)    # random
    # playthrough(1)  # truth
    # playthrough(2)    # mostly truth
    # generate_trace_path()

    # check coordinate fix
    # load_img(FILE_BASE, better_coords=True)

    load_data(img_folder=IMG_DIR, json_folder=JSON_DIR)
    cur_img, img_name, coords, rand_idx = load_random_data(img_folder=IMG_DIR, json_folder=JSON_DIR)
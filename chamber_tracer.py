import logging
import math
# import gym
# from gym import spaces
# from gym.utils import seeding
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches

from qlearning_data import get_coords, remove_duplicates

logger = logging.getLogger(__name__)

# current image - picked
FILE_BASE = 'DeRuyter-Inflamed_20170703mouse6_Day2_Right_160'
LOG_FOLDER = './dqn_log'
from sys import platform
if platform == "linux" or platform == "linux2":
    IMG_DIR = '/data/pepple/acseg/segmentations'
    JSON_DIR = '/data/pepple/acseg/jsons'
elif platform == "win32":
    IMG_DIR = './acseg/segmentations'
    JSON_DIR = './acseg/jsons'

# environment settings
RAW_HEIGHT = 512
RAW_WIDTH = 500
DO_CENTRALIZE = True
if DO_CENTRALIZE:
    FRAME_HEIGHT = 128
    FRAME_WIDTH = 128
else:
    FRAME_HEIGHT = 512  # Resized frame width
    FRAME_WIDTH = 500  # Resized frame height

# NUM_CHANNELS = 2    # observation and location
NUM_ACTIONS = 9     # directions

# storage settings
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network


class ChamberTracer():
    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second' : 50
    # }

    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
    def __init__(self, file_base=FILE_BASE):
        self.file_base = file_base
        observation, coords = load_img(file_base)
        self.coords = [make_coord_key(coord) for coord in coords]   # str for tracking if coord should be visited
        self.original_img = observation     # unadulterated image
        self.reset()

    def reset(self):  # reset original observation and starting coord
        init_coords = self.coords[0:STATE_LENGTH]  # str
        self.last_true_coord = init_coords[-1]  # for making images later
        self.visited_coords = init_coords  # str for easy comparison
        self.actions = [0] * len(init_coords)
        self.rewards = [0] * len(init_coords)
        self.state = make_state(self.original_img, visited_coords=self.visited_coords,
                                last_true_coord=self.last_true_coord,
                                true_coords=self.coords)
        return self.state[:, :, -1]

    def step(self, action, visualise=False):
        observation, reward, terminal = self.calc_observation_reward(action)
        # update self.state
        self.state = np.append(self.state[:, :, 1:], observation.reshape(FRAME_HEIGHT, FRAME_WIDTH, 1), axis=2)
        self.rewards.append(reward)
        self.actions.append(action)

        if visualise:
            visualise_state(observation)
            for idx in range(STATE_LENGTH):
                visualise_state(self.state[:, :, idx])

        return observation, reward, terminal, {}

    def _is_terminal(self):
        user_coords = set(self.coords)
        common_points = set(self.visited_coords).intersection(user_coords )
        if len(common_points)==len(user_coords):
            return True
        else:
            return False

    def render(self):
        all_coords = np.asarray([parse_coord_str(x) for x in self.visited_coords])
        plt.figure(1)
        plt.clf()
        plt.imshow(self.state[:, :, -1])     # image
        plt.scatter(y=all_coords[:, 1], x=all_coords[:, 0], c='red', s=2)
        # traced_coords = [parse_coord_str(x) for x in self.coords]
        # traced_coords_real = np.asarray(traced_coords)
        # plt.scatter(x=traced_coords_real[:,0], y=traced_coords_real[:,1], c='red', s=2)
        plt.show(block=False)
        return

    # class utils
    def calc_observation_reward(self, action):
        cur_state = self.state[:, :, -1]  # last state is current state
        new_observation, new_coord, terminal = self.action_to_coord(action, cur_state)
        reward = self.calc_reward(new_coord)

        return new_observation, reward, terminal

    # if it crosses all coords and has not been visited
    def calc_reward(self, new_coord):
        coord_key = make_coord_key(new_coord)
        if coord_key in self.visited_coords[:-1]:   # up to current coord
            reward = 0
        else:
            if make_coord_key(new_coord) in self.coords:
                reward = 1
                self.last_true_coord = new_coord
            else:
                # reward = 0
                reward = -self.coord_min_dist(new_coord)  # have negative rewards for divergence
        return reward

    def coord_min_dist(self, new_coord):
        true_coords = np.asarray([parse_coord_str(x) for x in self.coords])
        all_dist = np.sum((true_coords-new_coord)**2, axis=1)
        return np.min(all_dist)

    def action_to_coord(self, action, cur_state, visualise=False):  # 1- hot vector to direction
        cur_coord = parse_coord_str(self.visited_coords[-1])
        # map action to coord system
        pred = np.zeros(shape=(NUM_ACTIONS, 1), dtype=np.int64)
        # zero is do nothing - allow staying in 1 place
        if action==0:
            pred[4] = 1  # middle is do nothing
        else:
            pred[action-1] = 1  # 1-shifted because of zero
        box = pred.reshape((3, 3))
        # pred[action] = 1  # dont allow no action
        # pred2 = np.insert(pred, 4, 0)  # insert current position at center
        # box = pred2.reshape((3, 3))

        new_point = np.transpose(np.nonzero(box))
        x_diff = new_point[0][0] - 1
        y_diff = new_point[0][1] - 1

        new_x = cur_coord[0] + x_diff
        new_y = cur_coord[1] + y_diff
        new_coord = (new_x, new_y)

        terminal = False
        str_new_coord = make_coord_key(new_coord)
        self.visited_coords.append(str_new_coord)   # needs new coord for making state images
        if within_img_bounds(new_coord):
            if str_new_coord in self.coords:
                self.last_true_coord = str_new_coord
            new_observation = get_img_and_loc(self.original_img, self.visited_coords, self.last_true_coord, self.coords)
            terminal = self._is_terminal()  # check visited vs labeled
        else:
            new_observation = np.copy(cur_state)    # in case pointer problems
            terminal = True

        if visualise:
            box_orig = np.copy(box)
            box_orig[1, 1] = 1  # center point
            plt.imshow(box_orig)  # careful of meaning of x,y
            plt.scatter(y=x_diff + 1, x=y_diff + 1, c='blue', s=10)
        return new_observation, new_coord, terminal


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
    # img_shape = img.shape
    # cur_img_loc = np.ndarray(shape=(img_shape[0], img_shape[1], 2), dtype=np.float32)  # 2 channels
    # cur_img_loc[:, :, 0] = img
    # coord_mask = make_coord_mask(img_shape, coords, include_history)
    # cur_img_loc[:, :, 1] = coord_mask

    if visualise:
        plt.figure(1)
        plt.imshow(img)
        coord = visited_coords[0]   # only use current coord for now
        plt.scatter(x=coord[0], y=coord[1], c='red', s=10)
        # plt.figure(2)
        # plt.imshow(coord_mask)
        # plt.show()
    return img


# def make_coord_img(img, coords, include_history=False):
def make_coord_img(img, visited_coords, last_true_coord, true_coords, include_history=False, visualise=False):
    # Idea trace path of visited coords and link to rewards for that path
    num_mem = STATE_LENGTH if include_history else 1

    new_img = np.copy(img)
    for idx, coord in enumerate(visited_coords[-1:-num_mem-1:-1]):   # from last (most recent) to first
        # euclidean_dist = np.sqrt((cur_x - x) **2 + (cur_y - y)**2)
        # max(0, (1 - euclidean_dist * .05)) * 255
        x, y = parse_coord_str(coord)   # careful about meaning of x, y
        new_img[y, x] = (1-idx/(STATE_LENGTH+1.)) * 127 + 128  # max intensity and linear scale downwards (128-255)

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


def load_img(file_base=FILE_BASE):
    image_path = '{}/{}.png'.format(IMG_DIR, file_base)
    coord_path = '{}/{}.json'.format(JSON_DIR, file_base)

    image_npy = '{}/acseg_ql_{}.npy'.format(LOG_FOLDER, file_base)
    coords_npy = '{}/acseg_coords_ql_{}.npy'.format(LOG_FOLDER, file_base)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # cap 0-127 for img and reserve 128-255 for state
    img = img/255 * 127
    # img.astype(np.uint8)
    np.save(image_npy, img)

    fin = open(coord_path).read()
    json_data = json.loads(fin)
    all_xs, all_ys = get_coords(json_data)
    xs, ys = remove_duplicates(all_xs, all_ys)
    coords = np.ndarray(shape=(len(xs), 2), dtype=np.int64)
    coords[:, 0] = xs
    coords[:, 1] = ys
    np.save(coords_npy, coords)
    return img, coords


def load_data(file_base=FILE_BASE):
    image_npy = '{}/acseg_ql_{}.npy'.format(LOG_FOLDER, file_base)
    coords_npy = '{}/acseg_coords_ql_{}.npy'.format(LOG_FOLDER, file_base)
    img = np.load(image_npy)
    coords = np.load(coords_npy)
    return img, coords


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


def visualise_state(state):
    plt.figure()
    plt.imshow(state)  # careful of meaning of x,y
    if not DO_CENTRALIZE:
        cur_coord = get_coord_from_img(state)
        plt.scatter(y=cur_coord[1], x=cur_coord[0], c='blue', s=10)
    return
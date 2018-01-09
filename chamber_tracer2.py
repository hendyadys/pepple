import logging, math, random, os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches

from qlearning_data import get_coords, remove_duplicates
import dqn_chamber2

logger = logging.getLogger(__name__)

# current image - picked
FILE_BASE = 'DeRuyter-Inflamed_20170703mouse6_Day2_Right_160'
from sys import platform
if platform == "linux" or platform == "linux2":
    # IMG_DIR = '/data/pepple/acseg/segmentations'
    # JSON_DIR = '/data/pepple/acseg/jsons'
    # LOG_FOLDER = '/home/yue/pepple/dqn_log'
    IMG_DIR = '/data/yue/pepple_dqn/acseg/segmentations'
    JSON_DIR = '/data/yue/pepple_dqn/acseg/jsons'
    LOG_FOLDER = '/data/yue/pepple_dqn/dqn_log'
elif platform == "win32":
    IMG_DIR = './acseg/segmentations'
    JSON_DIR = './acseg/jsons'
    LOG_FOLDER = './acseg/dqn_log'

# environment settings
RAW_HEIGHT = 512
RAW_WIDTH = 500
FRAME_HEIGHT = dqn_chamber2.FRAME_HEIGHT
FRAME_WIDTH = dqn_chamber2.FRAME_WIDTH
# DO_CENTRALIZE = True
DO_CENTRALIZE = dqn_chamber2.DO_CENTRALIZE
# ADD_LAST = 1
# CHEAT_PEEK = 1
ADD_LAST = dqn_chamber2.ADD_LAST
CHEAT_PEEK = dqn_chamber2.CHEAT_PEEK
ADJUST_LOSS = dqn_chamber2.ADJUST_LOSS
# 0 for original system of {1:any_true, 0:revisited, -x:divergent}
REWARD_ONLY_NEXT_TRUE = dqn_chamber2.REWARD_ONLY_NEXT_TRUE
REWARD_STATIC_PENALTY = dqn_chamber2.REWARD_STATIC_PENALTY

# NUM_ACTIONS = 9     # directions - allows in place; this is inefficient
NUM_ACTIONS = 8     # directions - allows in place;
# storage settings
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
# STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
STATE_LENGTH = dqn_chamber2.STATE_LENGTH


class ChamberTracer():
    def __init__(self, file_base=FILE_BASE):
        self.file_base = file_base
        observation, coords = load_img(file_base)
        self.coords = [make_coord_key(coord) for coord in coords]   # str for tracking if coord should be visited
        self.original_img = observation     # unadulterated image
        self.reset()

    def reset(self):  # reset original observation and starting coord
        # init_coords = self.coords[0:STATE_LENGTH]  # str
        init_coords = self.coords[0]  # str
        self.last_true_coord = init_coords[-1]  # as traget for getting back on track
        self.visited_coords = init_coords  # str for easy comparison
        self.actions = [0] * len(init_coords)
        self.rewards = [0] * len(init_coords)
        self.state = make_state(self.original_img, visited_coords=self.visited_coords,
                                last_true_coord=self.last_true_coord, true_coords=self.coords)
        # return self.state[:, :, -1]
        return self.state

    def step(self, action, visualise=False):
        observation, reward, terminal = self.calc_observation_reward(action)
        # update self.state
        # self.state = np.append(self.state[:, :, 1:], observation.reshape(FRAME_HEIGHT, FRAME_WIDTH, 1), axis=2)
        self.state = observation
        self.rewards.append(reward)
        self.actions.append(action)

        if visualise:
            visualise_state(self)

        return observation, reward, terminal, {}

    def _is_terminal(self):
        true_coords = set(self.coords)
        common_points = set(self.visited_coords).intersection(true_coords)
        if len(common_points)==len(true_coords):
            return True
        else:
            return False

    def render(self):
        visited_coords = np.asarray([parse_coord_str(x) for x in self.visited_coords])
        plt.figure(1)
        plt.clf()
        plt.imshow(self.original_img)  # since static is this and for new relative images coordinates are to this
        plt.scatter(y=visited_coords[:, 1], x=visited_coords[:, 0], c='red', s=2)
        # traced_coords = [parse_coord_str(x) for x in self.coords]
        # traced_coords_real = np.asarray(traced_coords)
        # plt.scatter(x=traced_coords_real[:,0], y=traced_coords_real[:,1], c='white', s=2)
        plt.show(block=False)

        # visualise_state(self)
        return

    # class utils
    def calc_observation_reward(self, action):
        new_observation, new_coord, terminal = self.action_to_coord(action)
        reward = self.calc_reward(new_coord)
        return new_observation, reward, terminal

    # # if it crosses all coords and has not been visited
    # def calc_reward(self, new_coord):
    #     coord_key = make_coord_key(new_coord)
    #     if coord_key in self.visited_coords[:-1]:   # up to current coord
    #         reward = 0
    #     else:
    #         if coord_key in self.coords:
    #             reward = 1
    #             self.last_true_coord = coord_key    # for accounting purposes
    #         else:
    #             # reward = 0
    #             # since this gets clipped in dqn_chamber
    #             # reward = -self.coord_min_dist(new_coord)/float(max(RAW_HEIGHT, RAW_WIDTH)**2)
    #             x, y = get_coord_to_add(self.coords, self.last_true_coord)
    #             reward = -np.sum((np.asarray([x, y])-np.asarray(new_coord))**2)/float(max(FRAME_HEIGHT, FRAME_WIDTH)**2)
    #     return reward

     # if it crosses all coords and has not been visited
    def calc_reward(self, new_coord):
        coord_key = make_coord_key(new_coord)
        if coord_key in self.visited_coords[:-1]:  # up to current coord
            if REWARD_STATIC_PENALTY:
                reward = -0.1  # small penalty for faffing around
            else:
                reward = 0
        else:
            # always get next true
            x, y = get_coord_to_add(self.coords, self.last_true_coord,
                                    next_true=(CHEAT_PEEK or REWARD_ONLY_NEXT_TRUE))
            target_dist = np.sqrt(np.sum((np.asarray([x, y]) - np.asarray(new_coord)) ** 2))
            reward = -target_dist  # reward is divergence from target
            if ADJUST_LOSS:  # large losses if further away
                reward = reward / float(max(FRAME_HEIGHT, FRAME_WIDTH))

            if REWARD_ONLY_NEXT_TRUE:
                if reward == 0:
                    reward = 1
                    self.last_true_coord = coord_key  # for tracking purposes
            else:
                if coord_key in self.coords:
                    reward = 1
                    self.last_true_coord = coord_key  # for tracking purposes
        return reward

    def coord_min_dist(self, new_coord):
        min_dist, closest_coord = coord_min_dist(self.coords, new_coord)
        return min_dist

    def action_to_coord(self, action, visualise=False):  # 1- hot vector to direction
        # cur_state = self.state[:, :, -1]    # last state is current state
        cur_state = self.state  # last state is current state
        cur_coord = parse_coord_str(self.visited_coords[-1])    # last coord is current coord

        box = action_to_direction_box(action)   # map action to coord system
        new_point = np.transpose(np.nonzero(box))
        w_diff = new_point[0][1] - 1    # col diff is x/width change
        h_diff = new_point[0][0] - 1    # row diff is y/height change

        new_x = cur_coord[0] + w_diff
        new_y = cur_coord[1] + h_diff
        new_coord = (new_x, new_y)
        str_new_coord = make_coord_key(new_coord)
        self.visited_coords.append(str_new_coord)   # needs new coord for making state images

        terminal = False
        if within_img_bounds(new_coord):
            new_img, new_img2 = get_img_and_loc(self.original_img, self.visited_coords, self.last_true_coord, self.coords)
            new_observation = np.stack([new_img, new_img2], axis=2)
            terminal = self._is_terminal()  # check visited vs labeled
        else:   # also terminal if out of bounds
            new_observation = np.copy(cur_state)    # in case pointer problems
            terminal = True

        if visualise:
            box_orig = np.copy(box)
            box_orig[1, 1] = 1  # center point
            plt.imshow(box_orig)  # careful of meaning of x,y
            plt.scatter(y=h_diff + 1, x=w_diff + 1, c='blue', s=10)
        return new_observation, new_coord, terminal


### general utils
def make_coord_key(coord):
    key_str = '{}_{}'.format(coord[0], coord[1])
    return key_str


def parse_coord_str(coord_str):
    x, y = coord_str.split('_')
    return int(x), int(y)


def get_img_and_loc(img, visited_coords, last_true_coord, true_coords, include_history=True, visualise=False):
    # img = make_coord_img(img, visited_coords=visited_coords, last_true_coord=last_true_coord, true_coords=true_coords,
    #                      include_history=include_history)
    img, img2 = make_coord_img(img, visited_coords=visited_coords, last_true_coord=last_true_coord, true_coords=true_coords,
                         include_history=include_history)
    if visualise:
        plt.figure(1)
        plt.imshow(img)
        coord = visited_coords[0]   # only use current coord for now
        plt.scatter(x=coord[0], y=coord[1], c='red', s=10)
    return img, img2


def make_coord_img(img, visited_coords, last_true_coord, true_coords, add_last_true=ADD_LAST, include_history=True, visualise=False):
    # Idea trace path of visited coords and link to rewards for that path
    # num_mem = STATE_LENGTH if include_history else 1
    num_mem = 4 if include_history else 1

    new_img = np.copy(img)
    # No longer change intensity - instead use 2 channels
    # adjust_factor = 2 if add_last_true else 1
    # for idx, coord in enumerate(visited_coords[-1:-num_mem-1:-1]):   # from last (most recent) to first
    #     x, y = parse_coord_str(coord)   # careful about meaning of x, y
    #     # max intensity and linear scale downwards (128-255)
    #     new_img[y, x] = (1-idx/(STATE_LENGTH+1.)) * 127/adjust_factor + 128

    new_img2 = np.zeros(new_img.shape, dtype=np.uint8)
    adjust_factor = 2 if add_last_true else 1
    for idx, coord in enumerate(visited_coords[-1:-num_mem - 1:-1]):  # from last (most recent) to first
        x, y = parse_coord_str(coord)  # careful about meaning of x, y
        new_img2[y, x] = (1 - idx / (STATE_LENGTH + 1.)) * 127 / adjust_factor + 128
    if add_last_true:
        x, y = get_coord_to_add(true_coords, last_true_coord)
        new_img2[y, x] = 255

    # centralize
    if DO_CENTRALIZE:
        cur_coord = parse_coord_str(visited_coords[-1])     # last visited coord
        x, y = cur_coord
        lower_y = int(y-FRAME_HEIGHT/2.)
        upper_y = int(y+FRAME_HEIGHT/2.)
        lower_x = int(x-FRAME_WIDTH/2.)
        upper_x = int(x+FRAME_WIDTH/2.)
        new_img = new_img[lower_y:upper_y, lower_x:upper_x]
        new_img2 = new_img2[lower_y:upper_y, lower_x:upper_x]

    if visualise:
        changed_coords = np.asarray([parse_coord_str(x) for x in visited_coords[-1:-num_mem-1:-1]])
        if DO_CENTRALIZE:
            plt.figure(1); plt.clf()
            plt.imshow(new_img)
            plt.scatter(x=changed_coords[:, 0]-lower_x, y=changed_coords[:, 1]-lower_y, c='red', s=1)
            fig1, ax1 = plt.subplots(1)
            ax1.imshow(img)
            ax1.add_patch(patches.Rectangle((lower_x, lower_y), upper_x - lower_x, upper_y - lower_y, fill=False, color='white', linewidth=1))
            plt.figure(2); plt.clf()
            plt.imshow(new_img2)
        else:
            plt.figure(1); plt.clf()
            plt.imshow(new_img)
            plt.scatter(x=changed_coords[:, 0], y=changed_coords[:, 1], c='red', s=1)
            plt.figure(2); plt.clf()
            plt.imshow(new_img2)

    return new_img, new_img2


def get_coord_to_add(true_coords, last_true_coord):
    if CHEAT_PEEK:
        last_index = true_coords.index(last_true_coord)
        if last_index < len(true_coords):
            peek_coord = true_coords[last_index + 1]
        else:
            peek_coord = true_coords[0]
        x, y = parse_coord_str(peek_coord)
    else:
        x, y = parse_coord_str(last_true_coord)
    return x, y


def load_img(file_base=FILE_BASE):
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
    else:
        h_margin = 0
        w_margin = 0
    if (x < 0+w_margin or x >= RAW_WIDTH-w_margin) or (y < 0+h_margin or y >= RAW_HEIGHT-h_margin):
        return False
    else:
        return True


def make_state(observation, visited_coords, last_true_coord, true_coords):
    img, img2 = get_img_and_loc(observation, visited_coords, last_true_coord, true_coords)
    # state = [processed_observation for _ in range(STATE_LENGTH)]
    state = [img, img2]
    return np.stack(state, axis=2)  # img with 2 channel


def visualise_state(env):
    state_shape = env.state.shape
    # for idx in range(state_shape[-1]):
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

        if counter % 1000==0:
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

    # # zero is do nothing - allow staying in 1 place
    # if action == 0:
    #     pred[4] = 1  # middle is do nothing
    # elif action <= 4:
    #     pred[action - 1] = 1  # 1-shifted because of zero
    # box = pred.reshape((3, 3))
    # # pred[action] = 1  # dont allow no action
    # # pred2 = np.insert(pred, 4, 0)  # insert current position at center
    # # box = pred2.reshape((3, 3))
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


if __name__ == '__main__':
    # playthrough(0)
    # playthrough(1)
    playthrough(2)
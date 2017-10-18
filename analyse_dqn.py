import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dqn_chamber import LOG_FOLDER
from chamber_tracer import load_img

# plot coords and rewards
def visualise_episode(episode_name, episode_data, img, coords, save_movie=False):
    data_shape = episode_data.shape
    fig = plt.figure(1)

    if save_movie:
        ims = []
        for idx in range(data_shape[0]):
            # plt.imshow(img)
            # plt.scatter(x=episode_data[:idx+1, 2], y=episode_data[:idx+1, 3], c='red', s=10)
            ims.append(
                (plt.imshow(img), plt.scatter(x=episode_data[:idx + 1, 2], y=episode_data[:idx + 1, 3], c='red', s=10)))
            # ani = animation.FuncAnimation(fig, update_img, 300, interval=30)
        im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
        im_ani.save('{}/{}'.format(LOG_FOLDER, episode_name))

    # just plot everything at once
    plt.figure(1)
    plt.imshow(img)
    plt.scatter(x=episode_data[:, 2], y=episode_data[:, 3], c='red', s=10)
    return


# parse output files
def analyse_episode(episode_file):
    img, coords = load_img()

    episode_data = np.loadtxt(episode_file, delimiter=',')
    episode_name = episode_file.replace(LOG_FOLDER, '').replace('.txt', '').replace('\\', '').replace('/', '')
    if np.sum(episode_data[:,1])>100:
        visualise_episode(episode_name, episode_data, img, coords)
    return


if __name__ == '__main__':
    episode_logs = sorted(glob.glob('{0}/*.txt'.format(LOG_FOLDER)))
    for idx, log_name in enumerate(episode_logs):
        analyse_episode(log_name)
import os
import matplotlib.pyplot as plt
# from skimage import data
#
# astronaut = data.astronaut()
# ihc = data.immunohistochemistry()
# hubble = data.hubble_deep_field()
#
# # Initialize the subplot panels side by side
# fig, ax = plt.subplots(nrows=1, ncols=3)
#
# # Show an image in each subplot
# ax[0].imshow(astronaut)
# ax[0].set_title('Natural image')
# ax[1].imshow(ihc)
# ax[1].set_title('Microscopy image')
# ax[2].imshow(hubble)
# ax[2].set_title('Telescope image');

# Create a temporary directory
d = os.path.join('tempdir')
if not os.path.isdir(d):
    os.makedirs(d)

# Return the tail of the path
os.path.basename('http://google.com/attention.zip')

# from urllib.request import urlretrieve
# # Define URL
# url = 'http://www.fil.ion.ucl.ac.uk/spm/download/data/attention/attention.zip'
# # Retrieve the data
# fn, info = urlretrieve(url, os.path.join(d, 'attention.zip'))
# import zipfile
# # Extract the contents into the temporary directory we created earlier
# zipfile.ZipFile(fn).extractall(path=d)
#
# # List first 10 files
# [f.filename for f in zipfile.ZipFile(fn).filelist[:10]]
#
# import nibabel
#
# # Read the image
# struct = nibabel.load(os.path.join(d, 'attention/structural/nsM00587_0002.hdr'))

# # Get a plain NumPy array, without all the metadata
# struct_arr = struct.get_data()

from skimage import io

struct_arr = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")

plt.imshow(struct_arr[75])
plt.imshow(struct_arr[75], aspect=0.5)

struct_arr2 = struct_arr.T
plt.imshow(struct_arr2[34])
# plt.imshow(struct_arr2[5])

# def multi_slice_viewer(volume):
#     fig, ax = plt.subplots()
#     ax.volume = volume
#     ax.index = volume.shape[0] // 2
#     ax.imshow(volume[ax.index])
#     fig.canvas.mpl_connect('key_press_event', process_key)
#
#
# def process_key(event):
#     fig = event.canvas.figure
#     ax = fig.axes[0]
#     if event.key == 'j':
#         previous_slice(ax)
#     elif event.key == 'k':
#         next_slice(ax)
#     fig.canvas.draw()
#
#
# def previous_slice(ax):
#     """Go to the previous slice."""
#     volume = ax.volume
#     ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
#     ax.images[0].set_array(volume[ax.index])
#
#
# def next_slice(ax):
#     """Go to the next slice."""
#     volume = ax.volume
#     ax.index = (ax.index + 1) % volume.shape[0]
#     ax.images[0].set_array(volume[ax.index])
#
# multi_slice_viewer(struct_arr2)
# plt.rcParams['keymap.<command>'] = ['<key 1>', '<key 2>']


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


multi_slice_viewer(struct_arr2)
#
# import shutil
# # Remove the temporary directory
# shutil.rmtree(d)
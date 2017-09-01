import random
import numpy as np

img_rows = 512
img_cols = 512
cropped_rows = 128
cropped_cols = 496
pixel_overlap = 8
num_train = int(75*((img_rows - cropped_rows)/pixel_overlap * (img_cols - cropped_cols)/pixel_overlap))  # out of 79 at the moment


def trainGenerator(batch_size, data, mask):
  # sl_data, sl_mask = slice_data(data[0:num_train, :], mask[0:num_train, :])
  sl_data = data[0:num_train, :]
  sl_mask = mask[0:num_train, :]

  while True:
    # randomize order at the beginning of each epoch
    p = np.random.permutation(len(sl_data))

    for i in xrange(0, sl_data.shape[0], batch_size):
      batch_x = np.copy(sl_data[i:i + batch_size])
      batch_y = np.copy(sl_mask[i:i + batch_size])
      if batch_x.shape[0] != batch_size:
        break
      yield (batch_x, batch_y)


def validGenerator(batch_size, data, mask):
  # sl_data, sl_mask = slice_data(data[num_train:, :], mask[num_train:, :])
  sl_data = data[num_train:, :]
  sl_mask = mask[num_train:, :]

  while True:
    # randomize order at the beginning of each epoch
    p = np.random.permutation(len(sl_data))

    for i in xrange(0, sl_data.shape[0], batch_size):
      batch_x = np.copy(sl_data[i:i + batch_size])
      batch_y = np.copy(sl_mask[i:i + batch_size])
      if batch_x.shape[0] != batch_size:
        break
      yield (batch_x, batch_y)


# from aaron's dna work
def translationGenerator(batch_size, targetmut, condition):

  while True:
    # randomize order at the beginning of each epoch
    p = np.random.permutation(len(train_x))
    train_x = train_x[p]
    train_y = train_y[p]

    for i in xrange(0, train_x.shape[0], batch_size):
      batch_x = np.copy(train_x[i:i+batch_size])
      batch_y = np.copy(train_y[i:i+batch_size])
      if batch_x.shape[0] != batch_size:
        break
      yield (batch_x, batch_y)


if __name__ == '__main__':
  from data import load_train_data, load_valid_data, load_sliced_data
  imgs_train, imgs_mask_train = load_train_data()
  # slice_data(imgs_train, imgs_mask_train)
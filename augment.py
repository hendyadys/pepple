import random
import numpy as np

img_rows = 512
img_cols = 512
cropped_rows = 128
cropped_cols = 496
pixel_overlap = 8
num_train = int(75*((img_rows - cropped_rows)/pixel_overlap * (img_cols - cropped_cols)/pixel_overlap))  # out of 79 at the moment
train_val_split = 0.8


def trainGenerator(batch_size, data, mask):
  sl_data = data[0:num_train, :]
  sl_mask = mask[0:num_train, :]

  while True:
    # randomize order at the beginning of each epoch
    p = np.random.permutation(len(sl_data))
    sl_data = sl_data[p, ]
    sl_mask = sl_mask[p, ]

    for i in xrange(0, sl_data.shape[0], batch_size):
      batch_x = np.copy(sl_data[i:i + batch_size])
      batch_y = np.copy(sl_mask[i:i + batch_size])
      if batch_x.shape[0] != batch_size:
        break
      yield (batch_x, batch_y)


def validGenerator(batch_size, data, mask):
  sl_data = data[num_train:, :]
  sl_mask = mask[num_train:, :]

  while True:
    # randomize order at the beginning of each epoch
    p = np.random.permutation(len(sl_data))
    sl_data = sl_data[p, ]
    sl_mask = sl_mask[p, ]

    for i in xrange(0, sl_data.shape[0], batch_size):
      batch_x = np.copy(sl_data[i:i + batch_size])
      batch_y = np.copy(sl_mask[i:i + batch_size])
      if batch_x.shape[0] != batch_size:
        break
      yield (batch_x, batch_y)


def trainGenerator2(batch_size, data, masks, empty_data, empty_masks):
  num_imgs, _, _, _ = data.shape
  num_imgs_train = int(num_imgs * train_val_split)  # approx 82

  num_empty, _, _, _ = empty_data.shape
  # num_empty_train = int(min(num_empty * train_val_split, .2*num_imgs_train))   # limit empty to 16% of training data
  num_empty_train = int(.5*num_empty)   # more empty images for training. 87/2=43 -> approx 1/3 training imgs now empty

  seg_train = np.concatenate((data[:num_imgs_train, ], empty_data[:num_empty_train, ]), axis=0)
  masks_train = np.concatenate((masks[:num_imgs_train, ], empty_masks[:num_empty_train, ]), axis=0)

  # seg_train = seg_train.astype('float32')
  # masks_train = masks_train.astype('float32')

  while True:
    # randomize order at the beginning of each epoch
    p = np.random.permutation(len(seg_train))
    shuffled_data = seg_train[p, ]
    shuffled_masks = masks_train[p, ]

    for i in xrange(0, shuffled_data.shape[0], batch_size):
      batch_x = np.copy(shuffled_data[i:i + batch_size])
      batch_y = np.copy(shuffled_masks[i:i + batch_size])
      if batch_x.shape[0] != batch_size:
        break

      # augment training side; target/mask is fine
      for jdx in range(batch_size):
        snap_x = np.copy(batch_x[jdx, ])
        snap_y = np.copy(batch_y[jdx, ])

        rand_scale = np.random.uniform(low=0.75, high=1.25)
        # rand_shift = np.random.randint(-50, high=50)
        # snap_x += rand_shift
        snap_x *= rand_scale
        snap_x = np.round(snap_x)
        snap_x = np.clip(snap_x, 0., 255.)
        batch_x[jdx, ] = snap_x
        batch_y[jdx, ] = snap_y

      yield (batch_x, batch_y)


def validGenerator2(batch_size, data, masks, empty_data, empty_masks):
  num_imgs, _, _, _ = data.shape
  num_imgs_train = int(num_imgs * train_val_split)

  num_empty, _, _, _ = empty_data.shape
  # num_empty_train = int(min(num_empty * train_val_split, .2*num_imgs_train))   # limit empty to 16% of training data
  num_empty_train = int(.5 * num_empty)  # more empty images for training. 87/2=43 -> approx 1/3 training imgs now empty

  # need to respect ratio of segmented vs empty in validation
  sampled_idx = np.array(random.sample(range(num_empty_train, num_empty), int((num_imgs - num_imgs_train) * num_empty_train
                                                                              / (num_imgs_train+num_empty_train))))
  # print(num_empty, num_empty_train, len(sampled_idx), num_imgs, num_imgs_train)
  empty_sample = empty_data[sampled_idx, ]
  empty_masks_sample = empty_masks[sampled_idx, ]

  seg_valid = np.concatenate((data[num_imgs_train:, ], empty_sample), axis=0)
  masks_valid = np.concatenate((masks[num_imgs_train:, ], empty_masks_sample), axis=0)

  while True:
    # randomize order at the beginning of each epoch
    p = np.random.permutation(len(seg_valid))
    shuffled_data = seg_valid[p, ]
    shuffled_masks = masks_valid[p, ]

    for i in xrange(0, shuffled_data.shape[0], batch_size):
      batch_x = np.copy(shuffled_data[i:i + batch_size])
      batch_y = np.copy(shuffled_masks[i:i + batch_size])
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
import sys
import threading
import glob
from PIL import Image
import random
import argparse

import keras
from keras.models import Model
from keras.layers import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from tiramisu.model import create_tiramisu
from camvid.mapping import map_labels


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for training The One Hundred Layers Tiramisu network.')

    parser.add_argument('--output_path',
                        help='Path for saving a training model as a *.h5 file. Default is models/new_tiramisu.h5',
                        default='models/new_tiramisu.h5')
    parser.add_argument('--path_to_raw',
                        help='Path to raw images used for training. Default is camvid-master/701_StillsRaw_full/',
                        default='camvid-master/701_StillsRaw_full/')
    parser.add_argument('--image_size',
                        help='Size of the input image. Default is [360, 480]',
                        default=(360, 480))
    parser.add_argument('--path_to_labels',
                        help='Path to labeled images used for training. Default is camvid-master/LabeledApproved_full/',
                        default='camvid-master/LabeledApproved_full/')
    parser.add_argument('--path_to_labels_list',
                        help='Path to file defining classes used in camvid dataset. '
                             'Only used if convert_from_camvid = True. Default is camvid-master/label_colors.txt',
                        default='camvid-master/label_colors.txt')
    parser.add_argument('--log_dir',
                        help='Path for storing tensorboard logging. Default is logging/',
                        default='logging/')
    parser.add_argument('--convert_from_camvid',
                        help='Flag that defines if camvid data is used. '
                             'If enabled it maps camvid data labeling to integers. Default: True',
                        default=True)
    parser.add_argument('--training_percentage',
                        help='Defines percentage of total data that will be used for training. '
                             'Default: 70 training 30 validation',
                        default=70)
    parser.add_argument('--no_epochs',
                        type=int,
                        help='Defines number of epochs used for training. '
                             'Default: 250',
                        default=250)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Defines learning rate used for training. '
                             'Default: 1e-3',
                        default=1e-3)
    parser.add_argument('--patience',
                        type=int,
                        help='Defines patience for early stopping. '
                             'Default: 50',
                        default=50)
    parser.add_argument('--path_to_model_weights',
                        help='Path to saved model weights if training should be resumed. '
                             'Default: models/new_tiramisu.h5',
                        default='models/new_tiramisu.h5')
    parser.add_argument('--train_from_zero',
                        type=bool,
                        help='Boolean, defines if training from scratch or resuming from saved h5 file. '
                             'Default: True',
                        default=True)

    return parser.parse_args(args)


class BatchIndices(object):
    def __init__(self, n, bs, shuffle=False):
        self.n, self.bs, self.shuffle = n, bs, shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (np.random.permutation(self.n)
                     if self.shuffle else np.arange(0, self.n))
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n: self.reset()
            ni = min(self.bs, self.n-self.curr)
            res = self.idxs[self.curr:self.curr+ni]
            self.curr += ni
            return res


class AugmentationGenerator(object):
    def __init__(self, x, y, bs=64, out_sz=(224, 224), train=True):
        self.x, self.y, self.bs, self.train = x, y, bs, train
        self.n, self.ri, self.ci, _ = x.shape
        self.idx_gen = BatchIndices(self.n, bs, train)
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape) == 4 else 1

    def get_slice(self, i, o):
        start = random.randint(0, i-o) if self.train else (i-o)
        return slice(start, start+o)

    def get_item(self, idx):
        slice_r = self.get_slice(self.ri, self.ro)
        slice_c = self.get_slice(self.ci, self.co)
        x = self.x[idx, slice_r, slice_c]
        y = self.y[idx, slice_r, slice_c]

        # flip randomly
        if self.train and (random.random()>0.5):
            y = y[:, ::-1]
            x = x[:, ::-1]
        return x, y

    def __next__(self):
        idxs = next(self.idx_gen)

        # get random item
        items = (self.get_item(idx) for idx in idxs)
        xs, ys = zip(*items)
        return np.stack(xs), np.stack(ys).reshape(len(ys), -1, self.ych)


def load_image(fn, img_size):
    return np.array(Image.open(fn).resize(img_size, Image.NEAREST))


def load_data(path_to_raw, path_to_labels, img_size):

    # Load images
    image_files = glob.glob(path_to_raw + '*.png')
    images = np.stack([load_image(filename, img_size) for filename in image_files])

    # Load labels
    label_files = glob.glob(path_to_labels + '*.png')
    labels = np.stack([load_image(filename, img_size) for filename in label_files])

    # Normalize pixel values in images
    images = images / 255.
    images -= images.mean()
    images /= images.std()

    return images, labels


def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    img_size = args.image_size

    raw, labels = load_data(args.path_to_raw, args.path_to_labels, img_size)

    if args.convert_from_camvid:
        labels = map_labels(args.path_to_labels_list, labels, img_size[1], img_size[0])

    # take args.training_percentage of data for training
    n = len(raw)
    n_train = round(n*args.training_percentage/100)

    # divide into train and val
    train_set = raw[:n_train]
    train_labels = labels[:n_train]
    val_set = raw[n_train:]
    val_labels = labels[n_train:]

    train_generator = AugmentationGenerator(train_set, train_labels, 1, train=True)
    test_generator = AugmentationGenerator(val_set, val_labels, 1, train=False)

    input_shape = (224, 224, 3)
    img_input = Input(shape=input_shape)
    x = create_tiramisu(32, img_input)
    model = Model(img_input, x)

    if not args.train_from_zero:
        model.load_weights(args.path_to_model_weights)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(args.learning_rate, decay=1-0.99995), metrics=["accuracy"])

    logging = TensorBoard(log_dir=args.log_dir)
    checkpoint = ModelCheckpoint(args.log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 monitor='val_loss', save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=args.patience, verbose=1, mode='auto')

    model.fit_generator(train_generator, len(train_set), args.no_epochs, verbose=2,
                        validation_data=test_generator, validation_steps=len(val_set),
                        callbacks=[logging, checkpoint, early_stopping])

    model.save_weights(args.output_path)


if __name__ == '__main__':
    main()





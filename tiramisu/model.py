from keras.layers import *
from keras.regularizers import l2


def relu_bn(x): return Activation('relu')(BatchNormalization(axis=-1)(x))


def conv(x, nf, sz, wd, p, stride=1):
    x = Conv2D(nf, (sz, sz), strides=(stride, stride), padding='same', kernel_initializer='he_uniform',
               kernel_regularizer=l2(wd))(x)
    return Dropout(p)(x) if p else x


def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)


def dense_block(n, x, growth_rate, p, wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = Concatenate(axis=-1)([x, b])
        added.append(b)
    return x, added


def transition_dn(x, p, wd):
    # in the paper stride=1 but better results with stride=2
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)


def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i, n in enumerate(nb_layers):
        x, added = dense_block(n, x, growth_rate, p, wd)

        # keep track of skip connections
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added


def transition_up(added, wd=0):
    x = Concatenate(axis=-1)(added)
    _, r, c, ch = x.get_shape().as_list()
    return Conv2DTranspose(ch, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform',
                 kernel_regularizer = regularizers.l2(wd))(x)


def up_path(added, skips, nb_layers, growth_rate, p, wd):
    # use previously saved list of skip connections
    for i, n in enumerate(nb_layers):
        x = transition_up(added, wd)

        # concatenate the skip connections
        x = Concatenate(axis=-1)([x, skips[i]])
        x, added = dense_block(n, x, growth_rate, p, wd)
    return x


def reverse(a): return list(reversed(a))


def create_tiramisu(nb_classes, img_input, nb_dense_block=6,
                    growth_rate=16, nb_filter=48, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4):

    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips, added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = conv(x, nb_classes, 1, wd, 0)
    _, r, c, f = x.get_shape().as_list()
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)
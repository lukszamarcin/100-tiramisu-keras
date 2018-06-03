from PIL import Image
from keras.models import Model
from keras.layers import *
import argparse
import sys

from camvid.mapping import decode
from tiramisu.model import create_tiramisu


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for inference with models trained on CamVid data')

    parser.add_argument('--path_to_test_file',
                        help='Path to the image you would like to test with. Default is: images/testImage0.png',
                        default='images/test_image0.png')
    parser.add_argument('--path_to_result',
                        help='Path to the folder and filename where the result of segmentation should be saved. '
                             'Default is: images/test_outcome.png',
                        default='images/test_outcome.png')
    parser.add_argument('--path_to_model',
                        help='Path to the h5 file with the model weight that should be used for inference. '
                             'Default is: models/my_tiramisu.h5',
                        default='models/my_tiramisu.h5')
    parser.add_argument('--path_to_labels_list',
                        help='Path to file defining classes used in camvid dataset. '
                             'Only used if convert_from_camvid = True. Default is camvid-master/label_colors.txt',
                        default='camvid-master/label_colors.txt')

    return parser.parse_args(args)


def color_label(img, id2code):
    rows, cols = img.shape
    result = np.zeros((rows, cols, 3), 'uint8')
    for j in range(rows):
        for k in range(cols):
            result[j, k] = id2code[img[j, k]]
    return result


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Load model and weights
    input_shape = (224, 224, 3)
    number_classes = 32  # CamVid data consist of 32 classes

    img_input = Input(shape=input_shape)
    x = create_tiramisu(number_classes, img_input)
    model = Model(img_input, x)
    model.load_weights(args.path_to_model)

    # load your own image
    try_image = Image.open(args.path_to_test_file).resize((input_shape[0], input_shape[1]), Image.NEAREST)
    try_image.show()

    try_image = np.array(try_image)
    try_image = try_image / 255.
    try_image -= 0.39  # mean used for normalization - specific to CamVid dataset
    try_image /= 0.30  # std used for normalization - specific to CamVid dataset

    # Use loaded model for prediction on input image
    prediction = model.predict(np.expand_dims(try_image, 0), 1)
    prediction = np.argmax(prediction, axis=-1)

    # Visualize the outcome
    outcome = np.resize(prediction, (input_shape[0], input_shape[1]))
    label_codes, label_names, code2id = decode(args.path_to_labels_list)
    print(list(zip(label_codes, label_names)))

    id2code = {val: key for (key, val) in code2id.items()}
    outcome = color_label(outcome, id2code)

    img = Image.fromarray(outcome)
    img.show()

    img.save(args.path_to_result)


if __name__ == '__main__':
    main()

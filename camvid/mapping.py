"""
Maps the colors defined for each class (as R, G, B values) in CamVid dataset to integers for classification purposes

Color labeling in CamVid
64 128 64	Animal
192 0 128	Archway
0 128 192	Bicyclist
0 128 64	Bridge
128 0 0		Building
64 0 128	Car
64 0 192	CartLuggagePram
...

"""

import numpy as np


# make the segmented targets into integers for classification purposes
def parse_code(text):
    if len(text.strip().split("\t")) == 2:
        a, b = text.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), b
    else:
        a, b, c = text.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), c


def convert_one_label(i, failed_code, rows, cols, labels, code2id):
    res = np.zeros((rows, cols), 'uint8')

    for j in range(rows):
        for k in range(cols):
            try: res[j, k] = code2id[tuple(labels[i, j, k])]
            except: res[j, k] = failed_code
    return res


def decode(path_to_labels_list):
    label_codes, label_names = zip(*[parse_code(l) for l in open(path_to_labels_list)])
    label_codes, label_names = list(label_codes), list(label_names)  # combine into a list

    # remove unused classes

    # assign id to every color code
    code2id = {v: k for k, v in enumerate(label_codes)}

    return label_codes, label_names, code2id


def map_labels(path_to_labels_list, labels, rows, cols):

    label_codes, label_names, code2id = decode(path_to_labels_list)

    n = len(labels)  # number of sample images
    failed_code = len(label_codes) + 1

    # Map all labels
    print('Mapping camvid data to integers - this can take a while...\n')
    labels_int = np.stack([convert_one_label(x, failed_code, rows, cols, labels, code2id) for x in range(n)])

    # Set erroneous pixels to zero
    labels_int[labels_int == failed_code] = 0

    return labels_int






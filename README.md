# Semantic segmentation with The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation

Original paper:
https://arxiv.org/abs/1611.09326

Keras implementation of 100 layer Tiramisu for semantic segmentaton

## Running a demo
In progress...

## Training

Run the following to train with default configuration (training from scratch on CamVid dataset) - it assumes that camvid data is in the same folder (clone it first from the repo mentioned below)
```
python train.py
```

If you would like to change the default input parameters for training, type the following for help
```
python train.py -h
```

or check the overview of the inputs here:

```
 --output_path OUTPUT_PATH
                        Path for saving a training model as a *.h5 file.
                        Default is models/new_tiramisu.h5
  --path_to_raw PATH_TO_RAW
                        Path to raw images used for training. Default is
                        camvid-master/701_StillsRaw_full/
  --path_to_labels PATH_TO_LABELS
                        Path to labeled images used for training. Default is
                        camvid-master/LabeledApproved_full/
  --path_to_labels_list PATH_TO_LABELS_LIST
                        Path to file defining classes used in camvid dataset.
                        Only used if convert_from_camvid = True. Default is
                        camvid-master/label_colors.txt
  --log_dir LOG_DIR     Path for storing tensorboard logging. Default is
                        logging/
  --convert_from_camvid CONVERT_FROM_CAMVID
                        Flag that defines if camvid data is used. If enabled
                        it maps camvid data labeling to integers
  --training_percentage TRAINING_PERCENTAGE
                        Defines percentage of total data that will be used for
                        training.Default: 70 training 30 testing
```
Tensorboard is supported to view run:
~~~
tensorboard --logdir=path/to/log-directory
~~~

## Notes
Originally trained and tested with CamVid dataset available here: https://github.com/mostafaizz/camvid.git  
Inspired by Jeremy's Howard fastAI course: http://www.fast.ai/  

# Semantic segmentation with The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation

Original paper:
https://arxiv.org/abs/1611.09326

Keras implementation of 100 layer Tiramisu for semantic segmentaton

Tested with:
Python 3  
Tensorflow 1.6.0  
Keras 2.1.5  

## Running a demo
Test network trained with CamVid data on custom image by running:  
~~~
python run_tiramisu_camvid.py
~~~

With optional arguments:  
~~~
optional arguments:
  -h, --help            show this help message and exit
  --path_to_test_file PATH_TO_TEST_FILE
                        Path to the image you would like to test with. Default
                        is: images/testImage0.png
  --path_to_result PATH_TO_RESULT
                        Path to the folder and filename where the result of
                        segmentation should be saved. Default is:
                        images/test_image1_outcome.png
  --path_to_model PATH_TO_MODEL
                        Path to the h5 file with the model weight that should
                        be used for inference. Default is:
                        models/my_tiramisu.h5
  --path_to_labels_list PATH_TO_LABELS_LIST
                        Path to file defining classes used in camvid dataset.
                        Only used if convert_from_camvid = True. Default is
                        camvid-master/label_colors.txt
~~~
## Training

Run the following to train with default configuration (training from scratch on CamVid dataset) - it assumes that camvid data is in the same folder (clone it first from the repo mentioned in Notes section)
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

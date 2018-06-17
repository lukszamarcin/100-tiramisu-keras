# Semantic segmentation with The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation

Original paper:
https://arxiv.org/abs/1611.09326

Keras implementation of 100 layer Tiramisu for semantic segmentaton. Model FC-DenseNet103 from the paper above.

![alt text](https://raw.githubusercontent.com/xxmarl/100-tiramisu-keras/master/images/test_image3_small.png)
![alt text](https://raw.githubusercontent.com/xxmarl/100-tiramisu-keras/master/images/test_image3_outcome.png)

Tested with:
Python 3.5.2  
Tensorflow 1.6.0  
Keras 2.1.5  

## Running a demo
Before running a demo, download weights trained on CamVid data from the following link and place it under models\\  
https://drive.google.com/file/d/1T7GP7h0Q8DMLCQ3vgQdadBrFD_vZ9io3/view?usp=sharing

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

Run the following to train with default configuration (training on CamVid dataset) - it assumes that camvid data is in the same folder (clone it first from the repo mentioned in Notes section)
```
python train.py
```
with optional arguments:

```
  --output_path OUTPUT_PATH
                        Path for saving a training model as a *.h5 file.
                        Default is models/new_tiramisu.h5
  --path_to_raw PATH_TO_RAW
                        Path to raw images used for training. Default is
                        camvid-master/701_StillsRaw_full/
  --image_size IMAGE_SIZE
                        Size of the input image. Default is [360, 480]
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
                        it maps camvid data labeling to integers. Default:
                        True
  --training_percentage TRAINING_PERCENTAGE
                        Defines percentage of total data that will be used for
                        training. Default: 70 training 30 validation
  --no_epochs NO_EPOCHS
                        Defines number of epochs used for training. Default:
                        250
  --learning_rate LEARNING_RATE
                        Defines learning rate used for training. Default: 1e-3
  --patience PATIENCE   Defines patience for early stopping. Default: 50
  --path_to_model_weights PATH_TO_MODEL_WEIGHTS
                        Path to saved model weights if training should be
                        resumed. Default: models/new_tiramisu.h5
  --train_from_zero TRAIN_FROM_ZERO
                        Boolean, defines if training from scratch or resuming
                        from saved h5 file. Default: True
```
Tensorboard is supported to view run:
~~~
tensorboard --logdir=path/to/log-directory
~~~
## TODO
- Reproduce the paper error on CamVid
- Try other datasets

## Notes
Originally trained and tested with CamVid dataset available here: https://github.com/mostafaizz/camvid.git  
Inspired by Jeremy's Howard fastAI course: http://www.fast.ai/  
More on CamVid: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/#ClassLabels

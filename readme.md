Requirements

The code is written in Lua and Python. You need the Natural Language Toolkit (NLTK) (http://www.nltk.org/), Torch (http://torch.ch/), and the 19-layer VGGNet model found at http://www.robots.ox.ac.uk/~vgg/research/very_deep/. There are other dependencies that your system may or may not have. See the readme at https://github.com/VT-vision-lab/VQA_LSTM_CNN for more info. Download the data from https://inclass.kaggle.com/c/visual-question-answering/data and extract them into the data folder.

Training

The data is already provided in the data folder. The first step is preprocessing the questions. cd to the data folder and run 

$ python my2_vqa_preprocessing.py --split 3

This will preprocess the multiple choice questions and answers. You will get two files: 'vqa_raw_train.json' and 'vqa_raw_test.json'. 

Next, run

$ python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 2

to get the question features. --num_ans specifies the number of answers (classes) to use. Since we only are dealing with yes or no, this is 2. Two files, `data_prepro.h5` and `data_prepro.json`, will be generated in the main folder. 

Next, Torch is used to preprocess the images. Run

$ th prepro_img.lua -input_json data_prepro.json -image_root /data/train_images -cnn_proto VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model path_to_cnn_model

where path_to_cnn_model should be replaced to wherever the .caffemodel file is found for VGGNet. When I ran this step, I ran out of memory. Perhaps a better video card is needed, or the configuration files need changing.

Finally, train the data using

$th train.lua

This will generate the model under 'model/save'. The code provided is a slightly modified version of that found here

@misc{Lu2015,
author = {Jiasen Lu, Xiao Lin, Dhruv Batra and Devi Parikh},
title = {Deeper LSTM and normalized CNN Visual Question Answering model},
year = {2015},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/VT-vision-lab/VQA_LSTM_CNN}},
commit = {6c91cb9}
}

The dataset is a modified version of that from visualqa.org.

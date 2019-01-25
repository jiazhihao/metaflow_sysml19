#!/usr/bin/env bash

./cnn_cuda --export squeezenet_raw --dnn squeezenet --noopt > squeezenet_raw_output
./cnn_cuda --export inception_raw --dnn inception --noopt > inception_raw_output
./cnn_cuda --export resnet34_raw --dnn resnet34 --noopt > resnet34_raw_output
./cnn_cuda --export rnntc_raw --dnn rnntc --noopt > rnntc_raw_output

./cnn_cuda --export squeezenet_opt --dnn squeezenet > squeezenet_opt_output
./cnn_cuda --export inception_opt --dnn inception > inception_opt_output
./cnn_cuda --export resnet34_opt --dnn resnet34 > resnet34_opt_output
./cnn_cuda --export rnntc_opt --dnn rnntc > rnntc_opt_output

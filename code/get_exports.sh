#!/usr/bin/env bash

./cnn_cuda --export squeezenet_raw --squeezenet --noopt > squeezenet_raw_output
./cnn_cuda --export inception_raw --inception --noopt > inception_raw_output
./cnn_cuda --export resnet_raw --resnet --noopt > resnet_raw_output
./cnn_cuda --export rnntc_raw --rnntc --noopt > rnntc_raw_output
./cnn_cuda --export nmt_raw --nmt --noopt > nmt_raw_output

./cnn_cuda --export squeezenet_opt --squeezenet > squeezenet_opt_output
./cnn_cuda --export inception_opt --inception > inception_opt_output
./cnn_cuda --export resnet_opt --resnet > resnet_opt_output
./cnn_cuda --export rnntc_opt --rnntc > rnntc_opt_output
./cnn_cuda --export nmt_opt --nmt > nmt_opt_output

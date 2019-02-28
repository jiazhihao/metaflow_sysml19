#!/usr/bin/env bash

../mf --export squeezenet_raw --dnn squeezenet --noopt > squeezenet_raw_output
../mf --export inception_raw --dnn inception --noopt > inception_raw_output
../mf --export resnet50_raw --dnn resnet50 --noopt > resnet50_raw_output
../mf --export rnntc_raw --dnn rnntc --noopt > rnntc_raw_output

../mf --export squeezenet_opt --dnn squeezenet > squeezenet_opt_output
../mf --export inception_opt --dnn inception > inception_opt_output
../mf --export resnet50_opt --dnn resnet50 > resnet50_opt_output
../mf --export rnntc_opt --dnn rnntc > rnntc_opt_output

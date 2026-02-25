#!/bin/bash

# Activate the ml virtual environment
source /home/ac.zzheng/env/ml/bin/activate

# Run the dl.py script with default parameters
python dl.py --model vgg16 --num-gpus 2 --batch-size 8192 --epochs 3 --lr 0.001



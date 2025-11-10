#!/bin/bash
python main.py --dataset FB15k-237\
    --cuda True\
    --device cuda:0\
    --batch_size 512\
    --max_grad_norm 3.0\
    --nneg 100\
    --npos 1\
    --margin 1\
    --max_norm 5.\
    --lr 0.00717548\
    --gamma 0.9\
    --step_size 30\
    --num_epochs 50\
    --dim 32\
    --valid_steps 25\
    --early_stop 20\
    --optimizer radam\
    --noise_reg 0.01\

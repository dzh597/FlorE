#!/bin/bash
python main.py --dataset codex-s\
    --cuda True\
    --device cuda:0\
    --batch_size 512\
    --max_grad_norm 3.0\
    --nneg 200\
    --npos 1\
    --margin 1.08\
    --max_norm 5.\
    --lr 0.05\
    --gamma 0.9\
    --step_size 40\
    --num_epochs 50\
    --dim 32\
    --valid_steps 25\
    --early_stop 20\
    --optimizer radam\
    --noise_reg 0.05\

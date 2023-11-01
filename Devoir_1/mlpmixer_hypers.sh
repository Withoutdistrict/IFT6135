#!/bin/sh


parallel -j 5 python main.py --model=mlpmixer --device=cuda --model_config=model_configs/mlpmixer.json --logdir=logdir_mlpmixer/logdir_wd{1} --lr=0.001 --epochs=15 ::: 0.05 0.005 0.0005 0.00005 0.000005
parallel -j 4 python main.py --model=mlpmixer --device=cuda --model_config=model_configs/mlpmixer.json --logdir=logdir_mlpmixer/logdir_bs{1}  --epochs=15 --batch_size={1} ::: 128 256 512 1024
parallel -j 4 python main.py --model=mlpmixer --device=cuda --model_config=model_configs/mlpmixer.json --logdir=logdir_mlpmixer/logdir_{1} --optimizer={1} --lr=0.001 --epochs=15 ::: sgd momentum adam adamw
parallel -j 5 python main.py --model=mlpmixer --device=cuda --model_config=model_configs/mlpmixer.json --logdir=logdir_mlpmixer/logdir_lr{1} --optimizer=adam --lr={1} --epochs=15 ::: 0.1 0.01 0.001 0.0001 0.00001

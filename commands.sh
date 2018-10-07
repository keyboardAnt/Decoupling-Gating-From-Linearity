python datasets.py results/datasets.txt relu galu0s linear -k 16 32 48 64 80 96 --dataset mnist fmnist -a 1. 0.5 0.1 0.01 2. 10. --lr 0.001 0.0005 0.0001 --seed 0 --repeat 4
python r1_regression.py results/r1.txt relu galu galu_opt -m 2048 1024 512 -d 32 24 16 8 -k 32 24 16 8 --hill 0 512 --seed 0 --repeat 16
python r1_generalization.py results/generalization.txt relu galu galu_opt --problem_type regression classification --sigma_y 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --seed 0 --repeat 16
python minimal_k.py results/minimal_k.txt relu galu0 -m 1024 -d 32 -d_ 128 96 64 32 16 8 1 --seed 0 --repeat 8
python plots.py


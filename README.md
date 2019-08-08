# Discovering-Underlying-Plans-Based-on-Shallow-Models
Implementation of our paper "Discovering Underlying Plans Based on Shallow Models{https://arxiv.org/abs/1803.02208}"

Install Depedencies:

sudo pip install smart-open

sudo pip install cntk

sudo pip install --upgrade gensim

To run the RNNPlanner training script:

To run the DUP evaluation code: 
python train_and_test_final.py --win_range 1 1 --domain depots  --num_missing  10 10 --mode middle_cons --top_k 10 10--win_range 1 1 --domain depots  --num_missing  10 10 --mode middle_cons --top_k 10 10

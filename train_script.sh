CUDA_VISIBLE_DEVICES=0,1,6,7 python tools/plain_train_net.py --num_gpus 4 --gpu 0,1,6,7 --batch_size 8 --backbone dla34 --output output/DLA-34_LITE_FPN_003nd

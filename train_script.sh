CUDA_VISIBLE_DEVICES=1,2,3,4 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --gpu 1,2,3,4 --num_gpus 4 --output output/DLA-34_BCPV2.0_4_GPUS_001nd

CUDA_VISIBLE_DEVICES=1,2,3,4 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --gpu 1,2,3,4 --num_gpus 4 --output output/DLA-34_BCPV2.0_4_GPUS_002nd

CUDA_VISIBLE_DEVICES=1,2,3,4 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --gpu 1,2,3,4 --num_gpus 4 --output output/DLA-34_BCPV2.0_4_GPUS_003nd

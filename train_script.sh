CUDA_VISIBLE_DEVICES=1,4,5,6 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --gpu 1,4,5,6 --num_gpus 4 --output output/DLA-34_BCP_TRAINVAL_TEST_UNION_001nd
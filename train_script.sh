CUDA_VISIBLE_DEVICES=4,5 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --gpu 4,5 --num_gpus 2 --output output/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_001nd
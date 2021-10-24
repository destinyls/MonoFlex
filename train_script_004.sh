CUDA_VISIBLE_DEVICES=4 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --output checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_004nd

CUDA_VISIBLE_DEVICES=4 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --output checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_008nd
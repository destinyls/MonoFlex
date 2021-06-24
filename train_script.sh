CUDA_VISIBLE_DEVICES=1,2,3,4 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --gpu 1,2,3,4 --num_gpus 4 --output output/DLA-34_Baseline_MergeHead_SCH2_4_GPUS_005nd

CUDA_VISIBLE_DEVICES=1,2,3,4 python tools/plain_train_net.py --batch_size 8 --backbone dla34 --gpu 1,2,3,4 --num_gpus 4 --output output/DLA-34_Baseline_MergeHead_SCH2_4_GPUS_006nd


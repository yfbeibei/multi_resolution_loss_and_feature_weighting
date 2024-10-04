

python -m torch.distributed.launch --nproc_per_node=2 --master_port 253512 train_distributed.py --gpu_id '1,2' \
 --gray_aug --gray_p 0.1 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr_step 1200 --lr 0.00001 \
 --batch_size 10 --num_patch 1 --threshold 0.35 --test_per_epoch 20 --num_queries 700 \
 --dataset cod --crop_size 256 --pre None --test_per_epoch 20  --test_patch --save --save_path output_feature_weighting\
 --dm_count  --dilation --branch_merge --branch_merge_way 2 --transformer_flag merge3 --decoding_arch feature_weighting
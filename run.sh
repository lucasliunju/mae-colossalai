OMP_NUM_THREADS=1
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main_pretrain.py \
    --batch_size 64 \
    --norm_pix_loss \
    --model mae_vit_large_patch16 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --num_workers 0 \
    --data_path /data/ILSVRC2012 \
    --accum_iter 8
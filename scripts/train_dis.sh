CUDA_VISIBLE_DEVICES=MIG-ad2c22f2-c9e3-5b63-a091-a543a9a9add4 \
python -m torch.distributed.launch --nproc_per_node=1 ../main.py \
--model convnext_tiny --drop_path 0.1 \
--batch_size 1024 --lr 4e-3 --layer_decay 1.0 --update_freq 4 --epoch 300 \
--model_ema true --model_ema_eval true \
--data_path ~/data/imagenet \
--output_dir /checkpoint/yangtao/convnext/exp3

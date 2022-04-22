python -m torch.distributed.launch --nproc_per_node=8 ./main.py \                                              
--model convnext_gcc_cvx_lg_tiny --drop_path 0.1 \                                                             
--batch_size 256 --lr 4e-3 --layer_decay 1.0 --update_freq 2 --epoch 300 \                                     
--model_ema true --model_ema_eval true \                                                                       
--data_path /data/imagenet \                                                                                   
--output_dir /checkpoint/cvx_lg/gcc_cvx_lg_tiny0\                                                              
> ./log/cvx_lg/cvx_lg_tiny0.txt

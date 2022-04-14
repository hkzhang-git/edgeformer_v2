CUDA_VISIBLE_DEVICES=MIG-ad2c22f2-c9e3-5b63-a091-a543a9a9add4,MIG-550adc72-4106-50c3-959b-346dc4645c14 \
python ./run_with_submitit.py --nodes 1 --ngpus 2 \
--model convnext_tiny --drop_path 0.1 \
--batch_size 256 --lr 4e-3 --update_freq 8 \
--model_ema true --model_ema_eval true \
--data_path ~/data/imagenet \
--job_dir /checkpoint/yangtao/convnext/exp4

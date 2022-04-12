CUDA_VISIBILE_DEVICES=MIG-7f60cbca-9934-5444-a400-60e1c9226a38,MIG-7082cfb3-e904-57b5-9b66-5179a9477585,MIG-4cddfa2a-8f25-5e2e-af82-1de91f8d2271,MIG-4c2b66c2-ab0a-5751-8e35-fe4880183481 \
python -m torch.distributed.launch --nproc_per_node=1 ../main.py \
--model convnext_base --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path ~/data/imagenet


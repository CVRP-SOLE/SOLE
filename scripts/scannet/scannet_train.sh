#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_QUERY=100

# TEST
CUDA_VISIBLE_DEVICES=0 python main_instance_segmentation.py \
data/datasets=scannet \
model.backbone_weight="backbone_checkpoint/backbone_scannet.ckpt" \
model.class_embed=scannet \
general.eval_on_segments=false \
general.train_on_segments=false \
general.train_mode=true \
general.num_targets=19 \
data.num_labels=20 \
data.batch_size=5 \
model.num_queries=${CURR_QUERY}
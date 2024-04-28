#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_QUERY=150

# TEST
CUDA_VISIBLE_DEVICES=0 python main_instance_segmentation.py \
data/datasets=scannet200 \
model.backbone_weight="backbone_checkpoint/backbone_scannet200.ckpt" \
model.class_embed=scannet200 \
general.eval_on_segments=false \
general.train_on_segments=false \
general.train_mode=true \
general.num_targets=199 \
data.num_labels=200 \
data.batch_size=4 \
model.num_queries=${CURR_QUERY}
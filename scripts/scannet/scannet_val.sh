#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=500
CURR_QUERY=150

# TEST
CUDA_VISIBLE_DEVICES=0 python main_instance_segmentation.py \
general.checkpoint="checkpoint/scannet.ckpt" \
data/datasets=scannet \
model.backbone_weight="backbone_checkpoint/backbone_scannet.ckpt" \
model.class_embed=scannet \
general.eval_on_segments=true \
general.train_on_segments=false \
general.train_mode=false \
general.num_targets=19 \
data.num_labels=20 \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN} \
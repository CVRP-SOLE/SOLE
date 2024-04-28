#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=750
CURR_QUERY=150

# TEST
CUDA_VISIBLE_DEVICES=0 python main_instance_segmentation.py \
general.checkpoint="checkpoint/scannet200.ckpt" \
data/datasets=scannet200 \
model.backbone_weight="backbone_checkpoint/backbone_scannet200.ckpt" \
model.class_embed=scannet200 \
general.eval_on_segments=true \
general.train_on_segments=false \
general.train_mode=false \
general.num_targets=199 \
data.num_labels=200 \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}
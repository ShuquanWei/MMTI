#!/bin/bash
# ActivityNet Captions 评测，默认加载第 16 个 epoch 的权重。

MODEL_PATH="weight/"
OUTPUT_DIR="weight/activitynet_eval_result"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m torch.distributed.launch --nproc_per_node=1 --master_port "${MASTER_PORT:-29500}" \
main_mmti.py \
--init_model "$MODEL_PATH" \
--rerantopk 50 --do_eval --num_thread_reader=8 --lr 1e-4 \
--batch_size=64 --batch_size_val 64 --epochs=5 --n_display=10 \
--data_path "/data/chenxi/wsq/Datasets/ActivityNet_Captions/raw_data" \
--features_path "/data/chenxi/wsq/Datasets/ActivityNet_Captions/raw_videos" \
--output_dir "$OUTPUT_DIR" --max_words 32 --max_frames 18 --datatype activity \
--expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 --loose_type --linear_patch 2d \
--strongpatch_weight 0.2 \
--frame_attention_temp 0.07 --patch_attention_temp 0.05 --consistency_temp 0.07 \
--rerank_video_weight 0.35 --rerank_weakframe_weight 0.25 --rerank_strongframe_weight 0.2 \
--rerank_weakpatch_weight 0.1 --rerank_strongpatch_weight 0.1 \
--enable_kfs --frame_selection_ratio 0.67 --patch_selection_ratio 0.5 \
--assignment_temperature 0.07 --kfs_context_alpha 0.15 --safe_eps 1e-6 \
--logit_scale_min 1.0 --logit_scale_max 100.0 --max_grad_norm 1.0 \
--sim_header seqTransf --pretrained_clip_name ViT-B/32

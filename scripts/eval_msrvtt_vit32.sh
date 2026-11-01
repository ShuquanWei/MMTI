#!/bin/bash
# MSR-VTT 1k-A 评测：加载 epoch 3 权重（pytorch_model.bin.2），启用 KFS。

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 \
main_mmti.py \
--init_model weight/msrvtt/pytorch_model.bin.2 \
--rerantopk 50 --do_eval --num_thread_reader=8 --lr 1e-4 \
--batch_size=64 --batch_size_val 64 --epochs=5 --n_display=10 \
--train_csv "data/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv" \
--val_csv "data/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv" \
--test_csv "data/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv" \
--multi_data_path "data/MSRVTT/msrvtt_data/MSRVTT_data.json" \
--data_path "data/MSRVTT/msrvtt_data/MSRVTT_data.json" \
--features_path "data/MSRVTT/raw_videos" \
--output_dir weight/eval_result --max_words 32 --max_frames 18 --datatype msrvtt \
--expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 --loose_type --linear_patch 2d \
--video_weight 0.7 --weakframe_weight 0.25 --weakpatch_weight 0.15 \
--strongframe_weight 0.8 --strongpatch_weight 0.2 \
--fused_loss_weight 1.0 --weak_loss_weight 0.6 --strong_loss_weight 0.25 \
--aux_loss_weight 0.05 --align_video_weight 1.0 \
--align_frame_weight 1.0 --align_patch_weight 0.6 \
--frame_attention_temp 0.07 --patch_attention_temp 0.05 --consistency_temp 0.07 \
--rerank_video_weight 0.35 --rerank_weakframe_weight 0.25 --rerank_strongframe_weight 0.2 \
--rerank_weakpatch_weight 0.1 --rerank_strongpatch_weight 0.1 \
--enable_kfs --frame_selection_ratio 0.67 --patch_selection_ratio 0.5 \
--assignment_temperature 0.07 --kfs_context_alpha 0.15 --safe_eps 1e-6 \
--logit_scale_min 1.0 --logit_scale_max 100.0 --max_grad_norm 1.0 \
--sim_header seqTransf --pretrained_clip_name ViT-B/32

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

# MMTI 文本-视频检索入口。
# KFS 依据多维显著性筛选关键帧和关键块；
# TVIM 将文本查询分解为句子、帧、块三级查询，并与关键视觉特征做置信度调制交互。

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_mmti import MMTI
from modules.optimization import BertAdam
from tqdm import tqdm
from modules.until_module import CrossEn

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

torch.distributed.init_process_group(backend="nccl")

global logger

def compute_bidir_retrieval_loss(loss_fct, sim_matrix):
    """对称 InfoNCE：文本到视频与视频到文本两个方向取平均。"""
    loss_t2v = loss_fct(sim_matrix)
    loss_v2t = loss_fct(sim_matrix.T)
    return (loss_t2v + loss_v2t) / 2

def compose_rerank_logits(args, fused_sim_matrix, weakframe_sim_matrix, strongframe_sim_matrix,
                          weakpatch_sim_matrix, strongpatch_sim_matrix):
    """按权重融合融合相似度、帧级与块级相似度，作为重排分数。"""
    return fused_sim_matrix * args.rerank_video_weight + \
        weakframe_sim_matrix * args.rerank_weakframe_weight + \
        strongframe_sim_matrix * args.rerank_strongframe_weight + \
        weakpatch_sim_matrix * args.rerank_weakpatch_weight + \
        strongpatch_sim_matrix * args.rerank_strongpatch_weight

def apply_two_stage_rerank(main_sim_matrix, rerank_sim_matrix, topk):
    """主相似度前 topk 的候选替换为融合分数并加 100 的提升量，其余候选保持原分数。"""
    if main_sim_matrix.size == 0 or topk <= 0:
        return main_sim_matrix

    reranked_matrix = main_sim_matrix.copy()
    effective_topk = min(topk, reranked_matrix.shape[1])
    topk_indices = np.argsort(reranked_matrix, axis=1)[:, -effective_topk:]

    for row_index in range(len(reranked_matrix)):
        reranked_matrix[row_index][topk_indices[row_index]] = rerank_sim_matrix[row_index][topk_indices[row_index]] + 100.0

    return reranked_matrix

def get_args(description='MMTI on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    

    # 多粒度相似度融合权重
    parser.add_argument('--video_weight', type=float, default=0.5, help='视频级相似度融合权重')
    parser.add_argument('--weakframe_weight', type=float, default=0.5, help='帧级相似度融合权重')
    parser.add_argument('--strongframe_weight', type=float, default=1.0, help='融合帧级相似度权重')
    parser.add_argument('--weakpatch_weight', type=float, default=0.05, help='块级相似度融合权重')
    parser.add_argument('--strongpatch_weight', type=float, default=0.0, help='融合块级相似度权重')
    # InfoNCE 与辅助对齐/一致性权重
    parser.add_argument('--align_video_weight', type=float, default=1.0, help='视频级对齐损失权重')
    parser.add_argument('--align_frame_weight', type=float, default=1.0, help='帧级对齐损失权重')
    parser.add_argument('--align_patch_weight', type=float, default=0.0, help='块级对齐与一致性损失权重')
    parser.add_argument('--weak_loss_weight', type=float, default=0.8, help='弱融合相似度 InfoNCE 权重')
    parser.add_argument('--strong_loss_weight', type=float, default=0.2, help='强融合相似度 InfoNCE 权重')
    parser.add_argument('--fused_loss_weight', type=float, default=1.0, help='融合相似度 InfoNCE 权重')
    parser.add_argument('--aux_loss_weight', type=float, default=0.001, help='辅助损失总权重')
    # 两阶段重排
    parser.add_argument('--rerantopk', type=int, default=50, help='参与重排的候选数')
    parser.add_argument('--rerank_video_weight', type=float, default=0.35, help='重排时融合相似度权重（沿用旧参数名）')
    parser.add_argument('--rerank_weakframe_weight', type=float, default=0.25, help='重排时帧级相似度权重')
    parser.add_argument('--rerank_strongframe_weight', type=float, default=0.2, help='重排时融合帧级相似度权重')
    parser.add_argument('--rerank_weakpatch_weight', type=float, default=0.1, help='重排时块级相似度权重')
    parser.add_argument('--rerank_strongpatch_weight', type=float, default=0.1, help='重排时融合块级相似度权重')
    # TVIM 注意力温度与丢弃率
    parser.add_argument('--frame_attention_temp', type=float, default=0.07, help='帧级置信度调制注意力的温度')
    parser.add_argument('--patch_attention_temp', type=float, default=0.05, help='块级置信度调制注意力的温度')
    parser.add_argument('--consistency_temp', type=float, default=0.07, help='一致性正则的 KL 温度')
    parser.add_argument('--tvim_dropout', type=float, default=0.1, help='句子查询的丢弃率')

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--test_csv', type=str, default='data/.test.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--multi_augmentation", action='store_true', help="multiaug augmentation")
    parser.add_argument('--multi_data_path', type=str, default='data/caption.pickle', help='data pickle file path')

    # KFS（关键特征选择）
    parser.add_argument('--enable_kfs', action='store_true',
                        help='启用帧级与块级关键特征选择')
    parser.add_argument('--frame_selection_ratio', type=float, default=0.67,
                        help='关键帧保留比例 r_frame，取值 (0,1]')
    parser.add_argument('--patch_selection_ratio', type=float, default=0.5,
                        help='关键块保留比例 r_patch，取值 (0,1]')
    parser.add_argument('--assignment_temperature', type=float, default=0.07,
                        help='软分配温度 tau_asg')
    parser.add_argument('--kfs_context_alpha', type=float, default=0.15,
                        help='关键特征与全局上下文的融合系数')
    parser.add_argument('--safe_eps', type=float, default=1e-6,
                        help='归一化与分母保护的数值稳定项')
    parser.add_argument('--logit_scale_max', type=float, default=100.0,
                        help='CLIP logit scale 上界')
    parser.add_argument('--logit_scale_min', type=float, default=1.0,
                        help='CLIP logit scale 下界')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪范数')


    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # 参数校验
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # 固定随机种子
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # 多卡时同步随机状态
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        print(args.init_model)
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # 构建模型
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = MMTI.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=args.max_grad_norm)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # 仅保存模型参数
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # 构建模型并指定缓存目录
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = MMTI.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    loss_fct = CrossEn()


    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # 多卡模式下由并行模块负责散射
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        
        multi_input_ids, multi_input_mask, multi_segment_ids, video, video_mask = batch

        input_ids = multi_input_ids[:,0]
        input_mask = multi_input_mask[:,0]
        segment_ids = multi_segment_ids[:,0]
        
        loss = 0.0
        # TVIM 前向：输出五组相似度与三项辅助损失
        tvim_output, tvim_auxiliary_losses = model(input_ids, segment_ids, input_mask, video, video_mask, multi_augmentation=False)
        sentence_video_logits, sentence_frame_weak_logits, sentence_frame_strong_logits, sentence_patches_weak_logits, sentence_patch_strong_logits = tvim_output
        video_align_loss, frame_align_loss, patch_consistency_loss = tvim_auxiliary_losses

        # 弱融合相似度由融合、帧、块三级相似度加权得到
        weak_similarity = sentence_video_logits * args.video_weight + \
                    sentence_frame_weak_logits * args.weakframe_weight + \
                    sentence_patches_weak_logits * args.weakpatch_weight

        strong_similarity = sentence_frame_strong_logits * args.strongframe_weight + sentence_patch_strong_logits * args.strongpatch_weight

        # 双向 InfoNCE
        fused_loss = compute_bidir_retrieval_loss(loss_fct, sentence_video_logits)
        weak_loss = compute_bidir_retrieval_loss(loss_fct, weak_similarity)
        strong_loss = compute_bidir_retrieval_loss(loss_fct, strong_similarity)

        # 辅助损失：两级对齐与块级一致性
        auxiliary_loss = video_align_loss * args.align_video_weight \
                    + frame_align_loss * args.align_frame_weight \
                    + patch_consistency_loss * args.align_patch_weight
        retrieval_loss = fused_loss * args.fused_loss_weight + \
                weak_loss * args.weak_loss_weight + \
                strong_loss * args.strong_loss_weight

        loss = retrieval_loss + auxiliary_loss * args.aux_loss_weight


        if n_gpu > 1:
            loss = loss.mean()  # 多卡取平均
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if not torch.isfinite(loss):
            if local_rank == 0:
                logger.warning("Skip non-finite loss at epoch %d step %d", epoch + 1, step + 1)
            optimizer.zero_grad()
            continue

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if scheduler is not None:
                scheduler.step()  # 更新学习率

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data,
                             min=np.log(args.logit_scale_min), max=np.log(args.logit_scale_max))
            else:
                torch.clamp_(model.clip.logit_scale.data,
                             min=np.log(args.logit_scale_min), max=np.log(args.logit_scale_max))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, FusedLoss: %f, WeakLoss: %f, StrongLoss: %f, ConsistencyLoss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            float(fused_loss),
                            float(weak_loss),
                            float(strong_loss),
                            float(auxiliary_loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(args, model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_seq_features_list, batch_visual_output_list, batch_visual_patches_list):
    """在单卡上计算文本-视频的多粒度相似度矩阵。

    文本特征与视觉特征已按 batch 缓存，这里按文本 batch 与视频 batch 两两组合，
    调用 TVIM 得到融合、帧级、块级及其融合版本共五组相似度。
    """
    main_sim_matrix = []
    fused_sim_matrix = []
    weakframe_sim_matrix = []
    strongframe_sim_matrix = []
    weakpatch_sim_matrix = []
    strongpatch_sim_matrix = []

    start_patch_eval = args.strongpatch_weight > 0
    for idx1, b1 in tqdm(enumerate(batch_list_t)):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        seq_features = batch_seq_features_list[idx1]
        main_each_row = []
        fused_each_row = []
        weakframe_each_row = []
        strongframe_each_row = []
        weakpatch_each_row = []
        strongpatch_each_row = []

        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            visual_patches = batch_visual_patches_list[idx2]

            # TVIM 输出顺序：融合相似度、帧级、融合帧级、块级、融合块级
            tvim_output, _ = model.get_similarity_logits(
                sequence_output, seq_features, visual_output, visual_patches,
                input_mask, video_mask, loose_type=model.loose_type,
                start_patch_eval=start_patch_eval)
            fused_logits, frame_logits, frame_fused_logits, patch_logits, patch_fused_logits = tvim_output

            fused_logits = fused_logits.cpu().detach().numpy()
            frame_logits = frame_logits.cpu().detach().numpy()
            frame_fused_logits = frame_fused_logits.cpu().detach().numpy()
            patch_logits = patch_logits.cpu().detach().numpy()
            patch_fused_logits = patch_fused_logits.cpu().detach().numpy()

            main_each_row.append(fused_logits)
            fused_each_row.append(fused_logits)
            weakframe_each_row.append(frame_logits)
            strongframe_each_row.append(frame_fused_logits)
            weakpatch_each_row.append(patch_logits)
            strongpatch_each_row.append(patch_fused_logits)

        main_each_row = np.concatenate(tuple(main_each_row), axis=-1)
        fused_each_row = np.concatenate(tuple(fused_each_row), axis=-1)
        weakframe_each_row = np.concatenate(tuple(weakframe_each_row), axis=-1)
        strongframe_each_row = np.concatenate(tuple(strongframe_each_row), axis=-1)
        weakpatch_each_row = np.concatenate(tuple(weakpatch_each_row), axis=-1)
        strongpatch_each_row = np.concatenate(tuple(strongpatch_each_row), axis=-1)

        main_sim_matrix.append(main_each_row)
        fused_sim_matrix.append(fused_each_row)
        weakframe_sim_matrix.append(weakframe_each_row)
        strongframe_sim_matrix.append(strongframe_each_row)
        weakpatch_sim_matrix.append(weakpatch_each_row)
        strongpatch_sim_matrix.append(strongpatch_each_row)

    main_sim_matrix = np.concatenate(tuple(main_sim_matrix), axis=0)
    fused_sim_matrix = np.concatenate(tuple(fused_sim_matrix), axis=0)
    weakframe_sim_matrix = np.concatenate(tuple(weakframe_sim_matrix), axis=0)
    strongframe_sim_matrix = np.concatenate(tuple(strongframe_sim_matrix), axis=0)
    weakpatch_sim_matrix = np.concatenate(tuple(weakpatch_sim_matrix), axis=0)
    strongpatch_sim_matrix = np.concatenate(tuple(strongpatch_sim_matrix), axis=0)

    return main_sim_matrix, fused_sim_matrix, weakframe_sim_matrix, strongframe_sim_matrix, weakpatch_sim_matrix, strongpatch_sim_matrix


def eval_epoch(args, model, test_dataloader, device):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    # 多句检索所需的变量
    # multi_sentence_: 评测标记
    # cut_off_points: 计算指标时的标签切分点
    # sentence_num: 句子表示数量
    # video_num: 视频表示数量
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        batch_seq_features_list = []
        batch_visual_patches_list = []
        total_video_num = 0

        # ----------------------------
        # 1. 缓存文本与视觉特征
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            if multi_sentence_:
                # 多句检索：一个视频对应多条描述
                b, *_t = video.shape
                sequence_output, seq_features = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_seq_features_list.append(seq_features)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output, visual_patches, key_video_mask = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_visual_patches_list.append(visual_patches)
                    batch_list_v.append((key_video_mask,))
                total_video_num += b
            else:
                (sequence_output, seq_features), visual_output, visual_patches, key_video_mask = model.get_sequence_visual_output(
                    input_ids, segment_ids, input_mask, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_seq_features_list.append(seq_features)
                batch_list_t.append((input_mask, segment_ids,))
                batch_visual_output_list.append(visual_output)
                batch_list_v.append((key_video_mask,))
                batch_visual_patches_list.append(visual_patches)

            logger.info("{}/{}\r".format(bid, len(test_dataloader)))

        # ----------------------------------
        # 2. 计算多粒度相似度
        # ----------------------------------
        main_sim_matrix, fused_sim_matrix, weakframe_sim_matrix, strongframe_sim_matrix, weakpatch_sim_matrix, strongpatch_sim_matrix = _run_on_single_gpu(
            args, model, batch_list_t, batch_list_v,
            batch_sequence_output_list, batch_seq_features_list,
            batch_visual_output_list, batch_visual_patches_list)

    np.save(os.path.join(args.output_dir, "sim_matrix.npy"), main_sim_matrix)
    print("start save sim_matrix {}.".format(main_sim_matrix.shape))

    # 两阶段重排：主相似度前 topk 的候选改用多粒度融合分数并加 100 的提升量
    rerank_sim_matrix = compose_rerank_logits(
        args,
        fused_sim_matrix,
        weakframe_sim_matrix,
        strongframe_sim_matrix,
        weakpatch_sim_matrix,
        strongpatch_sim_matrix,
    )
    main_sim_matrix = apply_two_stage_rerank(main_sim_matrix, rerank_sim_matrix, args.rerantopk)

    sim_matrix = main_sim_matrix
    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))


    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))

    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']+tv_metrics['R5']+tv_metrics['R10']
    return R1

def main():
    global logger
    args = get_args()
    args.output_dir = args.output_dir
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # 冻结 CLIP 的前若干层
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # 顶层始终参与训练
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # 参与训练
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # 参与训练

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # 低于 freeze_layer_num 的层冻结
                param.requires_grad = False

    ## ####################################
    # 数据加载
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    # test 为空时使用 val 结果
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length
    
    if args.rank == 0:
        logger.info("***** Weight *****")
        logger.info("video_weight = {}".format(args.video_weight))
        logger.info("weakframe_weight = {}".format(args.weakframe_weight))
        logger.info("strongframe_weight = {}".format(args.strongframe_weight))
        logger.info("weakpatch_weight = {}".format(args.weakpatch_weight))
        logger.info("strongpatch_weight = {}".format(args.strongpatch_weight))
        logger.info("align_video_weight = {}".format(args.align_video_weight))
        logger.info("align_frame_weight = {}".format(args.align_frame_weight))
        logger.info("align_patch_weight = {}".format(args.align_patch_weight))
        logger.info("fused_loss_weight = {}".format(args.fused_loss_weight))
        logger.info("weak_loss_weight = {}".format(args.weak_loss_weight))
        logger.info("strong_loss_weight = {}".format(args.strong_loss_weight))
        logger.info("aux_loss_weight = {}".format(args.aux_loss_weight))
        logger.info("rerank_video_weight = {}".format(args.rerank_video_weight))
        logger.info("rerank_weakframe_weight = {}".format(args.rerank_weakframe_weight))
        logger.info("rerank_strongframe_weight = {}".format(args.rerank_strongframe_weight))
        logger.info("rerank_weakpatch_weight = {}".format(args.rerank_weakpatch_weight))
        logger.info("rerank_strongpatch_weight = {}".format(args.rerank_strongpatch_weight))
        logger.info("enable_kfs = {}".format(args.enable_kfs))
        logger.info("frame_selection_ratio = {}".format(args.frame_selection_ratio))
        logger.info("patch_selection_ratio = {}".format(args.patch_selection_ratio))
        logger.info("assignment_temperature = {}".format(args.assignment_temperature))
        logger.info("kfs_context_alpha = {}".format(args.kfs_context_alpha))
        logger.info("safe_eps = {}".format(args.safe_eps))
        logger.info("logit_scale_min = {}".format(args.logit_scale_min))
        logger.info("logit_scale_max = {}".format(args.logit_scale_max))
        logger.info("max_grad_norm = {}".format(args.max_grad_norm))
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)


    ## ####################################
    # 训练与评测
    ## ####################################
    # if args.rank == 0:
    #     eval_epoch(args, model, test_dataloader, device, n_gpu)
    if args.do_train:
        if args.multi_augmentation:
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train_multi"](args, tokenizer)
        else:
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        # train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # 恢复优化器状态以继续训练
        ## ##############################################################
        logger.info("start resume model ")
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        logger.info("over")
        
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                # 在验证集上选择最优模型；某些数据集（如 DiDeMo）没有发布验证集视频
                if len(val_dataloader.dataset) > 0:
                    logger.info("Eval on val dataset")
                    R1 = eval_epoch(args, model, val_dataloader, device)

                    if best_score <= R1:
                        best_score = R1
                        best_output_model_file = output_model_file
                else:
                    logger.info("Validation set is empty; skip validation and keep the latest checkpoint.")
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

        # 在最优检查点上评测测试集
        if args.rank == 0:
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            eval_epoch(args, model, test_dataloader, device)

    elif args.do_eval:
        if args.rank == 0:
            eval_epoch(args, model, test_dataloader, device)

if __name__ == "__main__":
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from modules.modeling import MMTIPreTrainedModel, show_log, update_attr, check_attr

# from modules.differential_topk import PerturbedTopK

logger = logging.getLogger(__name__)
allgather = AllGather.apply


class MMTI(MMTIPreTrainedModel):
    """MMTI 模型：CLIP 编码、KFS 关键特征选择与 TVIM 多粒度交互。"""

    def __init__(self, cross_config, clip_state_dict, task_config):
        super(MMTI, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))
        
        #########
        self.patches = (image_resolution//vision_patch_size)**2
        ########

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP' #"seqTransf"
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        num_words = task_config.max_words
        num_frames = self.task_config.max_frames

        # 帧级特征使用 CLIP 原始输出，保留未经过时序编码的局部判别性
        self.use_original_clip_for_frame_features = True

        # KFS：关键帧与关键块选择
        self.enable_kfs = bool(getattr(task_config, "enable_kfs", False))
        self.frame_selection_ratio = float(getattr(task_config, "frame_selection_ratio", 1.0))
        self.patch_selection_ratio = float(getattr(task_config, "patch_selection_ratio", 1.0))
        self.frame_selection_ratio = min(max(self.frame_selection_ratio, 0.0), 1.0)
        self.patch_selection_ratio = min(max(self.patch_selection_ratio, 0.0), 1.0)
        self.assignment_temperature = float(getattr(task_config, "assignment_temperature", 0.07))
        self.kfs_context_alpha = float(getattr(task_config, "kfs_context_alpha", 0.15))
        self.safe_eps = float(getattr(task_config, "safe_eps", 1e-6))
        self.logit_scale_max = float(getattr(task_config, "logit_scale_max", 100.0))
        self.logit_scale_min = float(getattr(task_config, "logit_scale_min", 1.0))
        show_log(task_config, "\t enable_kfs: {}".format(self.enable_kfs))
        show_log(task_config, "\t frame_selection_ratio: {}".format(self.frame_selection_ratio))
        show_log(task_config, "\t patch_selection_ratio: {}".format(self.patch_selection_ratio))
        show_log(task_config, "\t assignment_temperature: {}".format(self.assignment_temperature))
        show_log(task_config, "\t kfs_context_alpha: {}".format(self.kfs_context_alpha))

        # 视频级相似度的可学习投影
        self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)

        # 旧版帧级投影参数，保留用于检查点兼容；当前前向不使用
        # self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)

        # TVIM 注意力温度、一致性 KL 温度与句子查询丢弃率
        self.frame_attention_temp = getattr(task_config, "frame_attention_temp", 0.07)
        self.patch_attention_temp = getattr(task_config, "patch_attention_temp", 0.05)
        self.consistency_temp = getattr(task_config, "consistency_temp", 0.07)
        self.tvim_dropout = nn.Dropout(getattr(task_config, "tvim_dropout", 0.1))

        # TVIM 文本查询分解：门控融合、帧/块查询投影与置信度预测
        self.text_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.text_norm = nn.LayerNorm(embed_dim)
        self.frame_query_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.patch_query_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.frame_confidence = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )
        self.patch_confidence = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )
        # 多粒度融合的可学习偏置
        self.coarse_score_bias = nn.Parameter(torch.tensor([0.0, 0.15, 0.05], dtype=torch.float))
        self.strong_score_bias = nn.Parameter(torch.tensor([0.1, 0.0], dtype=torch.float))

        # 预留的细粒度投影参数
        # self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        # self.frame_mat_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        # self.word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        # self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        # self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)

        self.loss_fct = CrossEn()
        # self.visual_token_selector = PerturbedTopK(10)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,multi_augmentation=False):
        if multi_augmentation:
            input_ids = input_ids.squeeze(2)
            token_type_ids = token_type_ids.squeeze(2)
            attention_mask = attention_mask.squeeze(2)
            video_mask = video_mask.view(-1, video_mask.shape[-1])
        else:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # [bs, 1, dim], [bs, num_words, dim], [bs, num_frames, dim]

        if multi_augmentation:
            pass
        else:
            (sequence_output, seq_features), visual_output, visual_patches, video_mask = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask, 
                                                                    video, video_mask, shaped=True, video_frame=video_frame,multi_augmentation=False)

            if self.training:
                tvim_output,tvim_auxiliary_losses = self.get_similarity_logits(sequence_output, seq_features, visual_output, visual_patches, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)
                return tvim_output,tvim_auxiliary_losses
            else:
                return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden, seq_features = self.clip.encode_text(input_ids, return_hidden=True)
        sequence_hidden, seq_features = sequence_hidden.float(), seq_features.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden, seq_features

    def _safe_l2_normalize(self, x, dim=-1):
        x = self._safe_nan_to_num(x)
        return F.normalize(x, dim=dim, eps=self.safe_eps)

    def _safe_nan_to_num(self, x):
        if hasattr(torch, "nan_to_num"):
            return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    def _safe_softmax(self, x, dim=-1, temperature=1.0):
        x = self._safe_nan_to_num(x)
        x = x / max(temperature, self.safe_eps)
        x = torch.clamp(x, min=-50.0, max=50.0)
        return torch.softmax(x, dim=dim)

    def _topk_sorted_indices(self, score, k):
        _, idx = torch.topk(score, k=k, dim=1)
        idx, _ = torch.sort(idx, dim=1)
        return idx

    def _frame_importance_scores(self, frame_features, video_mask):
        """KFS 帧级重要性：新颖性、语义区分度与能量标准差。"""
        video_mask = video_mask.float()
        normed = self._safe_l2_normalize(frame_features, dim=-1)
        prev = torch.roll(normed, shifts=1, dims=1)
        prev[:, 0, :] = normed[:, 0, :]
        novelty = (1.0 - F.cosine_similarity(normed, prev, dim=-1)).clamp(min=0.0)
        center = normed.mean(dim=1, keepdim=True)
        semantic = (1.0 - F.cosine_similarity(normed, center, dim=-1)).clamp(min=0.0)
        energy = (frame_features - frame_features.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1).sqrt()
        importance = 0.55 * novelty + 0.35 * semantic + 0.10 * energy
        importance = importance * video_mask + video_mask * 1e-6
        return importance

    def _select_key_frames(self, frame_features, video_mask):
        """选择重要性最高的 K 个锚点帧，并以软分配聚合全部帧特征。"""
        if (not self.enable_kfs) or self.frame_selection_ratio >= 1.0:
            return frame_features, video_mask

        num_frames = frame_features.size(1)
        keep_frames = max(1, int(round(num_frames * self.frame_selection_ratio)))
        if keep_frames >= num_frames:
            return frame_features, video_mask

        frame_features = self._safe_nan_to_num(frame_features)
        importance = self._frame_importance_scores(frame_features, video_mask)

        # 选出 K 个锚点帧
        masked_importance = importance.masked_fill(video_mask <= 0, -1e4)
        topk_idx = self._topk_sorted_indices(masked_importance, keep_frames)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, frame_features.size(-1))
        selected = torch.gather(frame_features, dim=1, index=gather_idx)

        # 按余弦相似度计算软分配并加权聚合
        norm_all = self._safe_l2_normalize(frame_features, dim=-1)
        norm_sel = self._safe_l2_normalize(selected, dim=-1)
        sim = torch.matmul(norm_all, norm_sel.transpose(1, 2))
        assign = self._safe_softmax(sim, dim=-1, temperature=self.assignment_temperature)
        assign = assign * video_mask.unsqueeze(-1)

        merged_num = torch.matmul(assign.transpose(1, 2), frame_features)
        merged_den = assign.sum(dim=1).unsqueeze(-1).clamp_min(self.safe_eps)
        merged = merged_num / merged_den

        # 混入全局视频上下文，保留整体语义
        global_ctx = self._mean_pooling_for_similarity_visual(frame_features, video_mask)
        merged = (1.0 - self.kfs_context_alpha) * merged + self.kfs_context_alpha * global_ctx.unsqueeze(1)

        valid_video = (video_mask.sum(dim=1, keepdim=True) > 0).to(video_mask.dtype)
        key_frame_mask = valid_video.repeat(1, keep_frames)
        merged = self._safe_nan_to_num(merged)
        return merged, key_frame_mask

    def _select_key_patches(self, visual_patches):
        """逐帧选择重要性最高的 K' 个锚点块，并以软分配聚合块特征。"""
        if (not self.enable_kfs) or self.patch_selection_ratio >= 1.0:
            return visual_patches

        bsz, num_frames, num_patches, dim = visual_patches.shape
        keep_patches = max(1, int(round(num_patches * self.patch_selection_ratio)))
        if keep_patches >= num_patches:
            return visual_patches

        visual_patches = self._safe_nan_to_num(visual_patches)
        patches = visual_patches.view(bsz * num_frames, num_patches, dim)
        patch_center = patches.mean(dim=1, keepdim=True)
        norm_patch = self._safe_l2_normalize(patches, dim=-1)
        norm_center = self._safe_l2_normalize(patch_center, dim=-1)
        patch_sim = F.cosine_similarity(norm_patch, norm_center, dim=-1)
        patch_energy = (patches - patch_center).pow(2).mean(dim=-1).sqrt()
        # 块级重要性：与帧内均值的语义差异及能量标准差
        importance = (1.0 - patch_sim).clamp(min=0.0) + 0.2 * patch_energy + 1e-6

        topk_idx = torch.topk(importance, k=keep_patches, dim=1).indices
        topk_idx, _ = torch.sort(topk_idx, dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, dim)
        selected = torch.gather(patches, dim=1, index=gather_idx)

        sim = torch.matmul(norm_patch, self._safe_l2_normalize(selected, dim=-1).transpose(1, 2))
        assign = self._safe_softmax(sim, dim=-1, temperature=self.assignment_temperature)
        merged_num = torch.matmul(assign.transpose(1, 2), patches)
        merged_den = assign.sum(dim=1).unsqueeze(-1).clamp_min(self.safe_eps)
        merged = merged_num / merged_den
        merged = (1.0 - self.kfs_context_alpha) * merged + self.kfs_context_alpha * patch_center.expand_as(merged)
        merged = merged.view(bsz, num_frames, keep_patches, dim)
        merged = self._safe_nan_to_num(merged)
        return merged

    def _align_patch_sequence(self, visual_patches, target_frames):
        """将块序列的帧维重采样到 KFS 后的帧数。"""
        if visual_patches.size(1) == target_frames:
            return visual_patches

        bsz, num_frames, num_patches, dim = visual_patches.shape
        patches = visual_patches.permute(0, 2, 3, 1).contiguous().view(bsz * num_patches * dim, 1, num_frames)
        patches = F.adaptive_avg_pool1d(patches, target_frames)
        patches = patches.view(bsz, num_patches, dim, target_frames).permute(0, 3, 1, 2).contiguous()
        return patches

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        """提取视觉特征；启用 KFS 时依次筛选关键帧与关键块。"""
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden, visual_patches = self.clip.encode_image(video, video_frame=video_frame,return_hidden=True)

        visual_hidden = visual_hidden.float()
        visual_patches = visual_patches.float()

        # 去掉 CLS token，保留每个块的视觉特征
        visual_patches = visual_patches[:, 1:, :]
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        visual_patches = visual_patches.view(bs_pair, -1, self.patches, visual_hidden.size(-1))
        key_video_mask = video_mask
        if self.enable_kfs:
            # 关键帧选择后，块序列的帧维需与关键帧数一致
            visual_hidden, key_video_mask = self._select_key_frames(visual_hidden, video_mask)
            visual_patches = self._align_patch_sequence(visual_patches, visual_hidden.size(1))
            visual_patches = self._select_key_patches(visual_patches)
        return visual_hidden, visual_patches, key_video_mask

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1,multi_augmentation=False):
        if multi_augmentation:
            if shaped is False:
                input_ids = input_ids.squeeze(2)
                token_type_ids = token_type_ids.squeeze(2)
                attention_mask = attention_mask.squeeze(2)

                video_mask = video_mask.view(-1, video_mask.shape[-1])
                video = torch.as_tensor(video).float()
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)
                video_frame = bs * ts
            B,multi,dim = input_ids.shape
            sequence_output_list = []
            seq_features_list = []
            for i in range(multi):
                sequence_output, seq_features = self.get_sequence_output(input_ids[:,i], token_type_ids[:,i], attention_mask[:,i], shaped=True)
                sequence_output_list.append(sequence_output)
                seq_features_list.append(seq_features)
            
            visual_output, visual_patches, key_video_mask = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame) 

            return (sequence_output_list, seq_features_list), visual_output, visual_patches, key_video_mask

        else:
            if shaped is False:
                input_ids = input_ids.view(-1, input_ids.shape[-1])
                token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
                attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
                video_mask = video_mask.view(-1, video_mask.shape[-1])

                video = torch.as_tensor(video).float()
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)
                video_frame = bs * ts

            sequence_output, seq_features = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True) # [bs, 1, dim], [bs, num_words, dim]
            visual_output, visual_patches, key_video_mask = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)                  # [bs, num_frames, dim]

            return (sequence_output, seq_features), visual_output, visual_patches, key_video_mask

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    @staticmethod
    def _normalize_feature(feature, dim=-1, eps=1e-6):
        return feature / feature.norm(dim=dim, keepdim=True).clamp(min=eps)

    @staticmethod
    def _masked_softmax(similarity, mask, dim=-1):
        mask = mask.to(dtype=similarity.dtype)
        while mask.dim() < similarity.dim():
            mask = mask.unsqueeze(0)

        masked_similarity = similarity.masked_fill(mask <= 0, -1e4)
        attention = torch.softmax(masked_similarity, dim=dim)
        attention = attention * mask
        return attention / attention.sum(dim=dim, keepdim=True).clamp(min=1e-6)

    def _decompose_text_queries(self, sequence_output, seq_features, attention_mask):
        sentence_output = self._normalize_feature(sequence_output.squeeze(1))
        word_features = self._normalize_feature(seq_features)

        token_mask = attention_mask.to(dtype=word_features.dtype).unsqueeze(-1)
        token_mask[:, 0, :] = 0.
        token_denom = token_mask.sum(dim=1).clamp(min=1.0)
        token_summary = (word_features * token_mask).sum(dim=1) / token_denom

        gate = torch.sigmoid(self.text_gate(torch.cat((sentence_output, token_summary), dim=-1)))
        sentence_output = self.text_norm(sentence_output + gate * token_summary)
        sentence_output = self._normalize_feature(self.tvim_dropout(sentence_output))

        frame_query = self._normalize_feature(self.frame_query_proj(torch.cat((sentence_output, token_summary), dim=-1)))
        patch_query = self._normalize_feature(self.patch_query_proj(torch.cat((token_summary, sentence_output), dim=-1)))
        return sentence_output, frame_query, patch_query

    def _pairwise_context_logits(self, sentence_output, context_feature, logit_scale):
        context_feature = self._normalize_feature(context_feature)
        return logit_scale * torch.einsum('td,tbd->tb', sentence_output, context_feature)

    def _tvim_similarity(self, sequence_output, seq_features, visual_output, visual_patches, attention_mask, video_mask, sim_header="meanP", start_patch_eval=False):
        """TVIM 多粒度交互。

        sequence_output: 文本 CLS 特征 [bs, 1, dim]
        seq_features:    文本词级特征 [bs, num_words, dim]
        visual_output:   KFS 后的关键帧特征 [bs, num_frames, dim]
        visual_patches:  KFS 后的关键块特征 [bs, num_frames, num_patches, dim]

        返回五组相似度，顺序为：融合、帧级、融合帧级、块级、融合块级。
        """
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        visual_patches = visual_patches.contiguous()

        # 帧序列经 Transformer 编码时序上下文，并保留残差
        visual_output_original = visual_output
        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        visual_output = visual_output + frame_position_embeddings

        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)

        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        visual_output = visual_output + visual_output_original

        visual_patches = self._normalize_feature(visual_patches)
        visual_output_norm = self._normalize_feature(visual_output)
        video_output = self._mean_pooling_for_similarity_visual(visual_output_norm, video_mask)
        video_output = self._normalize_feature(video_output)

        if self.use_original_clip_for_frame_features:
            frame_features = self._normalize_feature(visual_output_original)
        else:
            frame_features = self._normalize_feature(visual_output)

        # 文本查询分解：句子查询 t、帧查询 q_frame、块查询 q_patch
        sentence_query, frame_query, patch_query = self._decompose_text_queries(sequence_output, seq_features, attention_mask)

        logit_scale = torch.clamp(self.clip.logit_scale,
                      min=math.log(self.logit_scale_min),
                      max=math.log(self.logit_scale_max)).exp()

        if self.training:
            video_output = allgather(video_output, self.task_config)
            frame_features = allgather(frame_features, self.task_config)
            sentence_query = allgather(sentence_query, self.task_config)
            frame_query = allgather(frame_query, self.task_config)
            patch_query = allgather(patch_query, self.task_config)
            visual_output_norm = allgather(visual_output_norm, self.task_config)
            visual_patches = allgather(visual_patches, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            torch.distributed.barrier()

        # 视频级相似度
        video_logits = logit_scale * torch.matmul(torch.matmul(sentence_query, self.global_mat_weight), video_output.t())

        # 帧级置信度调制注意力
        frame_scores = torch.einsum('td,bfd->tbf', frame_query, frame_features)
        frame_confidence = torch.sigmoid(self.frame_confidence(frame_features)).squeeze(-1).unsqueeze(0)
        frame_scores = frame_scores / self.frame_attention_temp + torch.log(frame_confidence.clamp(min=1e-6))
        frame_attention = self._masked_softmax(frame_scores, video_mask.unsqueeze(0), dim=-1)
        frame_context = torch.einsum('tbf,bfd->tbd', frame_attention, frame_features)
        frame_logits = self._pairwise_context_logits(sentence_query, frame_context, logit_scale)

        # 块级置信度调制注意力
        batch_size, frame_num, patch_num, dim = visual_patches.shape
        patch_features = visual_patches.reshape(batch_size, frame_num * patch_num, dim)
        patch_mask = video_mask.unsqueeze(-1).expand(-1, -1, patch_num).reshape(batch_size, frame_num * patch_num)
        patch_scores = torch.einsum('td,bkd->tbk', patch_query, patch_features)
        patch_confidence = torch.sigmoid(self.patch_confidence(patch_features)).squeeze(-1).unsqueeze(0)
        patch_scores = patch_scores / self.patch_attention_temp + torch.log(patch_confidence.clamp(min=1e-6))
        patch_attention = self._masked_softmax(patch_scores, patch_mask.unsqueeze(0), dim=-1)
        patch_context = torch.einsum('tbk,bkd->tbd', patch_attention, patch_features)
        patch_logits = self._pairwise_context_logits(sentence_query, patch_context, logit_scale)

        # 视频、帧、块三级上下文自适应融合
        video_context = video_output.unsqueeze(0).expand(sentence_query.size(0), -1, -1)
        coarse_scores = torch.stack((video_logits, frame_logits, patch_logits), dim=-1)
        coarse_weights = torch.softmax(coarse_scores + self.coarse_score_bias.view(1, 1, -1), dim=-1)
        fused_context = self._normalize_feature(
            coarse_weights[..., 0:1] * video_context +
            coarse_weights[..., 1:2] * frame_context +
            coarse_weights[..., 2:3] * patch_context
        )
        fused_logits = self._pairwise_context_logits(sentence_query, fused_context, logit_scale)

        # 帧级与块级上下文的融合版本，作为辅助监督
        strong_scores = torch.stack((frame_logits, patch_logits), dim=-1)
        strong_weights = torch.softmax(strong_scores + self.strong_score_bias.view(1, 1, -1), dim=-1)
        strong_local_context = self._normalize_feature(
            strong_weights[..., 0:1] * frame_context +
            strong_weights[..., 1:2] * patch_context
        )
        frame_fused_context = self._normalize_feature(0.6 * frame_context + 0.4 * fused_context)
        frame_fused_logits = self._pairwise_context_logits(sentence_query, frame_fused_context, logit_scale)

        if self.training or start_patch_eval:
            patch_fused_context = self._normalize_feature(0.6 * patch_context + 0.4 * strong_local_context)
            patch_fused_logits = self._pairwise_context_logits(sentence_query, patch_fused_context, logit_scale)
        else:
            patch_fused_context = strong_local_context
            patch_fused_logits = frame_fused_logits

        if self.training:
            # 正样本上下文与句子查询的余弦对齐
            diag_index = torch.arange(sentence_query.size(0), device=sentence_query.device)
            positive_video_context = fused_context[diag_index, diag_index]
            positive_frame_context = frame_fused_context[diag_index, diag_index]
            positive_patch_context = patch_fused_context[diag_index, diag_index]

            video_align_loss = (1.0 - (sentence_query * positive_video_context).sum(dim=-1)).mean()
            frame_align_loss = (1.0 - (sentence_query * positive_frame_context).sum(dim=-1)).mean()
            patch_align_loss = (1.0 - (sentence_query * positive_patch_context).sum(dim=-1)).mean()

            # 以帧级与块级融合相似度的均值为教师，约束融合相似度的行/列分布
            strong_teacher_logits = 0.5 * (frame_fused_logits.detach() + patch_fused_logits.detach())
            row_consistency = F.kl_div(
                F.log_softmax(fused_logits / self.consistency_temp, dim=-1),
                F.softmax(strong_teacher_logits / self.consistency_temp, dim=-1),
                reduction='batchmean'
            )
            col_consistency = F.kl_div(
                F.log_softmax(fused_logits.t() / self.consistency_temp, dim=-1),
                F.softmax(strong_teacher_logits.t() / self.consistency_temp, dim=-1),
                reduction='batchmean'
            )
            patch_consistency_loss = patch_align_loss + 0.5 * (row_consistency + col_consistency)

            tvim_output = [fused_logits, frame_logits, frame_fused_logits, patch_logits, patch_fused_logits]
            tvim_auxiliary_losses = [video_align_loss, frame_align_loss, patch_consistency_loss]
            return tvim_output, tvim_auxiliary_losses

        tvim_output = [fused_logits, frame_logits, frame_fused_logits, patch_logits, patch_fused_logits]
        tvim_auxiliary_losses = [[], [], []]
        return tvim_output, tvim_auxiliary_losses


    def get_similarity_logits(self, sequence_output, seq_features, visual_output, visual_patches, attention_mask, video_mask, shaped=False, loose_type=False, start_patch_eval=False):
        """计算 TVIM 的多粒度相似度。"""
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        assert loose_type and self.sim_header in ["seqTransf"]
        return self._tvim_similarity(
            sequence_output, seq_features, visual_output, visual_patches,
            attention_mask, video_mask, sim_header=self.sim_header,
            start_patch_eval=start_patch_eval)

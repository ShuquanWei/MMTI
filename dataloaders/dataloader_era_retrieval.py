from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.rawframe_util import RawVideoExtractor
from tqdm import tqdm

class ERA_DataLoader(Dataset):
    """ERA dataset loader."""
    def __init__(
            self,
            subset,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos
        self.subset = subset
        
        # Load JSON data
        print(f"Loading ERA {subset} data from {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Check JSON structure
        if "ERA_caption" not in self.data:
            raise ValueError(f"JSON file {json_path} does not contain 'ERA_caption' key")
        
        # Build video-caption pairs
        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        video_ids_set = set()
        
        for idx, item in enumerate(self.data["ERA_caption"]):
            if "video_id" not in item or "annotation" not in item:
                print(f"Skipping item {idx}: missing required keys")
                continue
                
            video_id = item["video_id"]
            video_ids_set.add(video_id)
            
            # Get captions
            if "English_caption" not in item["annotation"]:
                print(f"Skipping video {video_id}: no English captions")
                continue
                
            captions = item["annotation"]["English_caption"]
            
            # Store each caption as separate sample
            start_idx = len(self.sentences_dict)
            for caption in captions:
                self.sentences_dict[len(self.sentences_dict)] = (video_id, caption)
            self.cut_off_points.append(len(self.sentences_dict))
        
        if len(self.sentences_dict) == 0:
            raise ValueError(f"No valid data loaded from {json_path}")
        
        # For multi-sentence retrieval evaluation
        self.multi_sentence_per_video = True
        self.sentence_num = len(self.sentences_dict)
        self.video_num = len(video_ids_set)
        
        print(f"For ERA {subset}, sentence number: {self.sentence_num}")
        print(f"For ERA {subset}, video number: {self.video_num}")
        
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sentence_num

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(str(sentence))

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # ERA dataset structure: Videos/{subset}/Category/video_id
            # Extract category from video_id (e.g., "Baseball_001.mp4" -> "Baseball")
            category = video_id.split('_')[0]
            video_subdir = "Test" if self.subset == "test" or self.subset == "val" else "Tra"
            video_path = os.path.join(self.features_path, video_subdir, category, video_id)
            
            # Check for different video extensions
            if not os.path.exists(video_path):
                # Try without extension
                video_path_no_ext = video_path.rsplit('.', 1)[0]
                if os.path.exists(video_path_no_ext):
                    video_path = video_path_no_ext
                else:
                    print(f'Video path = {video_path} does not exist.')
                    continue

            try:
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
                if raw_video_data is None:
                    print(f"Failed to extract video data: {video_path}")
                    continue
                    
                raw_video_data = raw_video_data['video']
                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, 
                                                     num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(
                        video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max(max_video_length[i], slice_len)
                    if slice_len > 0:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print(f"Video shape error: {video_path}, shape: {raw_video_data.shape}")
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")

        for i, v_length in enumerate(max_video_length):
            if v_length > 0:
                video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask


class ERA_TrainDataLoader(Dataset):
    """ERA train dataset loader."""
    def __init__(
            self,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos
        self.unfold_sentences = unfold_sentences

        # Load JSON data
        print(f"Loading ERA training data from {json_path}")
        self.data = json.load(open(json_path, 'r'))
        
        # Build video-caption mapping
        self.video_captions = defaultdict(list)
        for item in self.data["ERA_caption"]:
            video_id = item["video_id"]
            captions = item["annotation"]["English_caption"]
            self.video_captions[video_id].extend(captions)
        
        # Create samples
        self.sample_len = 0
        self.sentences_dict = {}
        if self.unfold_sentences:
            for video_id, captions in self.video_captions.items():
                for caption in captions:
                    self.sentences_dict[len(self.sentences_dict)] = (video_id, caption)
            self.sample_len = len(self.sentences_dict)
        else:
            self.video_ids = list(self.video_captions.keys())
            self.sample_len = len(self.video_ids)
        
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(str(caption))
            else:
                # Randomly select a caption for this video
                captions = self.video_captions[video_id]
                selected_caption = random.choice(captions)
                words = self.tokenizer.tokenize(str(selected_caption))

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # ERA dataset structure for training
            category = video_id.split('_')[0]
            video_path = os.path.join(self.features_path, "Tra", category, video_id)
            
            if not os.path.exists(video_path):
                video_path_no_ext = video_path.rsplit('.', 1)[0]
                if os.path.exists(video_path_no_ext):
                    video_path = video_path_no_ext
                else:
                    print(f'Video path = {video_path} does not exist.')
                    continue

            try:
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
                if raw_video_data is None:
                    continue
                    
                raw_video_data = raw_video_data['video']
                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, 
                                                     num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(
                        video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max(max_video_length[i], slice_len)
                    if slice_len > 0:
                        video[i][:slice_len, ...] = video_slice
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")

        for i, v_length in enumerate(max_video_length):
            if v_length > 0:
                video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id = self.video_ids[idx]
            caption = None
            
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask

class ERA_TrainDataLoader_Multi(Dataset):
    """ERA train dataset loader with multiple augmentation."""
    def __init__(
            self,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos
        self.unfold_sentences = unfold_sentences

        # 处理多个JSON文件路径（用"+"分隔）
        json_path_list = json_path.strip().split("+")
        print(f"Loading ERA multi augmentation data from {len(json_path_list)} JSON files")

        # 加载多个JSON文件
        self.data_list = [json.load(open(x, 'r')) for x in json_path_list]

        self.sample_len = 0
        if self.unfold_sentences:
            # 为每个JSON文件构建sentences_dict（展开每个caption为独立样本）
            self.sentences_dict_list = []
            self.video_ids_set = set()

            for data_idx, data in enumerate(self.data_list):
                sentences_dict = {}
                if "ERA_caption" not in data:
                    print(f"Warning: JSON file {json_path_list[data_idx]} does not contain 'ERA_caption' key")
                    continue

                for item in data["ERA_caption"]:
                    if "video_id" not in item or "annotation" not in item:
                        continue

                    video_id = item["video_id"]
                    self.video_ids_set.add(video_id)

                    if "English_caption" not in item["annotation"]:
                        continue

                    captions = item["annotation"]["English_caption"]

                    # 展开每个caption作为独立样本（5 captions → 5 samples）
                    for caption in captions:
                        sentences_dict[len(sentences_dict)] = (video_id, caption)

                self.sentences_dict_list.append(sentences_dict)
                print(f"JSON {data_idx}: {len(sentences_dict)} sentences")

            # 确保所有JSON文件有相同的样本数
            sample_lengths = [len(d) for d in self.sentences_dict_list]
            if len(set(sample_lengths)) > 1:
                print(f"Warning: JSON files have different sample counts: {sample_lengths}")
                # 取最小长度作为公共长度
                self.sample_len = min(sample_lengths)
            else:
                self.sample_len = sample_lengths[0] if sample_lengths else 0

            print(f"Final sample length: {self.sample_len}")
            print(f"Unique video IDs: {len(self.video_ids_set)}")
        else:
            raise ValueError(
                "ERA_TrainDataLoader_Multi requires --expand_msrvtt_sentences to be set. "
                "Multi-augmentation mode needs sentences to be expanded into individual samples."
            )

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(str(caption))
            else:
                # 如果没有提供caption，随机选择一个（但这里应该不会发生）
                words = self.tokenizer.tokenize("")

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # ERA数据集结构
            category = video_id.split('_')[0]
            video_path = os.path.join(self.features_path, "Tra", category, video_id)
            
            if not os.path.exists(video_path):
                video_path_no_ext = video_path.rsplit('.', 1)[0]
                if os.path.exists(video_path_no_ext):
                    video_path = video_path_no_ext
                else:
                    print(f'Video path = {video_path} does not exist.')
                    continue

            try:
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
                if raw_video_data is None:
                    continue
                    
                raw_video_data = raw_video_data['video']
                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, 
                                                     num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(
                        video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max(max_video_length[i], slice_len)
                    if slice_len > 0:
                        video[i][:slice_len, ...] = video_slice
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")

        for i, v_length in enumerate(max_video_length):
            if v_length > 0:
                video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        multi_pairs_text = []
        multi_pairs_mask = []
        multi_pairs_segment = []

        # 遍历所有JSON文件的句子字典
        for sentences_dict in self.sentences_dict_list:
            if self.unfold_sentences:
                video_id, caption = sentences_dict[idx]
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)

            multi_pairs_text.append(pairs_text)
            multi_pairs_mask.append(pairs_mask)
            multi_pairs_segment.append(pairs_segment)

        # 堆叠多个文本描述
        multi_pairs_text = np.stack(multi_pairs_text, 0)      # shape: (num_json, 1, max_words)
        multi_pairs_mask = np.stack(multi_pairs_mask, 0)
        multi_pairs_segment = np.stack(multi_pairs_segment, 0)

        # 视频特征（多个文本对应同一个视频）
        video, video_mask = self._get_rawvideo(choice_video_ids)

        return multi_pairs_text, multi_pairs_mask, multi_pairs_segment, video, video_mask
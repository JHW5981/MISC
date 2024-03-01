import logging
import math
import os
from typing import List, Optional, Union
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from timm.models.swin_transformer import SwinTransformer
from torchvision.transforms.functional import resize, rotate
from transformers import (
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    MBartConfig,
    MBartForCausalLM,
)

from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from nougat.postprocessing import postprocess
from nouagt.transforms import train_transform, test_transform

class SwinEncoder(nn.Module):
    r"""
    Encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations

    Args:
        input_size: Input image size (width, height)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """
    def __init__(
            self,
            input_size: List[int],
            align_long_axis: bool,
            window_size: int,
            encoder_layer: List[int],
            patch_size: int,
            embed_dim: int,
            num_heads: List[int],
            name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.model = SwinTransformer(
            img_size=self.input_size,
            depth=self.encoder_layer,
            window_size=self.window_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_classes=0,
        )
        
        # weight init with swin
        if not name_or_path:
            swin_state_dict = timm.create_model(
                "swin_base_patch4_window12_384", pretrained=True
            ).state_dict()
            new_swin_state_dict = self.model.state_dict()
            for x in new_swin_state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(math.sqrt(len(pos_bias)))
                    new_len = int(2 * window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(
                        0, 3, 1, 2
                    )
                    pos_bias = F.interpolate(
                        pos_bias,
                        size=(new_len, new_len),
                        mode="bicubic",
                        align_corners=False,
                    )
                    new_swin_state_dict[x] = (
                        pos_bias.permute(0, 2, 3, 1)
                        .reshape(1, new_len**2, -1)
                        .squeeze(0)
                    )
                else:
                    new_swin_state_dict[x] = swin_state_dict[x]
            self.model.load_state_dict(new_swin_state_dict)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        return x
    
    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.unit8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) *255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray) # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        return img.crop((a, b, a + w, b + h))

    @property
    def to_tensor(self):
        if self.trainig:
            return train_transform
        else:
            return test_transform
        
    def prepare_input(
            self,
            img: Image.Image,
            random_padding: bool = False
    ) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            return
        # crop margins
        try:
            img = self.crop_margins(img.convert("RGB"))
        except OSError:
            # might throw an error for broken files
            return
        if img.height == 0 or img.width == 0:
            return
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = img.rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail(self.input_size[1], self.input_size[0])
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))
    
class BartDecoder(nn.Module):
    """
    Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a Nougat decoder

    Args:
        decoder_layer:
            Number of layers of BARTDecoder
        max_position_embeddings:
            The maximum sequence length to be trained
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local,
            otherwise, `facebook/mbart-large-50` will be set (using `transformers`)
"""
    def __init__(
            self,
            decoder_layer: int,
            max_position_embeddings: int,
            hidden_dimention: int = 1024,
            name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings
        if not name_or_path:
            tokenizer_file = Path(__file__).parent / "dataset" / "tokenizer.json"
        else:
            tokenizer_file = Path(name_or_path) / "tokenizer.json"
        if not tokenizer_file.exists():
            raise ValueError("Could not fine tokenizer file")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.unk_token = "<unk>"

        self.model = MBartForCausalLM(
            config=MBartConfig(
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=len(self.tokenizer),
                scale_embedding=True,
                add_final_layer_norm=True,
                d_model=hidden_dimention,
            )
        )
        self.model.config.is_encoder_decoder = True # to get cross-attention
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference

        if not name_or_path:
            bart_state_dict = MBartForCausalLM.from_pretrained(
                "facebook/mbart-large-50"
            ).state_dict()
            new_bart_state_dict = self.model.state_dict()
            for x in new_bart_state_dict:
                if (
                    x.endswith("embed_positions.weight")
                    and self.max_position_embeddings != 1024
                ):
                    new_bart_state_dict[x] = torch.nn.Parameter(
                        self.resize_bart_abs_pos_emb(
                            bart_state_dict[x],
                            self.max_position_embeddings
                            + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                        )
                    )
                elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                    new_bart_state_dict[x] = bart_state_dict[x][
                        : len(self.tokenizer), :
                    ]
                else:
                    new_bart_state_dict[x] = bart_state_dict[x]
            self.model.load_state_dict(new_bart_state_dict)
    
    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(list_of_tokens))}
        )
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def prepare_inputs_for_inference(
            self,
            input_ids: torch.Tensor,
            encoder_outputs: torch.Tensor,
            past=None,
            past_key_values=None,
            use_cache: bool=None,
            attention_mask: torch.Tensor=None,
    ):
        """
        Args:
            input_ids: (batch_size, sequence_length)

        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        past = past or past_key_values
        if past is not None:
            input_ids = input_ids[:, -1:] 
        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return outputs
    
    def forward(
            self,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: bool = None,
            output_attentions: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[torch.Tensor] = None,
            return_dict: bool = None
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )





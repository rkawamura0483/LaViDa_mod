"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from llava.utils import rank0_print


class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # SHIRG-FIX: 2025-07-27 - Handle variable resolution position embeddings
        # ISSUE: Position embeddings are fixed to 729 positions but high-res gives 2916 tokens
        # SOLUTION: Interpolate position embeddings to match actual token count
        # LAVIDA IMPACT: Maintains compatibility while enabling high-resolution processing
        # SHIRG IMPACT: Enables position-aware high-resolution token extraction
        
        batch_size, num_tokens, embed_dim = embeddings.shape
        
        if num_tokens == self.num_positions:
            # Standard case: use original position embeddings
            embeddings = embeddings + self.position_embedding(self.position_ids)
        else:
            # High-resolution case: interpolate position embeddings
            import torch.nn.functional as F
            
            # Get original position embeddings [729, embed_dim]
            orig_pos_embeds = self.position_embedding.weight  # [729, embed_dim]
            
            # Reshape to 2D grid: 729 = 27×27
            grid_size = int(self.num_positions ** 0.5)  # 27
            orig_pos_embeds_2d = orig_pos_embeds.view(grid_size, grid_size, embed_dim)  # [27, 27, embed_dim]
            
            # Calculate target grid size: e.g., 2916 = 54×54
            target_grid_size = int(num_tokens ** 0.5)
            
            # Permute for interpolation: [embed_dim, 27, 27]
            orig_pos_embeds_2d = orig_pos_embeds_2d.permute(2, 0, 1).unsqueeze(0)  # [1, embed_dim, 27, 27]
            
            # Interpolate to target size: [1, embed_dim, target_grid_size, target_grid_size]
            interp_pos_embeds_2d = F.interpolate(
                orig_pos_embeds_2d, 
                size=(target_grid_size, target_grid_size), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Reshape back to token sequence: [1, embed_dim, target_grid_size, target_grid_size] -> [num_tokens, embed_dim]
            interp_pos_embeds = interp_pos_embeds_2d.squeeze(0).permute(1, 2, 0).view(num_tokens, embed_dim)
            
            # Add interpolated position embeddings
            embeddings = embeddings + interp_pos_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            
        return embeddings


class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        self.image_processor = SigLipImageProcessor()

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        # SHIRG-FIX: 2025-07-27 - Dual-architecture approach for LaViDa + SHIRG compatibility
        # ISSUE: Need both LaViDa (729 tokens, reduced encoder) and SHIRG (3645 tokens, full encoder)
        # SOLUTION: Maintain LaViDa compatibility but create separate full encoder for SHIRG
        # LAVIDA IMPACT: Preserves original LaViDa architecture and performance
        # SHIRG IMPACT: Enables genuine high-resolution token extraction with full encoder
        
        # Keep original LaViDa approach for standard forward() method
        del self.vision_tower.vision_model.encoder.layers[-1:]
        
        # Create separate full encoder for SHIRG high-resolution extraction
        self._full_encoder_for_shirg = None  # Lazy-loaded when needed
        
        rank0_print("SHIRG: Dual-architecture setup - LaViDa compatibility (26 layers) + SHIRG capability (27 layers)")
        
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                # SHIRG-FIX: 2025-07-27 - Remove 729 token assertion to support variable resolutions
                # ISSUE: Hard assertion breaks when using high-resolution features
                # SOLUTION: Use flexible shape validation for both 729 and high-res modes
                # LAVIDA IMPACT: Maintains backward compatibility while enabling high-res processing
                # SHIRG IMPACT: Allows extraction of genuine high-resolution tokens (3,645+)
                if image_feature.shape[-2] not in [729, 2304, 2916, 3645]:  # Support standard and various high-res
                    rank0_print(f"SHIRG Warning: Unexpected token count {image_feature.shape[-2]}, expected 729, 2304, 2916, or 3645")
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            # SHIRG-FIX: Same flexible validation for batch processing
            if image_features.shape[-2] not in [729, 2304, 2916, 3645]:
                rank0_print(f"SHIRG Warning: Unexpected token count {image_features.shape[-2]}, expected 729, 2304, 2916, or 3645")

        return image_features

    def forward_with_high_quality(self, images, return_high_quality=False):
        """
        Forward pass with optional high-quality token extraction for SHIRG
        
        SHIRG-FIX: 2025-07-27 - Correct SHIRG methodology implementation
        ISSUE: SHIRG needs comparable features for valid research, not different resolutions
        SOLUTION: Same resolution (384x384), different encoder quality (26 vs 27 layers)
        LAVIDA IMPACT: Standard path unchanged, exact compatibility maintained
        SHIRG IMPACT: Provides comparable high-quality features for meaningful token selection
        
        Args:
            images: Input images (single tensor or list) at standard 384x384 resolution
            return_high_quality: If True, return both standard and high-quality features
            
        Returns:
            If return_high_quality=False: standard LaViDa features only
            If return_high_quality=True: (standard_features, high_quality_features)
        """
        
        if not return_high_quality:
            # Standard LaViDa path - unchanged
            return self.forward(images)
        
        # SHIRG path - dual processing with SAME resolution but different encoder quality
        if type(images) is list:
            standard_features = []
            high_quality_features = []
            
            for image in images:
                # Path 1: Standard LaViDa features (26 layers, 729 tokens)
                standard_feature = self._extract_high_quality_tokens(
                    image.unsqueeze(0) if image.dim() == 3 else image, 
                    use_full_encoder=False
                ).to(image.dtype)
                standard_features.append(standard_feature)
                
                # Path 2: SHIRG high-quality features (27 layers, 729 tokens, SAME resolution)
                high_quality_feature = self._extract_high_quality_tokens(
                    image.unsqueeze(0) if image.dim() == 3 else image,
                    use_full_encoder=True
                ).to(image.dtype)
                high_quality_features.append(high_quality_feature)
                
            return standard_features, high_quality_features
        else:
            # Batch processing
            # Path 1: Standard LaViDa features (26 layers)
            standard_features = self._extract_high_quality_tokens(
                images, use_full_encoder=False
            ).to(images.dtype)
            
            # Path 2: SHIRG high-quality features (27 layers)
            high_quality_features = self._extract_high_quality_tokens(
                images, use_full_encoder=True
            ).to(images.dtype)
            
            return standard_features, high_quality_features

    def _get_full_encoder_for_shirg(self):
        """
        Get full SigLIP encoder for SHIRG - using SAME resolution but MORE layers
        
        SHIRG-FIX: 2025-07-27 - Correct SHIRG approach
        ISSUE: Previous approach changed resolution AND layers = incompatible features
        SOLUTION: Keep SAME resolution (384x384) but use FULL encoder (27 layers vs 26)
        LAVIDA IMPACT: No impact - maintains 384x384 with 26 layers
        SHIRG IMPACT: Higher quality features at SAME resolution for meaningful comparison
        """
        if self._full_encoder_for_shirg is None:
            rank0_print("SHIRG: Loading full SigLIP encoder (27 layers) for same-resolution high-quality extraction")
            
            # Load complete SigLIP model with all 27 layers
            self._full_encoder_for_shirg = SigLipVisionModel.from_pretrained(
                self.vision_tower_name
            ).to(device=self.device, dtype=self.dtype)
            
            # Remove head but keep ALL encoder layers (this is the key difference)
            self._full_encoder_for_shirg.vision_model.head = nn.Identity()
            self._full_encoder_for_shirg.requires_grad_(False)
            
            rank0_print(f"SHIRG: Full encoder ready - {len(self._full_encoder_for_shirg.vision_model.encoder.layers)} layers for high-quality features")
            
        return self._full_encoder_for_shirg
    
    def _extract_high_quality_tokens(self, images, use_full_encoder=True):
        """
        Extract high-quality tokens using full SigLIP encoder at SAME resolution
        
        SHIRG-FIX: 2025-07-27 - Correct SHIRG methodology
        ISSUE: SHIRG doesn't need different resolutions - it needs BETTER features for selection
        SOLUTION: Same 384x384 resolution, but use FULL encoder (27 layers) vs reduced (26 layers)
        LAVIDA IMPACT: None - this preserves exact LaViDa compatibility
        SHIRG IMPACT: Higher quality features for better token selection, comparable to LaViDa baseline
        
        Args:
            images: Input images tensor [B, C, H, W] at standard 384x384 resolution
            use_full_encoder: If True, use full 27-layer encoder; if False, use LaViDa 26-layer
            
        Returns:
            High-quality token features [B, 729, D] with same spatial layout as LaViDa
        """
        
        # Handle single image vs batch
        if images.dim() == 3:  # Single image [C, H, W]
            images = images.unsqueeze(0)
        
        # SHIRG-FIX: Keep images at standard LaViDa resolution (384x384)
        # The key insight: SHIRG needs better QUALITY features, not different RESOLUTION
        input_images = images.to(device=self.device, dtype=self.dtype)
        
        if use_full_encoder:
            # Use full 27-layer encoder for higher quality features
            full_encoder = self._get_full_encoder_for_shirg()
            
            with torch.no_grad():
                outputs = full_encoder(input_images, output_hidden_states=True)
                tokens = outputs.last_hidden_state
                
        else:
            # Use standard LaViDa path (26 layers) for comparison
            with torch.no_grad():
                outputs = self.vision_tower(input_images, output_hidden_states=True)
                tokens = outputs.hidden_states[-1]
        
        # Both paths should give exactly 729 tokens (27x27 grid from 384x384)
        expected_tokens = 729
        actual_tokens = tokens.shape[1]
        
        if actual_tokens != expected_tokens:
            rank0_print(f"SHIRG Warning: Expected {expected_tokens} tokens, got {actual_tokens}")
        
        return tokens

    def get_high_quality_tokens_for_shirg(self, images, token_budget=729):
        """
        Extract high-quality tokens for SHIRG selection using correct methodology
        
        SHIRG-FIX: 2025-07-27 - Implement actual SHIRG research approach
        ISSUE: SHIRG research is about TOKEN SELECTION, not multi-view or resolution changes
        SOLUTION: Extract high-quality tokens at standard resolution for selection
        LAVIDA IMPACT: Maintains exact LaViDa compatibility
        SHIRG IMPACT: Provides high-quality token pool for SHIRG selection algorithm
        
        Args:
            images: Input images [B, C, H, W] at standard 384x384 resolution
            token_budget: Number of tokens to extract (default: 729 for LaViDa compatibility)
            
        Returns:
            High-quality tokens [B, token_budget, D] ready for SHIRG selection
        """
        
        # Extract high-quality tokens using full encoder at standard resolution
        high_quality_tokens = self._extract_high_quality_tokens(
            images, use_full_encoder=True
        )
        
        current_tokens = high_quality_tokens.shape[1]
        
        if current_tokens != token_budget:
            if current_tokens > token_budget:
                # Trim to budget
                high_quality_tokens = high_quality_tokens[:, :token_budget, :]
                rank0_print(f"SHIRG: Trimmed from {current_tokens} to {token_budget} tokens")
            else:
                # Pad if needed (shouldn't happen with standard 384x384)
                batch_size = high_quality_tokens.shape[0]
                pad_size = token_budget - current_tokens
                mean_token = high_quality_tokens.mean(dim=1, keepdim=True)
                padding = mean_token.expand(batch_size, pad_size, -1)
                high_quality_tokens = torch.cat([high_quality_tokens, padding], dim=1)
                rank0_print(f"SHIRG: Padded from {current_tokens} to {token_budget} tokens")
        
        rank0_print(f"SHIRG: High-quality token extraction complete: {high_quality_tokens.shape}")
        return high_quality_tokens

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size

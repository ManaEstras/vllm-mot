# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Processor for HunYuanVL-MoT (Mixture of Tokens) model."""

from typing import Union

import numpy as np
import torch

from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput

logger = logging.get_logger(__name__)

# MoT-specific token IDs (distinct from base HunYuanVL 120120)
IMAGE_TOKEN_ID = 120687
VIDEO_TOKEN_ID = 120688
NEWLINE_TOKEN_ID = 120689
VISION_START_TOKEN_ID = 120684
VISION_END_TOKEN_ID = 120685
LATENT_TOKEN_ID = 120690


class HunYuanVLMoTProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class HunYuanVLMoTProcessor(ProcessorMixin):
    """Processor for HunYuanVL-MoT combining image/video processing and tokenization.

    Token IDs differ from base HunYuanVL:
    - IMAGE_TOKEN_ID: 120687 (base uses 120120)
    - VIDEO_TOKEN_ID: 120688
    - NEWLINE_TOKEN_ID: 120689
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )
        self.image_token = (
            "<｜hy_place▁holder▁no▁669｜>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<｜hy_place▁holder▁no▁670｜>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_start_token = (
            "<｜hy_place▁holder▁no▁666｜>"
            if not hasattr(tokenizer, "vision_start_token")
            else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<｜hy_place▁holder▁no▁667｜>"
            if not hasattr(tokenizer, "vision_end_token")
            else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput,
            PreTokenizedInput,
            list[TextInput],
            list[PreTokenizedInput],
        ] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[HunYuanVLMoTProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            HunYuanVLMoTProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(
                images=images, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(
                videos=videos, **output_kwargs["videos_kwargs"]
            )
            video_grid_thw = videos_inputs["video_grid_thw"]
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = list(text)  # copy to avoid modifying caller's list

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    # Build row-based image prompt: image_token*cols + row_sep
                    rows = image_grid_thw[index][1] // self.image_processor.merge_size
                    cols = image_grid_thw[index][2] // self.image_processor.merge_size
                    T = image_grid_thw[index][0]
                    image_prompt = (
                        "<|placeholder|>" * cols + "<｜hy_place▁holder▁no▁671｜>"
                    ) * (rows * T)
                    text[i] = text[i].replace(self.image_token, image_prompt, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "HunYuanVL-MoT requires frame timestamps but fps "
                            "could not be inferred. Defaulting to fps=24."
                        )
                        metadata.fps = 24
                    rows = video_grid_thw[index][1] // self.video_processor.merge_size
                    cols = video_grid_thw[index][2] // self.video_processor.merge_size
                    video_prompt = (
                        "<|placeholder|>" * cols + "<｜hy_place▁holder▁no▁671｜>"
                    ) * rows
                    video_placeholder = ""
                    for _ in range(video_grid_thw[index][0]):
                        video_placeholder += (
                            self.vision_start_token
                            + video_prompt
                            + self.vision_end_token
                        )
                    target = (
                        f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
                    )
                    if target in text[i]:
                        text[i] = text[i].replace(target, video_placeholder, 1)
                    else:
                        text[i] = text[i].replace(
                            self.video_token, video_placeholder, 1
                        )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", None
        )
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )

    def _get_num_multimodal_tokens(
        self, image_sizes=None, video_sizes=None, **kwargs
    ):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = HunYuanVLMoTProcessorKwargs._defaults.get(
                "images_kwargs", {}
            )
            images_kwargs.update(kwargs)
            merge_size = (
                images_kwargs.get("merge_size", None) or self.image_processor.merge_size
            )
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(
                    *image_size, images_kwargs
                )
                for image_size in image_sizes
            ]
            num_image_tokens = [
                (num_patches // merge_size ** 2) for num_patches in num_image_patches
            ]
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )

        if video_sizes is not None:
            videos_kwargs = HunYuanVLMoTProcessorKwargs._defaults.get(
                "videos_kwargs", {}
            )
            videos_kwargs.update(kwargs)
            merge_size = (
                videos_kwargs.get("merge_size", None) or self.video_processor.merge_size
            )
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(
                    *video_size, videos_kwargs
                )
                for video_size in video_sizes
            ]
            num_video_tokens = [
                (num_patches // merge_size ** 2) for num_patches in num_video_patches
            ]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["HunYuanVLMoTProcessor"]
AutoProcessor.register("HunYuanVLMoTProcessor", HunYuanVLMoTProcessor)

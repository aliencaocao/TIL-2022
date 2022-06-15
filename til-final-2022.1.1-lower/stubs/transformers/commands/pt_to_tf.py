# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser, Namespace
from importlib import import_module

import numpy as np
from datasets import load_dataset

from huggingface_hub import Repository, upload_file

from .. import AutoConfig, AutoFeatureExtractor, AutoTokenizer, is_tf_available, is_torch_available
from ..utils import logging
from . import BaseTransformersCLICommand


if is_tf_available():
    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(False)

if is_torch_available():
    import torch


MAX_ERROR = 5e-5  # larger error tolerance than in our internal tests, to avoid flaky user-facing errors
TF_WEIGHTS_NAME = "tf_model.h5"


def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model PyTorch checkpoint in a TensorFlow 2 checkpoint.

    Returns: ServeCommand
    """
    return PTtoTFCommand(args.model_name, args.local_dir, args.no_pr, args.new_weights)


class PTtoTFCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        train_parser = parser.add_parser(
            "pt-to-tf",
            help=(
                "CLI tool to run convert a transformers model from a PyTorch checkpoint to a TensorFlow checkpoint."
                " Can also be used to validate existing weights without opening PRs, with --no-pr."
            ),
        )
        train_parser.add_argument(
            "--model-name",
            type=str,
            required=True,
            help="The model name, including owner/organization, as seen on the hub.",
        )
        train_parser.add_argument(
            "--local-dir",
            type=str,
            default="",
            help="Optional local directory of the model repository. Defaults to /tmp/{model_name}",
        )
        train_parser.add_argument(
            "--no-pr", action="store_true", help="Optional flag to NOT open a PR with converted weights."
        )
        train_parser.add_argument(
            "--new-weights",
            action="store_true",
            help="Optional flag to create new TensorFlow weights, even if they already exist.",
        )
        train_parser.set_defaults(func=convert_command_factory)

    @staticmethod
    def find_pt_tf_differences(pt_model, pt_input, tf_model, tf_input):
        """
        Compares the TensorFlow and PyTorch models, given their inputs, returning a dictionary with all tensor
        differences.
        """
        pt_outputs = pt_model(**pt_input, output_hidden_states=True)
        tf_outputs = tf_model(**tf_input, output_hidden_states=True)

        # 1. All output attributes must be the same
        pt_out_attrs = set(pt_outputs.keys())
        tf_out_attrs = set(tf_outputs.keys())
        if pt_out_attrs != tf_out_attrs:
            raise ValueError(
                f"The model outputs have different attributes, aborting. (Pytorch: {pt_out_attrs}, TensorFlow:"
                f" {tf_out_attrs})"
            )

        # 2. For each output attribute, computes the difference
        def _find_pt_tf_differences(pt_out, tf_out, differences, attr_name=""):

            # If the current attribute is a tensor, it is a leaf and we make the comparison. Otherwise, we will dig in
            # recursivelly, keeping the name of the attribute.
            if isinstance(pt_out, torch.Tensor):
                tensor_difference = np.max(np.abs(pt_out.detach().numpy() - tf_out.numpy()))
                differences[attr_name] = tensor_difference
            else:
                root_name = attr_name
                for i, pt_item in enumerate(pt_out):
                    # If it is a named attribute, we keep the name. Otherwise, just its index.
                    if isinstance(pt_item, str):
                        branch_name = root_name + pt_item
                        tf_item = tf_out[pt_item]
                        pt_item = pt_out[pt_item]
                    else:
                        branch_name = root_name + f"[{i}]"
                        tf_item = tf_out[i]
                    differences = _find_pt_tf_differences(pt_item, tf_item, differences, branch_name)

            return differences

        return _find_pt_tf_differences(pt_outputs, tf_outputs, {})

    def __init__(self, model_name: str, local_dir: str, no_pr: bool, new_weights: bool, *args):
        self._logger = logging.get_logger("transformers-cli/pt_to_tf")
        self._model_name = model_name
        self._local_dir = local_dir if local_dir else os.path.join("/tmp", model_name)
        self._no_pr = no_pr
        self._new_weights = new_weights

    def get_text_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained(self._local_dir)
        sample_text = ["Hi there!", "I am a batch with more than one row and different input lengths."]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pt_input = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
        tf_input = tokenizer(sample_text, return_tensors="tf", padding=True, truncation=True)
        return pt_input, tf_input

    def get_audio_inputs(self):
        processor = AutoFeatureExtractor.from_pretrained(self._local_dir)
        num_samples = 2
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]
        raw_samples = [x["array"] for x in speech_samples]
        pt_input = processor(raw_samples, return_tensors="pt", padding=True)
        tf_input = processor(raw_samples, return_tensors="tf", padding=True)
        return pt_input, tf_input

    def get_image_inputs(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(self._local_dir)
        num_samples = 2
        ds = load_dataset("cifar10", "plain_text", split="test")[:num_samples]["img"]
        pt_input = feature_extractor(images=ds, return_tensors="pt")
        tf_input = feature_extractor(images=ds, return_tensors="tf")
        return pt_input, tf_input

    def run(self):
        # Fetch remote data
        # TODO: implement a solution to pull a specific PR/commit, so we can use this CLI to validate pushes.
        repo = Repository(local_dir=self._local_dir, clone_from=self._model_name)
        repo.git_pull()  # in case the repo already exists locally, but with an older commit

        # Load config and get the appropriate architecture -- the latter is needed to convert the head's weights
        config = AutoConfig.from_pretrained(self._local_dir)
        architectures = config.architectures
        if architectures is None:  # No architecture defined -- use auto classes
            pt_class = getattr(import_module("transformers"), "AutoModel")
            tf_class = getattr(import_module("transformers"), "TFAutoModel")
            self._logger.warn("No detected architecture, using AutoModel/TFAutoModel")
        else:  # Architecture defined -- use it
            if len(architectures) > 1:
                raise ValueError(f"More than one architecture was found, aborting. (architectures = {architectures})")
            self._logger.warn(f"Detected architecture: {architectures[0]}")
            pt_class = getattr(import_module("transformers"), architectures[0])
            try:
                tf_class = getattr(import_module("transformers"), "TF" + architectures[0])
            except AttributeError:
                raise AttributeError(f"The TensorFlow equivalent of {architectures[0]} doesn't exist in transformers.")

        # Load models and acquire a basic input for its modality.
        pt_model = pt_class.from_pretrained(self._local_dir)
        main_input_name = pt_model.main_input_name
        if main_input_name == "input_ids":
            pt_input, tf_input = self.get_text_inputs()
        elif main_input_name == "pixel_values":
            pt_input, tf_input = self.get_image_inputs()
        elif main_input_name == "input_features":
            pt_input, tf_input = self.get_audio_inputs()
        else:
            raise ValueError(f"Can't detect the model modality (`main_input_name` = {main_input_name})")
        tf_from_pt_model = tf_class.from_pretrained(self._local_dir, from_pt=True)

        # Extra input requirements, in addition to the input modality
        if config.is_encoder_decoder or (hasattr(pt_model, "encoder") and hasattr(pt_model, "decoder")):
            decoder_input_ids = np.asarray([[1], [1]], dtype=int) * pt_model.config.decoder_start_token_id
            pt_input.update({"decoder_input_ids": torch.tensor(decoder_input_ids)})
            tf_input.update({"decoder_input_ids": tf.convert_to_tensor(decoder_input_ids)})

        # Confirms that cross loading PT weights into TF worked.
        crossload_differences = self.find_pt_tf_differences(pt_model, pt_input, tf_from_pt_model, tf_input)
        max_crossload_diff = max(crossload_differences.values())
        if max_crossload_diff > MAX_ERROR:
            raise ValueError(
                "The cross-loaded TensorFlow model has different outputs, something went wrong! Exaustive list of"
                f" maximum tensor differences above the error threshold ({MAX_ERROR}):\n"
                + "\n".join(
                    [f"{key}: {value:.3e}" for key, value in crossload_differences.items() if value > MAX_ERROR]
                )
            )

        # Save the weights in a TF format (if needed) and confirms that the results are still good
        tf_weights_path = os.path.join(self._local_dir, TF_WEIGHTS_NAME)
        if not os.path.exists(tf_weights_path) or self._new_weights:
            tf_from_pt_model.save_weights(tf_weights_path)
        del tf_from_pt_model  # will no longer be used, and may have a large memory footprint
        tf_model = tf_class.from_pretrained(self._local_dir)
        conversion_differences = self.find_pt_tf_differences(pt_model, pt_input, tf_model, tf_input)
        max_conversion_diff = max(conversion_differences.values())
        if max_conversion_diff > MAX_ERROR:
            raise ValueError(
                "The converted TensorFlow model has different outputs, something went wrong! Exaustive list of maximum"
                f" tensor differences above the error threshold ({MAX_ERROR}):\n"
                + "\n".join(
                    [f"{key}: {value:.3e}" for key, value in conversion_differences.items() if value > MAX_ERROR]
                )
            )

        if not self._no_pr:
            # TODO: remove try/except when the upload to PR feature is released
            # (https://github.com/huggingface/huggingface_hub/pull/884)
            try:
                self._logger.warn("Uploading the weights into a new PR...")
                hub_pr_url = upload_file(
                    path_or_fileobj=tf_weights_path,
                    path_in_repo=TF_WEIGHTS_NAME,
                    repo_id=self._model_name,
                    create_pr=True,
                    pr_commit_summary="Add TF weights",
                    pr_commit_description=(
                        "Model converted by the `transformers`' `pt_to_tf` CLI -- all converted model outputs and"
                        " hidden layers were validated against its Pytorch counterpart. Maximum crossload output"
                        f" difference={max_crossload_diff:.3e}; Maximum converted output"
                        f" difference={max_conversion_diff:.3e}."
                    ),
                )
                self._logger.warn(f"PR open in {hub_pr_url}")
            except TypeError:
                self._logger.warn(
                    f"You can now open a PR in https://huggingface.co/{self._model_name}/discussions, manually"
                    f" uploading the file in {tf_weights_path}"
                )

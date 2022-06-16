import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import requests

from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from .file_download import hf_hub_download, is_torch_available
from .hf_api import HfApi, HfFolder
from .repository import Repository
from .utils import logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class ModelHubMixin:
    """
    A Generic Base Model Hub Mixin. Define your own mixin for anything by
    inheriting from this class and overwriting `_from_pretrained` and
    `_save_pretrained` to define custom logic for saving/loading your classes.
    See `huggingface_hub.PyTorchModelHubMixin` for an example.
    """

    def save_pretrained(
        self,
        save_directory: str,
        config: Optional[dict] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save weights in local directory.

                Parameters:
                    save_directory (`str`):
                        Specify directory in which you want to save weights.
                    config (`dict`, *optional*):
                        specify config (must be dict) in case you want to save
                        it.
                    push_to_hub (`bool`, *optional*, defaults to `False`):
                        Set it to `True` in case you want to push your weights
                        to huggingface_hub
                    kwargs (`Dict`, *optional*):
                        kwargs will be passed to `push_to_hub`
        """

        os.makedirs(save_directory, exist_ok=True)

        # saving config
        if isinstance(config, dict):
            path = os.path.join(save_directory, CONFIG_NAME)
            with open(path, "w") as f:
                json.dump(config, f)

        # saving model weights/files
        self._save_pretrained(save_directory)

        if push_to_hub:
            return self.push_to_hub(save_directory, **kwargs)

    def _save_pretrained(self, save_directory):
        """
        Overwrite this method in subclass to define how to save your model.
        """
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Dict = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **model_kwargs,
    ):
        r"""
        Instantiate a pretrained PyTorch model from a pre-trained model
                configuration from huggingface-hub. The model is set in
                evaluation mode by default using `model.eval()` (Dropout modules
                are deactivated). To train the model, you should first set it
                back in training mode with `model.train()`.

                Parameters:
                    pretrained_model_name_or_path (`str` or `os.PathLike`):
                        Can be either:
                            - A string, the `model id` of a pretrained model
                              hosted inside a model repo on huggingface.co.
                              Valid model ids can be located at the root-level,
                              like `bert-base-uncased`, or namespaced under a
                              user or organization name, like
                              `dbmdz/bert-base-german-cased`.
                            - You can add `revision` by appending `@` at the end
                              of model_id simply like this:
                              `dbmdz/bert-base-german-cased@main` Revision is
                              the specific model version to use. It can be a
                              branch name, a tag name, or a commit id, since we
                              use a git-based system for storing models and
                              other artifacts on huggingface.co, so `revision`
                              can be any identifier allowed by git.
                            - A path to a `directory` containing model weights
                              saved using
                              [`~transformers.PreTrainedModel.save_pretrained`],
                              e.g., `./my_model_directory/`.
                            - `None` if you are both providing the configuration
                              and state dictionary (resp. with keyword arguments
                              `config` and `state_dict`).
                    force_download (`bool`, *optional*, defaults to `False`):
                        Whether to force the (re-)download of the model weights
                        and configuration files, overriding the cached versions
                        if they exist.
                    resume_download (`bool`, *optional*, defaults to `False`):
                        Whether to delete incompletely received files. Will
                        attempt to resume the download if such a file exists.
                    proxies (`Dict[str, str]`, *optional*):
                        A dictionary of proxy servers to use by protocol or
                        endpoint, e.g., `{'http': 'foo.bar:3128',
                        'http://hostname': 'foo.bar:4012'}`. The proxies are
                        used on each request.
                    use_auth_token (`str` or `bool`, *optional*):
                        The token to use as HTTP bearer authorization for remote
                        files. If `True`, will use the token generated when
                        running `transformers-cli login` (stored in
                        `~/.huggingface`).
                    cache_dir (`Union[str, os.PathLike]`, *optional*):
                        Path to a directory in which a downloaded pretrained
                        model configuration should be cached if the standard
                        cache should not be used.
                    local_files_only(`bool`, *optional*, defaults to `False`):
                        Whether to only look at local files (i.e., do not try to
                        download the model).
                    model_kwargs (`Dict`, *optional*):
                        model_kwargs will be passed to the model during
                        initialization

                <Tip>

                Passing `use_auth_token=True` is required when you want to use a
                private model.

                </Tip>
        """

        model_id = pretrained_model_name_or_path

        revision = None
        if len(model_id.split("@")) == 2:
            model_id, revision = model_id.split("@")

        if os.path.isdir(model_id) and CONFIG_NAME in os.listdir(model_id):
            config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    use_auth_token=use_auth_token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                logger.warning(f"{CONFIG_NAME} not found in HuggingFace Hub")
                config_file = None

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_kwargs.update({"config": config})

        return cls._from_pretrained(
            model_id,
            revision,
            cache_dir,
            force_download,
            proxies,
            resume_download,
            local_files_only,
            use_auth_token,
            **model_kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        use_auth_token,
        **model_kwargs,
    ):
        """Overwrite this method in subclass to define how to load your model from
        pretrained"""
        raise NotImplementedError

    def push_to_hub(
        self,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        commit_message: Optional[str] = "Add model",
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        api_endpoint: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        git_user: Optional[str] = None,
        git_email: Optional[str] = None,
        config: Optional[dict] = None,
        skip_lfs_files: bool = False,
    ) -> str:
        """
        Upload model checkpoint or tokenizer files to the Hub while
        synchronizing a local clone of the repo in `repo_path_or_name`.

        Parameters:
            repo_path_or_name (`str`, *optional*):
                Can either be a repository name for your model or tokenizer in
                the Hub or a path to a local folder (in which case the
                repository will have the name of that local folder). If not
                specified, will default to the name given by `repo_url` and a
                local directory with that name will be created.
            repo_url (`str`, *optional*):
                Specify this in case you want to push to an existing repository
                in the hub. If unspecified, a new repository will be created in
                your namespace (unless you specify an `organization`) with
                `repo_name`.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"add config"`,
                `"add tokenizer"` or `"add model"` depending on the type of the
                class.
            organization (`str`, *optional*):
                Organization in which you want to push your model or tokenizer
                (you must be a member of this organization).
            private (`bool`, *optional*):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            use_auth_token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
                If `True`, will use the token generated when running
                `transformers-cli login` (stored in `~/.huggingface`). Will
                default to `True` if `repo_url` is not specified.
            git_user (`str`, *optional*):
                will override the `git config user.name` for committing and
                pushing files to the hub.
            git_email (`str`, *optional*):
                will override the `git config user.email` for committing and
                pushing files to the hub.
            config (`dict`, *optional*):
                Configuration object to be saved alongside the model weights.
            skip_lfs_files (`bool`, *optional*, defaults to `False`):
                Whether to skip git-LFS files or not.


        Returns:
            The url of the commit of your model in the given repository.
        """

        if repo_path_or_name is None and repo_url is None:
            raise ValueError(
                "You need to specify a `repo_path_or_name` or a `repo_url`."
            )

        if use_auth_token is None and repo_url is None:
            token = HfFolder.get_token()
            if token is None:
                raise ValueError(
                    "You must login to the Hugging Face hub on this computer by typing"
                    " `huggingface-cli login` and entering your credentials to use"
                    " `use_auth_token=True`. Alternatively, you can pass your own token"
                    " as the `use_auth_token` argument."
                )
        elif isinstance(use_auth_token, str):
            token = use_auth_token
        else:
            token = None

        if repo_path_or_name is None:
            repo_path_or_name = repo_url.split("/")[-1]

        # If no URL is passed and there's no path to a directory containing files, create a repo
        if repo_url is None and not os.path.exists(repo_path_or_name):
            repo_id = Path(repo_path_or_name).name
            if organization:
                repo_id = f"{organization}/{repo_id}"
            repo_url = HfApi(endpoint=api_endpoint).create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type=None,
                exist_ok=True,
            )

        repo = Repository(
            repo_path_or_name,
            clone_from=repo_url,
            use_auth_token=use_auth_token,
            git_user=git_user,
            git_email=git_email,
            skip_lfs_files=skip_lfs_files,
        )
        repo.git_pull(rebase=True)

        # Save the files in the cloned repo
        self.save_pretrained(repo_path_or_name, config=config)

        # Commit and push!
        repo.git_add()
        repo.git_commit(commit_message)
        return repo.git_push()


class PyTorchModelHubMixin(ModelHubMixin):
    def __init__(self, *args, **kwargs):
        """
        Mix this class with your torch-model class for ease process of saving &
        loading from huggingface-hub.

        Example usage:

        ```python
        >>> from huggingface_hub import PyTorchModelHubMixin


        >>> class MyModel(nn.Module, PyTorchModelHubMixin):
        ...     def __init__(self, **kwargs):
        ...         super().__init__()
        ...         self.config = kwargs.pop("config", None)
        ...         self.layer = ...

        ...     def forward(self, *args):
        ...         return ...


        >>> model = MyModel()
        >>> model.save_pretrained(
        ...     "mymodel", push_to_hub=False
        >>> )  # Saving model weights in the directory
        >>> model.push_to_hub(
        ...     "mymodel", "model-1"
        >>> )  # Pushing model-weights to hf-hub

        >>> # Downloading weights from hf-hub & model will be initialized from those weights
        >>> model = MyModel.from_pretrained("username/mymodel@main")
        ```
        """

    def _save_pretrained(self, save_directory):
        """
        Overwrite this method in case you don't want to save complete model,
        rather some specific layers
        """
        path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        use_auth_token,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):
        """
        Overwrite this method in case you wish to initialize your model in a
        different way.
        """
        map_location = torch.device(map_location)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )
        model = cls(**model_kwargs)

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model

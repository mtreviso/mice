from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional
import logging
from os import PathLike
import json
import re
from contextlib import contextmanager

import numpy
import torch
from pathlib import Path
from torch.utils.hooks import RemovableHandle
from torch import Tensor
from torch import backends

from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.common import Registrable, plugins
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.batch import Batch
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.nn import util

logger = logging.getLogger(__name__)

class T5Predictor(TextClassifierPredictor):
    """
    a `Predictor` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True) -> None:
        super().__init__(model, dataset_reader, frozen=frozen)

    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        # Parameters

        instances : `List[Instance]`

        # Returns

        `Tuple[Dict[str, Any], Dict[str, Any]]`
            The first item is a Dict of gradient entries for each input.
            The keys have the form  `{grad_input_1: ..., grad_input_2: ... }`
            up to the number of inputs given. The second item is the model's output.

        # Notes

        Takes a `JsonDict` representing the inputs of the model and converts
        them to [`Instances`](../data/instance.md)), sends these through
        the model [`forward`](../models/model.md#forward) function after registering hooks on the embedding
        layer of the model. Calls `backward` on the loss and then removes the
        hooks.
        """
        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self._model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        embedding_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)

        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        dataset_tensor_dict = util.move_to_device(dataset.as_tensor_dict(), self.cuda_device)
        # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
        with backends.cudnn.flags(enabled=False):
            outputs = self._model.make_output_human_readable(
                self._model.forward(**dataset_tensor_dict)  # type: ignore
            )

            loss = outputs["loss"]
            # Zero gradients.
            # NOTE: this is actually more efficient than calling `self._model.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for p in self._model.parameters():
                p.grad = None
            loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # restore the original requires_grad values of the parameters
        for param_name, param in self._model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

#        print("grad dict:", grad_dict)
        return grad_dict, outputs

    def get_interpretable_layer(self) -> torch.nn.Module:
        """
        Returns the input/embedding layer of the model.
        If the predictor wraps around a non-AllenNLP model,
        this function should be overridden to specify the correct input/embedding layer.
        For the cases where the input layer _is_ an embedding layer, this should be the
        layer 0 of the embedder.
        """
       
#        print("token embeddings dir:", dir(self._model._seq2seq_encoder.token_embeddings))
        return self._model._seq2seq_encoder.token_embeddings

    def get_interpretable_text_field_embedder(self) -> torch.nn.Module:
        """
        Returns the first `TextFieldEmbedder` of the model.
        If the predictor wraps around a non-AllenNLP model,
        this function should be overridden to specify the correct embedder.
        """
        return self._model._seq2seq_encoder

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the embedding layer of the model.  Used to save the gradients
        of the embeddings for use in get_gradients()

        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.

        We additionally add a hook on the _forward_ pass of the model's `TextFieldEmbedder` to save
        token offsets, if there are any.  Having token offsets means that you're using a mismatched
        token indexer, so we need to aggregate the gradients across wordpieces in a token.  We do
        that with a simple sum.
        """

        def hook_layers(module, grad_in, grad_out):
            grads = grad_out[0]
            if self._token_offsets:
                # If you have a mismatched indexer with multiple TextFields, it's quite possible
                # that the order we deal with the gradients is wrong.  We'll just take items from
                # the list one at a time, and try to aggregate the gradients.  If we got the order
                # wrong, we should crash, so you'll know about it.  If you get an error because of
                # that, open an issue on github, and we'll see what we can do.  The intersection of
                # multiple TextFields and mismatched indexers is pretty small (currently empty, that
                # I know of), so we'll ignore this corner case until it's needed.
                offsets = self._token_offsets.pop(0)
                span_grads, span_mask = util.batched_span_select(grads.contiguous(), offsets)
                span_mask = span_mask.unsqueeze(-1)
                span_grads *= span_mask  # zero out paddings

                span_grads_sum = span_grads.sum(2)
                span_grads_len = span_mask.sum(2)
                # Shape: (batch_size, num_orig_tokens, embedding_size)
                grads = span_grads_sum / torch.clamp_min(span_grads_len, 1)

                # All the places where the span length is zero, write in zeros.
                grads[(span_grads_len == 0).expand(grads.shape)] = 0

            embedding_gradients.append(grads)

        def get_token_offsets(module, inputs, outputs):
            offsets = util.get_token_offsets_from_text_field_inputs(inputs)
            if offsets is not None:
                self._token_offsets.append(offsets)

        hooks = []
        text_field_embedder = self.get_interpretable_text_field_embedder()
        hooks.append(text_field_embedder.register_forward_hook(get_token_offsets))
        embedding_layer = self.get_interpretable_layer()
        hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return hooks

    @classmethod
    def from_path(
        cls,
        archive_path: Union[str, Path],
        predictor_name: str = None,
        cuda_device: int = -1,
        dataset_reader_to_load: str = "validation",
        frozen: bool = True,
        import_plugins: bool = True,
        overrides: Union[str, Dict[str, Any]] = "",
    ) -> "Predictor":
        """
        Instantiate a `Predictor` from an archive path.
        If you need more detailed configuration options, such as overrides,
        please use `from_archive`.
        # Parameters
        archive_path : `Union[str, Path]`
            The path to the archive.
        predictor_name : `str`, optional (default=`None`)
            Name that the predictor is registered as, or None to use the
            predictor associated with the model.
        cuda_device : `int`, optional (default=`-1`)
            If `cuda_device` is >= 0, the model will be loaded onto the
            corresponding GPU. Otherwise it will be loaded onto the CPU.
        dataset_reader_to_load : `str`, optional (default=`"validation"`)
            Which dataset reader to load from the archive, either "train" or
            "validation".
        frozen : `bool`, optional (default=`True`)
            If we should call `model.eval()` when building the predictor.
        import_plugins : `bool`, optional (default=`True`)
            If `True`, we attempt to import plugins before loading the predictor.
            This comes with additional overhead, but means you don't need to explicitly
            import the modules that your predictor depends on as long as those modules
            can be found by `allennlp.common.plugins.import_plugins()`.
        overrides : `Union[str, Dict[str, Any]]`, optional (default = `""`)
            JSON overrides to apply to the unarchived `Params` object.
        # Returns
        `Predictor`
            A Predictor instance.
        """
        if import_plugins:
            plugins.import_plugins()
        return T5Predictor.from_archive(
            load_archive(archive_path, cuda_device=cuda_device, overrides=overrides),
            predictor_name,
            dataset_reader_to_load=dataset_reader_to_load,
            frozen=frozen,
        )

    # TODO: predictor_name not used, hackily setting predictor class to T5Predictor always
    @classmethod
    def from_archive(
        cls,
        archive: Archive,
        predictor_name: str = None,
        dataset_reader_to_load: str = "validation",
        frozen: bool = True,
    ) -> "Predictor":
        """
        Instantiate a `Predictor` from an [`Archive`](../models/archival.md);
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, we try to find a corresponding predictor in `DEFAULT_PREDICTORS`, or if
        one is not found, the base class (i.e. `Predictor`) will be used. Optionally specify
        which [`DatasetReader`](../data/dataset_readers/dataset_reader.md) should be loaded;
        otherwise, the validation one will be used if it exists followed by the training dataset reader.
        Optionally specify if the loaded model should be frozen, meaning `model.eval()` will be called.
        """
        # Duplicate the config so that the config inside the archive doesn't get consumed
        config = archive.config.duplicate()

        if dataset_reader_to_load == "validation":
            dataset_reader = archive.validation_dataset_reader
        else:
            dataset_reader = archive.dataset_reader

        model = archive.model
        if frozen:
            model.eval()

        return T5Predictor(model, dataset_reader)

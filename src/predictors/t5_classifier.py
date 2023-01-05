from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.transformer.t5 import T5 as T5Module, T5Output, IntT, BoolT
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("t5_classifier")
class T5Classifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
       
        from allennlp.common import cached_transformers

        self.model_name = model_name 
        model = T5Module.from_pretrained_module(
            model_name,
        )
#        model = cached_transformers.get(
#            model_name,
#            False,
#            override_weights_file,
#            override_weights_strip_prefix,
#        )

        self._seq2seq_encoder = model.encoder
        self._classifier_input_dim = self._seq2seq_encoder.hidden_size

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        token_ids = tokens['tokens']['token_ids']
        attention_mask = tokens['tokens']['mask']

        # Encode inputs.
#        print("\ninput_ids", token_ids)
#        print("\ninput_ids", token_ids.shape)
#        print("\nattention mask:", attention_mask.shape)
#        print("\nattention mask:", attention_mask)

        if token_ids.shape != attention_mask.shape:
            token_ids = torch.cat([token_ids[:, :-2], token_ids[:, -1:]], dim=1)
#        print("\nafter input_ids", token_ids)
#        print("\nafter input_ids", token_ids.shape)

        try:
            encoder_outputs: T5StackOutput = self._seq2seq_encoder(
                    input_ids=token_ids,
                    attention_mask=attention_mask,
            )
        except:
            print("token ids:", token_ids)

        encoder_outputs = encoder_outputs.last_hidden_state
#        print("encoder_outputs.shape:", encoder_outputs.shape)
#        print("token_ids.shape:", token_ids.shape)
#        print("classifier input dim:", self._classifier_input_dim)

        if self._dropout:
            encoder_outputs = self._dropout(encoder_outputs)

        encoder_outputs = encoder_outputs.sum(1)
#        print("after summing: encoder_outputs.shape:", encoder_outputs.shape)

        logits = self._classification_layer(encoder_outputs).squeeze(-1)
#        print("logits.shape:", logits.shape)
        probs = torch.nn.functional.softmax(logits, dim=-1)
#        print("probs.shape:", probs.shape)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
#        print(label)
#        print(label.long().view(-1))
#        print("\nlogits.shape:", logits.shape)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
#            loss = self._loss(logits, label.long().unsqueeze(-1))
            output_dict["loss"] = loss
#            print("logits.shape:", logits.shape)
#            print("label.shape:", label.shape)
            self._accuracy(logits, label)

#        print(output_dict)
        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics

    default_predictor = "text_classifier"

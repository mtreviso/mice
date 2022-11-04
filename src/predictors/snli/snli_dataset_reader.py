import logging

from allennlp.data import DatasetReader
from typing import List, Optional, Dict
from allennlp.data import DatasetReader, Instance

from overrides import overrides

from allennlp.data import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.fields import TextField, MetadataField, LabelField

from pathlib import Path
from itertools import chain
import os.path as osp
import tarfile
from tqdm import tqdm as tqdm
import json

from src.predictors.predictor_utils import clean_text 
import os

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)

@DatasetReader.register("rawsnli")
class SnliDatasetReader(DatasetReader):

    def __init__(
        self, 
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Optional[Tokenizer] = None,
        data_dir = "data/snli", 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.data_dir = data_dir
        self.train_file = os.path.join(self.data_dir, "train.jsonl")
        self.dev_file = os.path.join(self.data_dir, "dev.jsonl")
        self.test_file = os.path.join(self.data_dir, "test.jsonl")

    def get_path(self, file_path):
        if file_path == "train":
            return Path(self.train_file)
        elif file_path == "test":
            return Path(self.test_file)
        elif file_path == "dev":
            return Path(self.dev_file)
        raise ValueError("Invalid value for file_path")

    @overrides
    def _read(self, file_path: str):
        path = self.get_path(file_path)
        for idx, line in enumerate(path.open('r', encoding='utf8')):
            data = json.loads(line)
            label = data['gold_label']
            premise = data['sentence1']
            hypothesis = data['sentence2']
            if label == '-':
                # These were cases where the annotators disagreed; we'll just
                # skip them. It's like 800 / 500k examples in the train data
                continue
            tokens = premise + ' [SEP] ' + hypothesis
            yield self.text_to_instance(tokens, label)

    def get_inputs(self, file_path: str, return_labels: bool = False):
        logger.info(f"Getting SNLI task inputs from file path: {file_path}")
        path = self.get_path(file_path)
        inputs, labels = [], []
        for idx, line in enumerate(path.open('r', encoding='utf8')):
            data = json.loads(line)
            label = data['gold_label']
            premise = data['sentence1']
            hypothesis = data['sentence2']
            if label == '-':
                # These were cases where the annotators disagreed; we'll just
                # skip them. It's like 800 / 500k examples in the train data
                continue
            inputs.append(premise + ' [SEP] ' + hypothesis)
            labels.append(label)
        if return_labels:
            return inputs, labels
        return inputs

    @overrides
    def text_to_instance(
        self,  # type: ignore
        tokens: str,
        label: Optional[int] = None,
    ) -> Instance:
        # tokenize
        sequence = self._tokenizer.tokenize(tokens)
        fields = {
            "tokens": TextField(sequence, self._token_indexers),
        }
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)

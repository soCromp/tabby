import random
import typing as tp

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class GReaTDataset(Dataset):
    """GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def set_stuff(self, tokenizer, is_moe):
        """Set the Tokenizer and whether to use MOE format

        Args:
            tokenizer: Tokenizer from HuggingFace
            is_moe: whether _getitem should return the column indices as needed for MOE models
        """
        self.tokenizer = tokenizer
        self.is_moe = is_moe

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        """Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)
        strings = [f'{str(k).strip()} is {str(row[k][0].as_py()).strip()}<EOC>' for k in row.column_names]

        shuffle_idx = list(range(row.shape[1])) # number of columns
        random.shuffle(shuffle_idx)
        
        shuffled_text = [strings[i] for i in shuffle_idx]
        
        if self.is_moe:
            tokenized_text = self.tokenizer(shuffled_text, padding=True)
            tokenized_text['cols_iterator'] = shuffle_idx
        else:
            tokenized_text = self.tokenizer(''.join(shuffled_text), padding=True)
        return tokenized_text 

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """

    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        # print(features)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

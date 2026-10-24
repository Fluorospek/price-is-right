from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Self
from utils.items import Item

def push_to_hub(dataset_name: str, train: list[Item], val: list[Item], test: list[Item]):
        """Push Item lists to HuggingFace Hub"""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)
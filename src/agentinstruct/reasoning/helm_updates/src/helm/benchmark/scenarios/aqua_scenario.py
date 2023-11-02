import json
import os
from typing import List, Dict

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class AQuAScenario(Scenario):

    name = "aqua"
    description = "AQuA Dataset"
    tags = ["question_answering"]

    def __init__(self):
        super().__init__()

    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        split_to_filename: Dict[str, str] = {TRAIN_SPLIT: "train", VALID_SPLIT: "dev", TEST_SPLIT: "test"}

        for split, filename in split_to_filename.items():
            url: str = f"https://raw.githubusercontent.com/deepmind/AQuA/master/{filename}.json"
            target_path: str = os.path.join(data_path, filename)
            ensure_file_downloaded(source_url=url, target_path=target_path, unpack=False)

            with open(target_path, "r") as f:
                data_lst = list(f)

                for data in data_lst:
                    entry = json.loads(data)
                    question = entry["question"]
                    options = entry["options"]
                    answer = ord(entry["correct"]) - ord("A")

                    references: List[Reference] = []
                    for index, option in enumerate(options):
                        tags = [CORRECT_TAG] if index == answer else []
                        references.append(Reference(Output(text=option[2:]), tags=tags))

                    instance: Instance = Instance(
                        input=Input(text=question),
                        references=references,
                        split=split,
                    )
                    instances.append(instance)

        return instances

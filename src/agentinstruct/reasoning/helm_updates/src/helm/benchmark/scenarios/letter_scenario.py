import json
import os
from typing import List, Dict

from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class LetterScenario(Scenario):

    name = "letter"
    description = "Last Letter Concatenation Dataset"
    tags = ["symbolic_reasoning"]

    def __init__(self):
        super().__init__()

    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join(self.output_path, "data")

        instances: List[Instance] = []
        split_to_filename: Dict[str, str] = {TRAIN_SPLIT: "train", TEST_SPLIT: "test"}

        for split, filename in split_to_filename.items():
            target_path: str = os.path.join(data_path, filename)

            with open(target_path, "r") as f:
                data = json.load(f)
                for entry in data:
                    question = entry["question"]
                    answer = entry["answer"]

                    references: List[Reference] = [Reference(Output(text=answer), tags=[CORRECT_TAG])]

                    instance: Instance = Instance(
                        input=Input(text=question),
                        references=references,
                        split=split,
                    )
                    instances.append(instance)

        return instances

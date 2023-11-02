import json
import os
from typing import List, Dict

from helm.common.hierarchical_logger import hlog

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Input,
    Output,
)


class BigBenchHardScenario(Scenario):

    name = "big_bench_hard"
    description = "Big-Bench-Hard Benchmark"
    tags = ["question_answering"]

    def __init__(self, dataset: str):
        super().__init__()
        self.dataset: str = dataset

    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join(self.output_path, self.dataset)
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        split_to_filename: Dict[str, str] = {TEST_SPLIT: "test"}

        for split, filename in split_to_filename.items():
            url: str = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{self.dataset}.json"
            target_path: str = os.path.join(data_path, f"{self.dataset}_{filename}")
            ensure_file_downloaded(source_url=url, target_path=target_path, unpack=False)

            with open(target_path, "r") as f:
                data = json.load(f)
                
                for instance in data['examples']:
                    question = instance['input']
                    answer = instance['target']
                    
                    references: List[Reference] = [Reference(Output(text=answer), tags=[CORRECT_TAG])]
                    instance: Instance = Instance(
                                input=Input(text=question),
                                references=references,
                                split=split,
                            )
                    instances.append(instance)

        return instances

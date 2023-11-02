import json
import os
from typing import List, Dict

from helm.common.hierarchical_logger import hlog

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
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


class AddSubScenario(Scenario):

    name = "addsub"
    description = "AddSub Dataset"
    tags = ["question_answering"]

    def __init__(self):
        super().__init__()

    def get_instances(self) -> List[Instance]:
        def delete_extra_zero(n):
            try:
                n = float(n)
            except:
                hlog(f"None {n}")
                return n
            if isinstance(n, int):
                return str(n)
            if isinstance(n, float):
                n = str(n).rstrip("0")
                n = int(n.rstrip(".")) if n.endswith(".") else float(n)
                n = str(n)
                return n

        def make_train_set(data_path: str):
            train = [
                {
                    "iIndex": 0,
                    "sQuestion": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                    "lSolutions": [39],
                },
                {
                    "iIndex": 1,
                    "sQuestion": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                    "lSolutions": [6],
                },
                {
                    "iIndex": 2,
                    "sQuestion": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                    "lSolutions": [5],
                },
                {
                    "iIndex": 3,
                    "sQuestion": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                    "lSolutions": [9],
                },
                {
                    "iIndex": 4,
                    "sQuestion": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                    "lSolutions": [33],
                },
            ]

            with open(os.path.join(data_path, "train"), "w") as f:
                f.write(json.dumps(train, indent=4))

        data_path: str = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)

        url: str = "https://raw.githubusercontent.com/chuanyang-Zheng/Progressive-Hint/main/dataset/AddSub/AddSub.json"
        test_path: str = os.path.join(data_path, "test")
        ensure_file_downloaded(source_url=url, target_path=test_path, unpack=False)

        make_train_set(data_path)

        instances: List[Instance] = []
        split_to_filename: Dict[str, str] = {TRAIN_SPLIT: "train", TEST_SPLIT: "test"}

        for split, filename in split_to_filename.items():
            target_path: str = os.path.join(data_path, filename)

            with open(target_path, "r") as f:
                data = json.load(f)
                for entry in data:
                    question = entry["sQuestion"].strip()
                    answer = str(entry["lSolutions"][0])
                    if answer[-2:] == ".0":
                        answer = answer[:-2]
                    instance: Instance = Instance(
                        input=Input(text=question),
                        references=[Reference(Output(text=delete_extra_zero(answer)), tags=[CORRECT_TAG])],
                        split=split,
                    )
                    instances.append(instance)

        return instances

"""STL model builder: AutoModelForSequenceClassification."""
from transformers import AutoModelForSequenceClassification, AutoConfig

from ..common.config import TASK_CONFIGS


def build_stl_model(model_id: str, task_key: str):
    task_meta = TASK_CONFIGS[task_key]
    problem_type = ("regression" if task_meta["task_type"] == "regression"
                    else "single_label_classification")
    cfg = AutoConfig.from_pretrained(
        model_id,
        num_labels=task_meta["num_labels"],
        problem_type=problem_type,
    )
    return AutoModelForSequenceClassification.from_pretrained(model_id, config=cfg)

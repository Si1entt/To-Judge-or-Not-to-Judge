"""Post fact judgment task."""
from experiments.common import Experiment as BaseExperiment


class Experiment(BaseExperiment):
    post_fact = True

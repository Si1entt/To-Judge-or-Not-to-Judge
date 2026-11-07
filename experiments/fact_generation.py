"""Fact generation task."""
from experiments.common import Experiment as BaseExperiment


class Experiment(BaseExperiment):
    fact = True

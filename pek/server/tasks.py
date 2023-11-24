import uuid
from abc import ABC
from enum import StrEnum

from ..clustering import (
    ProgressiveEnsembleElbowProcess,
    ProgressiveEnsembleKMeansProcess,
)
from ..data import DatasetLoader


class TaskStatus(StrEnum):
    pending = "pending"
    running = "running"
    paused = "paused"
    killed = "killed"
    completed = "completed"


class _Task(ABC):
    def __init__(self, queue):
        self.id = str(uuid.uuid4())
        self.queue = queue
        self.status = TaskStatus.pending
        self.process = None

    def start(self):
        if self.status != TaskStatus.pending:
            raise RuntimeError(f"Task {self.id} has already been started.")
        self.process.start()
        self.status = TaskStatus.running

    def pause(self):
        if self.status != TaskStatus.running:
            raise RuntimeError(f"Task {self.id} is not running.")
        self.process.pause()
        self.status = TaskStatus.paused

    def resume(self):
        if self.status != TaskStatus.paused:
            raise RuntimeError(f"Task {self.id} is not paused.")
        self.process.resume()
        self.status = TaskStatus.running

    def kill(self):
        if self.status != TaskStatus.running:
            raise RuntimeError(f"Task {self.id} is not running.")
        self.process.kill()
        self.status = TaskStatus.killed


class EnsembleTask(_Task):
    def __init__(self, args, queue):
        super().__init__(queue)
        self.id = "ENS-" + self.id

        X = DatasetLoader.load(args["dataset"]).data
        args["resultsQueue"] = queue
        args["taskId"] = self.id
        self.process = ProgressiveEnsembleKMeansProcess(X, **args)

    def killRun(self, runId):
        if self.status != TaskStatus.running:
            raise RuntimeError(f"Task {self.id} is not running.")
        self.process.killRun(runId)


class ElbowTask(_Task):
    def __init__(self, args, queue):
        super().__init__(queue)
        self.id = "ELB-" + self.id

        X = DatasetLoader.load(args["dataset"]).data
        args["resultsQueue"] = queue
        args["taskId"] = self.id
        self.process = ProgressiveEnsembleElbowProcess(X, **args)

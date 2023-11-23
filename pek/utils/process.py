import json
from abc import ABC, abstractmethod
from enum import StrEnum
from multiprocessing import Process, Queue

from sklearn.utils import Bunch
from sklearn.utils._param_validation import InvalidParameterError


class ProcessStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    KILLED = "killed"
    COMPLETED = "completed"


class ProcessControlMessageType(StrEnum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    KILL = "kill"
    KILL_RUN = "kill_run"


class _ProcessControlMessage:
    def __init__(self, messageType, messageData=None):
        self.messageType = messageType
        self.messageData = messageData

        if messageType not in ProcessControlMessageType:
            raise InvalidParameterError(f"The param messageType is not valid. Must be one in ProcessControlMessageType")

    def encode(self):
        return str(self)

    def __str__(self):
        return json.dumps({"messageType": self.messageType, "messageData": self.messageData})


class ProcessControlMessage(ABC):
    @staticmethod
    def START():
        return _ProcessControlMessage(ProcessControlMessageType.START)

    @staticmethod
    def PAUSE():
        return _ProcessControlMessage(ProcessControlMessageType.PAUSE)

    @staticmethod
    def RESUME():
        return _ProcessControlMessage(ProcessControlMessageType.RESUME)

    @staticmethod
    def KILL():
        return _ProcessControlMessage(ProcessControlMessageType.KILL)

    @staticmethod
    def KILL_RUN(runId):
        return _ProcessControlMessage(ProcessControlMessageType.KILL_RUN, messageData=Bunch(runId=runId))

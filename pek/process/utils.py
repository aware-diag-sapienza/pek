from enum import StrEnum


class ProcessStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    KILLED = "killed"
    COMPLETED = "completed"


class ProcessControlMessage(StrEnum):
    KILL = "kill"
    PAUSE = "pause"
    RESUME = "resume"

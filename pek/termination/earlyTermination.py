from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from sklearn.utils._param_validation import InvalidParameterError


def _check_output_action(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result not in EarlyTerminationAction:
            raise TypeError(
                f"The output of {func.__name__} must be an element of {EarlyTerminationAction.__name__}. Got '{result}' instead."
            )
        return result

    return wrapper


class EarlyTerminationAction(Enum):
    NONE = None
    NOTIFY = "notify"
    KILL = "kill"


class AbstractEarlyTerminator(ABC):
    """Early Terminator interface."""

    def __init__(self, name: str):
        self.name = name

    @_check_output_action
    @abstractmethod
    def checkEarlyTermination(self, partialResult):
        """Method called by the ensemble to check if early termination occurs, at each partial result.
        The implementation of this function must return a value from EarlyTerminationAction.
        """
        pass


class _ET(AbstractEarlyTerminator, ABC):
    """Generic early Terminator based on ratio inertia."""

    def __init__(self, name: str, threshold: float, action=EarlyTerminationAction.NOTIFY, minIteration=4):
        super().__init__(name)
        self.threshold = threshold
        self.minIteration = minIteration
        self.action = action
        self.lastInertia = None

        if action not in EarlyTerminationAction:
            raise InvalidParameterError(f"The action={action} does not exist as an EarlyTerminationAction.")

    def checkEarlyTermination(self, partialResult):
        currentInertia = partialResult.metrics.inertia

        if self.lastInertia is not None:
            ratioInertiaPrev = currentInertia / self.lastInertia
            if (np.abs(1 - ratioInertiaPrev) <= self.threshold) and partialResult.info.iteration >= 4:
                return self.action

        self.lastInertia = currentInertia
        return EarlyTerminationAction.NONE


class EarlyTerminatorKiller(_ET):
    """Early Terminator that kill the ensemble when the termination occurs."""

    def __init__(self, name: str, threshold: float, minIteration=4):
        super().__init__(name, threshold, EarlyTerminationAction.KILL, minIteration)


class EarlyTerminatorNotifier(_ET):
    """Early Terminator that notify the ensemble when the termination occurs."""

    def __init__(self, name: str, threshold: float, minIteration=4):
        super().__init__(name, threshold, EarlyTerminationAction.NOTIFY, minIteration)


__all__ = [
    "AbstractEarlyTerminator",
    "EarlyTerminationAction",
    "EarlyTerminatorKiller",
    "EarlyTerminatorNotifier",
]

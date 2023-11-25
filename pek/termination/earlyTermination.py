from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from sklearn.utils import Bunch
from sklearn.utils._param_validation import InvalidParameterError

_T_SLOW = 5.256791e-05
_T_FAST = 0.004361538


def checkEarlyTerminationAction(action):
    """Checks if the action is a valid EarlyTerminationAction. Returns the action otherwise raises an exception."""
    if action not in EarlyTerminationAction:
        raise TypeError(f"The action '{action}' must be an element of {EarlyTerminationAction.__name__}.")
    return action


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


class _EarlyTerminatorRatioInertia(AbstractEarlyTerminator):
    """Generic early Terminator based on ratio inertia."""

    def __init__(self, name: str, threshold: float, action=EarlyTerminationAction.NOTIFY, minIteration=5):
        super().__init__(name)
        self.threshold = threshold
        self.minIteration = minIteration
        self.action = action

        self._lastInertia = None

        if action not in EarlyTerminationAction:
            raise InvalidParameterError(f"The action={action} does not exist as an EarlyTerminationAction.")

    def checkEarlyTermination(self, partialResult):
        currentInertia = partialResult.info.inertia

        if self._lastInertia is not None:
            ratioInertiaPrev = currentInertia / self._lastInertia
            if (np.abs(1 - ratioInertiaPrev) <= self.threshold) and partialResult.info.iteration >= self.minIteration:
                return self.action

        self._lastInertia = currentInertia
        return EarlyTerminationAction.NONE


class _EarlyTerminatorKiller(_EarlyTerminatorRatioInertia):
    """Early Terminator that kills the ensemble when the termination occurs."""

    def __init__(self, name: str, threshold: float, minIteration=5):
        super().__init__(name, threshold, EarlyTerminationAction.KILL, minIteration)


class _EarlyTerminatorNotifier(_EarlyTerminatorRatioInertia):
    """Early Terminator that notifies the ensemble when the termination occurs."""

    def __init__(self, name: str, threshold: float, minIteration=5):
        super().__init__(name, threshold, EarlyTerminationAction.NOTIFY, minIteration)


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

_DEFAULT_ET = {
    "fast-notify": _EarlyTerminatorNotifier("fast-notify", _T_FAST),
    "fast-kill": _EarlyTerminatorKiller("fast-kill", _T_FAST),
    "slow-notify": _EarlyTerminatorNotifier("slow-notify", _T_SLOW),
    "slow-kill": _EarlyTerminatorKiller("slow-kill", _T_SLOW),
}


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


class EarlyTerminatorKiller:
    """Early Terminator that kills the ensemble when the termination occurs."""

    SLOW = _DEFAULT_ET["slow-kill"]
    FAST = _DEFAULT_ET["fast-kill"]

    @staticmethod
    def CUSTOM(name: str, threshold: float, minIteration=5):
        return _EarlyTerminatorKiller(name, threshold, minIteration=minIteration)


class EarlyTerminatorNotifier:
    """Early Terminator that notifies the ensemble when the termination occurs."""

    SLOW = _DEFAULT_ET["slow-notify"]
    FAST = _DEFAULT_ET["fast-notify"]

    @staticmethod
    def CUSTOM(name: str, threshold: float, minIteration=5):
        return _EarlyTerminatorNotifier(name, threshold, minIteration=minIteration)


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


def _check_et_list(ets):
    if ets is None:
        return []
    elif isinstance(ets, list):
        return [_check_et(x) for x in ets]
    raise InvalidParameterError(f"The 'ets' parameter must be a list of instances of or None")


def _check_et(d):
    if isinstance(d, AbstractEarlyTerminator):
        return d
    elif isinstance(d, str) and (d in _DEFAULT_ET):
        return _DEFAULT_ET[d]
    elif isinstance(d, dict) or isinstance(d, Bunch):
        if ("name" in d) and ("threshold" in d) and ("action" in d):
            return _EarlyTerminatorRatioInertia(d["name"], d["threshold"], d["action"])
    raise InvalidParameterError(f"The 'et' parameter is invalid.")


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

__all__ = [
    "AbstractEarlyTerminator",
    "EarlyTerminationAction",
    "EarlyTerminatorKiller",
    "EarlyTerminatorNotifier",
]

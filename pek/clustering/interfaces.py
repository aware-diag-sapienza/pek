from abc import ABC, abstractmethod

from ..results.ensemble import EnsemblePartialResult
from ..results.run import RunPartialResult


class ProgressiveClusteringRun(ABC):
    """
    The ProgressiveClusteringRun interface declares the hasNextIteration() and  executeNextIteration() methods.
    """

    @abstractmethod
    def hasNextIteration(self) -> bool:
        pass

    @abstractmethod
    def executeNextIteration(self) -> RunPartialResult:
        pass


class ProgressiveClusteringEnsemble(ABC):
    """
    The ProgressiveClusteringEnsemble interface declares the hasNextIteration() and  executeNextIteration() methods.
    """

    @abstractmethod
    def hasNextIteration(self) -> bool:
        pass

    @abstractmethod
    def executeNextIteration(self) -> EnsemblePartialResult:
        pass


__all = [ProgressiveClusteringRun, ProgressiveClusteringEnsemble]
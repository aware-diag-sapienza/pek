from abc import ABC, abstractmethod

from .results import EnsemblePartialResult, RunPartialResult


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

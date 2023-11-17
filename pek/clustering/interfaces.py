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

    @abstractmethod
    def kill(self):
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

    @abstractmethod
    def executeAllIterations(self) -> EnsemblePartialResult:
        pass

    @abstractmethod
    def kill(self):
        pass

    @abstractmethod
    def killRun(self, run):
        pass


class ProgressiveClusteringEnsembleElbow(ABC):
    """
    The ProgressiveClusteringEnsembleElbow interface declares the hasNextIteration() and  executeNextIteration() methods.
    """

    @abstractmethod
    def hasNextIteration(self) -> bool:
        pass

    @abstractmethod
    def executeNextIteration(self) -> EnsemblePartialResult:
        pass

    @abstractmethod
    def executeAllIterations(self) -> EnsemblePartialResult:
        pass

    @abstractmethod
    def kill(self):
        pass


__all = [ProgressiveClusteringRun, ProgressiveClusteringEnsemble]

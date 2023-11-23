from .listener import ResultsListener
from .tasks import ElbowTask, EnsembleTask
from .wss import WebSocketServer


class PEKServer:
    def __init__(self, port=21000):
        self.name = self.__class__.__name__
        self.port = port
        self.rls = ResultsListener(self)
        self.wss = WebSocketServer(self, port=port)
        self.tasks = {}

    def start(self):
        self.rls.start()
        self.wss.start()

        self.rls.join()
        self.wss.join()

    def createEnsembleTask(self, args):
        task = EnsembleTask(args=args, queue=self.rls.queue)
        self.tasks[task.id] = task
        return task.id

    def createElbowTask(self, args):
        task = ElbowTask(args=args, queue=self.rls.queue)
        self.tasks[task.id] = task
        return task.id

    def getTask(self, taskId):
        return self.tasks[taskId]

    def sendPartialResult(self, partialResult):
        taskId = partialResult.taskId

        self.wss.sendPartialResult(taskId, partialResult)

        if partialResult.info.last:
            del self.tasks[taskId]

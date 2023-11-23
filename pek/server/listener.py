from multiprocessing import Queue
from threading import Thread


class ResultsListener(Thread):
    def __init__(self, server):
        super().__init__()
        self.server = server
        self.queue = Queue()

    def run(self) -> None:
        while True:
            try:
                partialResult = self.queue.get()
                self.server.sendPartialResult(partialResult)
            except:
                pass

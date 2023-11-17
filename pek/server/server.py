import json
import time
import traceback
from multiprocessing import Queue
from threading import Thread

from sklearn.utils import Bunch

from ..data import BuiltInDatasetLoader

# from ..datasets import Dataset
from ._json_websocket_server import JsonWebSocketServer
from .jobs import ElbowJob, EnsembleJob

# from .jobs import AsyncJob, ElbowJob
from .log import Log


class PEKServer:
    def __init__(self, port, host="0.0.0.0"):
        self.port = port
        self.host = host
        self.partialResultsQueue = Queue()
        self.elbowResultsQueue = Queue()
        self.socketServer = JsonWebSocketServer(
            self.port, host=self.host, fn_onMessage=self.onMessage, fn_onRequest=self.onRequest
        )
        # self.partialResultListener = Thread(target=self.fn_partialResultListener)
        self.jobs = {}
        self.jobsLastPartialResultTimestamp = {}

    def onMessage(self, client, messageId, message):
        try:
            if message.startswith("startJob:"):
                jobId = message.replace("startJob:", "")
                Log.print(f"{Log.GREEN}Starting {jobId}")
                self.jobs[jobId].start()
            ##
            elif message.startswith("stopJob:"):
                jobId = message.replace("stopJob:", "")
                Log.print(f"{Log.RED}Stopping {jobId}")
                self.jobs[jobId].stop()
                del self.jobs[jobId]
            ##
            elif message.startswith("pauseJob:"):
                jobId = message.replace("pauseJob:", "")
                Log.print(f"{Log.YELLOW}Pausing {jobId}")
                self.jobs[jobId].pause()
            ##
            elif message.startswith("resumeJob:"):
                jobId = message.replace("resumeJob:", "")
                Log.print(f"{Log.BLUE}Resuming {jobId}")
                self.jobs[jobId].resume()
            ##
            else:
                raise NameError(f"Undefined message type '{message}'")
            ##
        except Exception:
            traceback.print_exc()
            self.socketServer.sendMessage(client, traceback.format_exc())

    def onRequest(self, client, requestId, request):
        try:
            if request == "datasetNames":
                response = BuiltInDatasetLoader.allNames()
                self.socketServer.sendRequestResponse(client, requestId, response)
            ##
            elif request.startswith("dataset:"):
                datasetName = request.replace("dataset:", "")
                dataset = BuiltInDatasetLoader.load(datasetName)
                response = dataset.toDict(insertData=False)
                self.socketServer.sendRequestResponse(client, requestId, response)
            ##
            elif request.startswith("createEnsembleJob:"):
                d = Bunch(**json.loads(request.replace("createEnsembleJob:", "")))
                jobId = self.createEnsembleJob(client, requestId, d)
                self.socketServer.sendRequestResponse(client, requestId, jobId)
            ##
            elif request.startswith("createElbowJob:"):
                d = Bunch(**json.loads(request.replace("createElbowJob:", "")))
                jobId = self.createElbowJob(client, requestId, d)
                self.socketServer.sendRequestResponse(client, requestId, jobId)
            ##
            else:
                raise RuntimeError(f"Undefined request type '{request}'")
            ##
        except Exception:
            traceback.print_exc()
            self.socketServer.sendMessage(client, traceback.format_exc())

    def createEnsembleJob(self, client, requestId, d):
        job = EnsembleJob(
            d.type, d.dataset, d.k, d.r, d.s, self.partialResultsQueue, client=client, resultsMinFreq=d.resultsMinFreq
        )
        self.jobs[job.id] = job
        self.jobsLastPartialResultTimestamp[job.id] = None
        Log.print(f"{Log.PINK}Created EnsembleJob {job.id}")
        return job.id

    def createElbowJob(self, client, requestId, d):
        job = ElbowJob(
            d.type, d.dataset, d.kMin, d.kMax, d.r, d.s, self.partialResultsQueue, client=client, earlyTermination=d.et
        )
        self.jobs[job.id] = job
        self.jobsLastPartialResultTimestamp[job.id] = None
        Log.print(f"{Log.PINK}Created ElbowJob {job.id}")
        return job.id

    """def fn_partialResultListener(self):
        while True:
            data = self.partialResultsQueue.get()

            if data.jobType == "AsyncJob":
                jobId = data.pr.job_id
                if not data.pr.info.is_last:
                    Log.print(f"{Log.GRAY}Sending partial result #{data.pr.info.iteration} of {jobId}")
                    if data.pr.metrics.earlyTermination.fast:
                        Log.save(
                            "ET_FAST",
                            self.jobs[jobId],
                            iteration=data.pr.info.iteration,
                            clientAddress=self.socketServer.getClientAddress(data.client),
                        )
                    if data.pr.metrics.earlyTermination.slow:
                        Log.save(
                            "ET_SLOW",
                            self.jobs[jobId],
                            iteration=data.pr.info.iteration,
                            clientAddress=self.socketServer.getClientAddress(data.client),
                        )
                else:
                    Log.print(
                        f"{Log.GRAY}Sending partial result #{data.pr.info.iteration} of {jobId} -- {Log.RED}last{Log.ENDC}"
                    )
                    Log.save(
                        "END",
                        self.jobs[jobId],
                        iteration=data.pr.info.iteration,
                        clientAddress=self.socketServer.getClientAddress(data.client),
                    )

                if jobId in self.jobs:
                    self.resultsDelay(jobId, minFreq=self.jobs[jobId].resultsMinFreq)  # dealy

                self.socketServer.sendMessage(data.client, {"type": "partial-result", "data": data.pr})

            elif data.jobType == "ElbowJob":
                jobId = data.pr.jobId
                if not data.pr.isLast:
                    Log.print(f"{Log.GRAY}Sending elbow partial result K={data.pr.k} of {jobId}")
                    Log.save(
                        "ELBOW_PR",
                        self.jobs[jobId],
                        clientAddress=self.socketServer.getClientAddress(data.client),
                        elbowK=data.pr.k,
                    )
                else:
                    Log.print(
                        f"{Log.GRAY}Sending elbow partial result k={data.pr.k} of {jobId} -- {Log.RED}last{Log.ENDC}"
                    )
                    Log.save("END", self.jobs[jobId], clientAddress=self.socketServer.getClientAddress(data.client))

                self.socketServer.sendMessage(
                    data.client, {"type": "elbow-partial-result", "data": data.pr, "prova": data}
                )"""

    def resultsDelay(self, jobId, minFreq=None):
        lastTime = self.jobsLastPartialResultTimestamp[jobId]
        if minFreq is None or lastTime is None:
            self.jobsLastPartialResultTimestamp[jobId] = time.time()
            return
        currentTime = time.time()
        delay = max(0, minFreq - (currentTime - lastTime))
        if delay > 0:
            Log.print(f"Sleeping {delay} sec")
            time.sleep(delay)
        self.jobsLastPartialResultTimestamp[jobId] = time.time()

    def start(self):
        Log.print(f"Starting PEK Server on {Log.BLUE}ws://{self.host}:{self.port}")
        # self.partialResultListener.start()
        self.socketServer.start()  ##bloccante

import json
from threading import Thread

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, join_room
from sklearn.utils import Bunch

from ..data import DatasetLoader
from ..version import __version__
from .log import Log

"""
When creating an ensemble task or elbow task, the client is added to a room named as the taskId.
This speed up the sending of partial results because they are sent to the room that contain a single client.
"""


class WebSocketServer(Thread):
    def __init__(self, server, port=21000):
        super().__init__()
        self.server = server
        self.port = port

        self.app = None
        self.socketio = None

    def run(self) -> None:
        app = Flask(self.server.name)
        CORS(app, resources={r"/*": {"origins": "*"}})
        socketio = SocketIO(app, cors_allowed_origins="*")

        self.app = app
        self.socketio = socketio

        ############ STATIC DATA ###############

        @socketio.on("get-pek-version")
        def handle_get_pek_version(_):
            return __version__

        @socketio.on("get-datasets-list")
        def handle_get_datasets_list(_):
            return DatasetLoader.allNames()

        @socketio.on("get-dataset")
        def handle_get_dataset(datajson):
            d = Bunch(**json.loads(datajson))  # {'name': '...', 'insertData': bool}
            dataset = DatasetLoader.load(d.name)
            if dataset is None:
                return None
            return dataset.toJson(insertData=d.insertData)

        ############ TASK CREATION ###############

        @socketio.on("create-elbow-task")
        def handle_create_elbow_task(argsjson):
            args = Bunch(**json.loads(argsjson))
            taskId = self.server.createElbowTask(args)
            Log.print(f"Creating task.", taskId=taskId)
            join_room(taskId)
            return taskId

        @socketio.on("create-ensemble-task")
        def handle_create_ensemble_task(argsjson):
            args = Bunch(**json.loads(argsjson))
            taskId = self.server.createEnsembleTask(args)
            Log.print(f"Creating task.", taskId=taskId)
            join_room(taskId)
            return taskId

        ############ TASK ACTIONS ###############

        @socketio.on("start-task")
        def handle_start_task(datajson):
            d = Bunch(**json.loads(datajson))  # {'taskId': '...', 'args': {}}
            Log.print(f"{Log.BLUE}Starting task.", taskId=d.taskId)
            self.server.getTask(d.taskId).start()

        @socketio.on("pause-task")
        def handle_pause_task(datajson):
            d = Bunch(**json.loads(datajson))  # {'taskId': '...', 'args': {}}
            Log.print(f"{Log.YELLOW}Pausing task.", taskId=d.taskId)
            self.server.getTask(d.taskId).pause()

        @socketio.on("resume-task")
        def handle_resume_task(datajson):
            d = Bunch(**json.loads(datajson))  # {'taskId': '...', 'args': {}}
            Log.print(f"{Log.YELLOW}Resuming task.", taskId=d.taskId)
            self.server.getTask(d.taskId).resume()

        @socketio.on("kill-task")
        def handle_kill_task(datajson):
            d = Bunch(**json.loads(datajson))  # {'taskId': '...', 'args': {}}
            Log.print(f"{Log.RED}Killing task.", taskId=d.taskId)
            self.server.getTask(d.taskId).kill()

        @socketio.on("kill-ensemble-task-run")
        def handle_kill_ensemble_task_run(datajson):
            d = Bunch(**json.loads(datajson))  # {'taskId': '...', 'args': {'runId': '...'} }
            Log.print(f"{Log.RED}Killing run #{d.args.runId}", taskId=d.taskId)
            self.server.getTask(d.taskId).killRun(d.args.runId)

        socketio.run(app, port=self.port, host="0.0.0.0")

    def sendPartialResult(self, taskId, partialResult):
        self.socketio.emit(taskId, partialResult.toJson(), to=taskId)

        if taskId.startswith("ENS"):
            Log.print(
                f"{Log.BLUE}Sending pr#{partialResult.info.iteration}{Log.ENDC} --- info={partialResult.info} et={partialResult.earlyTermination}",
                taskId=taskId,
            )
        elif taskId.startswith("ELB"):
            Log.print(
                f"{Log.BLUE}Sending pr#{partialResult.info.iteration}{Log.ENDC} --- info={partialResult.info}",
                taskId=taskId,
            )

        else:
            raise RuntimeError(f"Undefined type of task {taskId}")

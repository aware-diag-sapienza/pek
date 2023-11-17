import json
import time

import numpy as np

from ._websocket_server_lib import WebsocketServer
from .log import Log


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    # pylint: disable=method-hidden
    def default(self, obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, float) and (np.isneginf(obj) or np.isneginf(obj)):
            return None
        elif isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class JsonWebSocketServer:
    def __init__(self, port, host="0.0.0.0", fn_onMessage=None, fn_onRequest=None):
        self.port = port
        self.host = host
        self.__clients = {}
        self.__server = WebsocketServer(self.port, host=self.host)
        self.__server.set_fn_new_client(self.__fnNewClient)
        self.__server.set_fn_client_left(self.__fnClientLeft)
        self.__server.set_fn_message_received(self.__fnMmessageReceived)
        self.__sentCounter = 0
        self.__recievedCounter = 0

        self.__fn_onMessage = fn_onMessage if fn_onMessage is not None else self.onMessage
        self.__fn_onRequest = fn_onRequest if fn_onRequest is not None else self.onRequest

    def __fnNewClient(self, client, server):
        Log.print(f"Client connected: {client['id']}@{client['address'][0]}:{client['address'][1]}")
        self.__clients[client["id"]] = client

    def __fnClientLeft(self, client, server):
        Log.print(f"Client disconnected: {client['id']}@{client['address'][0]}:{client['address'][1]}")
        del self.__clients[client["id"]]

    def __fnMmessageReceived(self, client, server, message):
        self.__recievedCounter += 1
        obj = json.loads(message)
        # print(f"Incoming message: {obj}")
        if obj["type"] == "M":
            self.__fn_onMessage(client["id"], obj["id"], obj["body"])
        elif obj["type"] == "R":
            self.__fn_onRequest(client["id"], obj["id"], obj["body"])
        elif obj["type"] == "RR":
            pass
        else:
            print(f"Undefined message type recieved: " + obj["type"])

    def __encodeOutgoingMessage(self, type, data, requestId=None):
        self.__sentCounter += 1
        messageId = self.__sentCounter
        message = {"type": type, "timestamp": time.time(), "id": messageId, "body": data}
        if requestId is not None:
            message["requestId"] = requestId
        messageString = json.dumps(message, cls=NumpyEncoder)
        return messageId, messageString

    def getClientAddress(self, clientId):
        client = self.__clients[clientId]
        addr = f"{client['address'][0]}:{client['address'][1]}"
        return addr

    def start(self):
        self.__server.run_forever()

    def sendMessage(self, client, data):
        c = client if not isinstance(client, int) else self.__clients[client]
        messageId, messageString = self.__encodeOutgoingMessage("M", data)
        self.__server.send_message(c, messageString)

    def sendRequest(self, client, data):
        c = client if not isinstance(client, int) else self.__clients[client]
        messageId, messageString = self.__encodeOutgoingMessage("R", data)
        self.__server.send_message(c, messageString)

    def sendRequestResponse(self, client, requestId, data):
        c = client if not isinstance(client, int) else self.__clients[client]
        messageId, messageString = self.__encodeOutgoingMessage("RR", data, requestId=requestId)
        self.__server.send_message(c, messageString)

    def onMessage(self, client, messageId, message):
        pass

    def onRequest(self, client, requestId, request):
        pass

from __future__ import annotations

import json
import socketserver
import threading


DEFAULT_DETECTION = {
    "version": 1,
    "type": "detection_result",
    "status": "ok",
    "left": {
        "status": "BAD",
        "class_name": "bad_tire",
        "confidence": 0.91,
        "stable": True,
        "votes": 3,
        "frames": 3,
        "detections": [],
    },
    "right": {
        "status": "GOOD",
        "class_name": "good_tire",
        "confidence": 0.88,
        "stable": True,
        "votes": 3,
        "frames": 3,
        "detections": [],
    },
}


class VisionHandler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline()
        if self.server.raw_response is not None:
            self.wfile.write(self.server.raw_response)
            return
        request = json.loads(line.decode("utf-8"))
        if request["type"] == "health":
            response = {
                "version": 1,
                "type": "health",
                "request_id": request["request_id"],
                "status": "ok",
                "health": {"ok": True},
            }
        else:
            response = dict(self.server.detection_response)
            response["request_id"] = request["request_id"]
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class FakeVisionServer:
    def __init__(self, response=None, raw_response=None):
        class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server(("127.0.0.1", 0), VisionHandler)
        self.server.detection_response = response or DEFAULT_DETECTION
        self.server.raw_response = raw_response
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def host(self):
        return self.server.server_address[0]

    @property
    def port(self):
        return self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

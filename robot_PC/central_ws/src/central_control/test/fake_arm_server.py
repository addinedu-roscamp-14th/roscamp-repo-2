from __future__ import annotations

import socketserver
import threading
import time


class ArmHandler(socketserver.StreamRequestHandler):
    def handle(self):
        command = self.rfile.readline().decode("utf-8").strip()
        if self.server.delay_sec:
            time.sleep(self.server.delay_sec)
        if self.server.fail_commands and command in self.server.fail_commands:
            self.wfile.write(f"ERR:{command}\n".encode("utf-8"))
        else:
            self.wfile.write(f"OK:{command}\n".encode("utf-8"))


class FakeArmServer:
    def __init__(self, fail_commands=None, delay_sec=0.0):
        class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server(("127.0.0.1", 0), ArmHandler)
        self.server.fail_commands = set(fail_commands or [])
        self.server.delay_sec = float(delay_sec)
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

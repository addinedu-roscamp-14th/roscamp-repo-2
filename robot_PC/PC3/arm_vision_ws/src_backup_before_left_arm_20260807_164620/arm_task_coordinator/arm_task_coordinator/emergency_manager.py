from __future__ import annotations
import threading


class EmergencyActive(RuntimeError):
    pass


class EmergencyManager:
    def __init__(self, clients):
        self.clients = dict(clients)
        self._latched = threading.Event()
        self.reason = ""

    @property
    def active(self):
        return self._latched.is_set()

    def require_clear(self):
        if self.active:
            raise EmergencyActive(f"emergency active: {self.reason}")

    def trigger(self, reason):
        self.reason = str(reason)
        self._latched.set()
        results = {}
        threads = []

        def stop(side, client):
            try:
                results[side] = client.stop()
            except Exception as error:
                results[side] = error

        for side, client in self.clients.items():
            thread = threading.Thread(target=stop, args=(side, client), daemon=True)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)
        return results

    def reset(self):
        self.reason = ""
        self._latched.clear()

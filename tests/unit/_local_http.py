"""A local HTTP server for download tests, recording what it sent.

Paths: /small, /refused (403 page), /big (streams 200 MB at about 25 MB/s
until the client goes away), /trickle (one byte every 50 ms), /stall (64 KiB then silence),
/hang (an image that never finishes), /script-download (a page whose script
starts a /big download while /hang keeps it loading), /blocked (403 bot page),
/redirect/<n> (n redirects, then /small), /cookie-start (sets a cookie
and redirects to /cookie-file, which needs it).
"""

import http.server
import threading
import time

MB = 1024 * 1024


class Server:
    def __init__(self):
        self.sent = {}            # path -> bytes written
        self.disconnected = {}    # path -> threading.Event, set on BrokenPipe
        self.started = {}         # path -> threading.Event, set after the first write
        self.headers = {}         # path -> request headers
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _write(self, data):
                self.wfile.write(data)
                self.wfile.flush()
                # setdefault: reset() may clear the dicts while a handler of
                # the previous test is still writing
                server.sent[self.path] = server.sent.get(self.path, 0) + len(data)
                server.started.setdefault(self.path, threading.Event()).set()

            def _head(self, status=200, ctype="application/octet-stream", length=None, attach=True):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                if attach:
                    self.send_header("Content-Disposition", "attachment; filename=f.pdf")
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.end_headers()

            def do_GET(self):
                server.headers[self.path] = dict(self.headers)
                server.started.setdefault(self.path, threading.Event())
                server.disconnected.setdefault(self.path, threading.Event())
                try:
                    self._route()
                except (BrokenPipeError, ConnectionResetError):
                    server.disconnected.setdefault(self.path, threading.Event()).set()

            def _route(self):
                if self.path.startswith("/redirect/"):
                    left = int(self.path.rsplit("/", 1)[1])
                    self.send_response(302)
                    self.send_header("Location", f"/redirect/{left - 1}" if left > 1 else "/small")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path == "/cookie-start":
                    self.send_response(302)
                    self.send_header("Set-Cookie", "session=ok; Path=/")
                    self.send_header("Location", "/cookie-file")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path == "/cookie-file":
                    if "session=ok" in self.headers.get("Cookie", ""):
                        body = b"%PDF-1.4 cookie"
                        self._head(length=len(body))
                        self._write(body)
                    else:
                        self._head(status=401, ctype="text/plain", length=0, attach=False)
                elif self.path == "/auth-start":
                    self.send_response(302)
                    self.send_header("Location", "/auth-file")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path == "/auth-file":
                    if self.headers.get("Authorization") == "Basic dXNlcjpwYXNz":  # user:pass
                        body = b"%PDF-1.4 auth"
                        self._head(length=len(body))
                        self._write(body)
                    else:
                        self._head(status=401, ctype="text/plain", length=0, attach=False)
                elif self.path == "/small":
                    body = b"%PDF-1.4 small"
                    self._head(length=len(body))
                    self._write(body)
                elif self.path == "/big":
                    self._head()
                    # about 25 MB/s, so a client that stops early is visible
                    while server.sent.get(self.path, 0) < 200 * MB:
                        self._write(b"0" * (256 * 1024))
                        time.sleep(0.01)
                elif self.path == "/trickle":
                    self._head(length=10 * MB)
                    for _ in range(600):
                        self._write(b"0")
                        time.sleep(0.05)
                elif self.path == "/stall":
                    self._head(length=10 * MB)
                    self._write(b"%PDF-" + b"0" * 64 * 1024)
                    time.sleep(30)
                elif self.path == "/hang":
                    self._head(ctype="image/png", length=100000, attach=False)
                    time.sleep(30)
                elif self.path == "/script-download":
                    body = (b"<html><head><title>loading</title></head><body><img src='/hang'>"
                            b"<script>const a = document.createElement('a'); a.href = '/big';"
                            b" a.download = 'f.pdf'; document.body.appendChild(a); a.click();</script>"
                            b"</body></html>")
                    self._head(ctype="text/html", length=len(body), attach=False)
                    self._write(body)
                else:  # /refused, /blocked
                    body = b"<html><head><title>Just a moment...</title></head><body>check</body></html>"
                    self._head(status=403, ctype="text/html", length=len(body), attach=False)
                    self._write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def reset(self):
        self.sent.clear()
        self.disconnected.clear()
        self.started.clear()

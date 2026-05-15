import socket
import struct
import threading
import json


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_frame(sock):
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack("!I", header)
    return _recv_exact(sock, length)


def send_frame(sock, body):
    sock.sendall(struct.pack("!I", len(body)) + body)


class MainServer():

    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 5000

        self.clients = {}
        self.clients_lock = threading.Lock()

        self.server_main()

    def handle_client(self, client_socket, addr):
        print(f"[server] handshake waiting for name from {addr}")
        name_bytes = recv_frame(client_socket)
        if name_bytes is None:
            print(f"[server] {addr} closed before handshake")
            client_socket.close()
            return
        client_name = name_bytes.decode("utf-8")
        print(f"[server] Client {client_name!r} connected from {addr}")
        with self.clients_lock:
            self.clients[client_name] = client_socket

        msg_count = 0
        try:
            while True:
                body = recv_frame(client_socket)
                if body is None:
                    break

                msg_count += 1
                try:
                    message_data = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    print(f"[server] malformed JSON from {client_name}, skipping frame ({len(body)} bytes)")
                    continue

                receiver = message_data.get("To")
                if msg_count <= 5 or msg_count % 50 == 0:
                    print(f"[server] {client_name} -> {receiver} ({len(body)} bytes, #{msg_count})")
                with self.clients_lock:
                    target = self.clients.get(receiver)

                if target is None:
                    print(f"[server] {receiver!r} not connected (from {client_name})")
                    continue

                try:
                    send_frame(target, body)
                except OSError as e:
                    print(f"[server] forward to {receiver} failed: {e}")
        except (ConnectionResetError, OSError) as e:
            print(f"[server] {client_name} socket error: {e}")

        print(f"[server] Client {client_name} disconnected after {msg_count} messages.")
        with self.clients_lock:
            self.clients.pop(client_name, None)
        try:
            client_socket.close()
        except OSError:
            pass

    def server_main(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            reachable_ip = probe.getsockname()[0]
        finally:
            probe.close()
        print(f"Server listening on {self.host}:{self.port}")
        print(f"Connect Windows clients to: {reachable_ip}:{self.port}")

        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"[server] accept() from {addr}")
            threading.Thread(
                target=self.handle_client,
                args=(client_socket, addr),
                daemon=True,
            ).start()


server = MainServer()

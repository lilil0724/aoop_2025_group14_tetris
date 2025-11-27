import socket
import pickle
import threading
import time

class NetworkManager:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = None
        self.conn = None # For server use
        self.is_server = False
        self.connected = False
        self.latest_remote_data = None
        self.lock = threading.Lock()
        self.running = True

    def host_game(self, port=5555):
        self.is_server = True
        try:
            # Get local IP to display
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"Hosting on {local_ip}:{port}")
            
            self.client.bind(("0.0.0.0", port))
            self.client.listen(1)
            self.client.settimeout(0.2) # Non-blocking accept loop
            
            start_time = time.time()
            while self.running:
                try:
                    self.conn, self.addr = self.client.accept()
                    self.connected = True
                    print(f"Connected to {self.addr}")
                    threading.Thread(target=self._receive_loop, daemon=True).start()
                    return True
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Host error: {e}")
                    return False
            return False
        except Exception as e:
            print(f"Bind error: {e}")
            return False

    def join_game(self, ip, port=5555):
        self.is_server = False
        try:
            self.client.connect((ip, port))
            self.connected = True
            threading.Thread(target=self._receive_loop, daemon=True).start()
            return True
        except Exception as e:
            print(f"Join error: {e}")
            return False

    def send(self, data):
        if not self.connected: return
        try:
            serialized = pickle.dumps(data)
            length = len(serialized).to_bytes(4, byteorder='big')
            target = self.conn if self.is_server else self.client
            target.sendall(length + serialized)
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False

    def _receive_loop(self):
        target = self.conn if self.is_server else self.client
        while self.running and self.connected:
            try:
                length_bytes = self._recv_all(target, 4)
                if not length_bytes: break
                length = int.from_bytes(length_bytes, byteorder='big')
                
                data_bytes = self._recv_all(target, length)
                if not data_bytes: break
                
                data = pickle.loads(data_bytes)
                with self.lock:
                    self.latest_remote_data = data
            except Exception as e:
                print(f"Receive error: {e}")
                break
        self.connected = False

    def _recv_all(self, sock, n):
        data = b''
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet: return None
                data += packet
            except:
                return None
        return data

    def get_latest_data(self):
        with self.lock:
            return self.latest_remote_data

    def close(self):
        self.running = False
        if self.conn: 
            try: self.conn.close()
            except: pass
        try: self.client.close()
        except: pass

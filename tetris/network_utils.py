import socket
import pickle
import threading
import time
import select

class NetworkManager:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.is_server = False
        self.connected = False
        self.my_id = None
        self.players = {} # {id: data}
        self.lock = threading.Lock()
        self.running = True
        
        # Garbage tracking
        self.total_garbage_sent = 0
        self.total_garbage_received = 0 # From server
        
        # Server specific
        self.clients = {} # {conn: id}
        self.next_id = 1
        self.server_garbage_tracking = {} # {id: last_sent_count}
        self.server_garbage_received = {} # {id: total_received}
        
        self.game_started = False # [NEW] Game start flag

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            try:
                return socket.gethostbyname(socket.gethostname())
            except:
                return "127.0.0.1"

    def host_game(self, port=5555, max_players=2):
        self.is_server = True
        self.my_id = 0
        try:
            local_ip = self.get_local_ip()
            print(f"Hosting on {local_ip}:{port}")
            
            self.client.bind(("0.0.0.0", port))
            self.client.listen(max_players - 1)
            self.client.settimeout(0.2)
            
            print("Server started. If clients cannot connect, please check your FIREWALL settings.")
            print("Ensure python.exe is allowed to accept incoming connections.")
            
            self.server_garbage_tracking[0] = 0
            self.server_garbage_received[0] = 0
            
            threading.Thread(target=self._server_loop, args=(max_players,), daemon=True).start()
            return True
        except Exception as e:
            print(f"Bind error: {e}")
            return False

    def join_game(self, ip, port=5555):
        self.is_server = False
        try:
            self.client.settimeout(5.0) # Set timeout for connection
            self.client.connect((ip, port))
            
            # Wait for handshake (init packet)
            msg = self._recv_packet(self.client)
            if msg and msg.get('type') == 'init':
                self.my_id = msg['id']
                print(f"Assigned ID: {self.my_id}")
                self.connected = True
                self.client.settimeout(None) # Reset timeout to blocking (or default)
                threading.Thread(target=self._client_loop, daemon=True).start()
                return True
            else:
                print("Handshake failed")
                self.client.close()
                return False
        except Exception as e:
            print(f"Join error: {e}")
            return False

    def start_game(self):
        """ Host calls this to signal game start """
        if not self.is_server: return
        self.game_started = True
        for sock in self.clients:
            try:
                self._send_packet(sock, {'type': 'start'})
            except:
                pass

    def send(self, data):
        if not self.connected and not self.is_server: return
        
        # Inject garbage count
        data['total_garbage_sent'] = self.total_garbage_sent
        
        if self.is_server:
            # Update local state directly
            with self.lock:
                self.players[0] = data
        else:
            try:
                self._send_packet(self.client, data)
            except:
                self.connected = False

    def _send_packet(self, sock, data):
        serialized = pickle.dumps(data)
        length = len(serialized).to_bytes(4, byteorder='big')
        sock.sendall(length + serialized)

    def _recv_packet(self, sock):
        try:
            length_bytes = self._recv_all(sock, 4)
            if not length_bytes: return None
            length = int.from_bytes(length_bytes, byteorder='big')
            data_bytes = self._recv_all(sock, length)
            if not data_bytes: return None
            return pickle.loads(data_bytes)
        except:
            return None

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

    def _server_loop(self, max_players):
        print("Server loop started")
        self.connected = True
        
        while self.running:
            # 1. Accept new connections
            if len(self.clients) < max_players - 1:
                try:
                    conn, addr = self.client.accept()
                    print(f"New connection: {addr}")
                    pid = self.next_id
                    self.next_id += 1
                    self.clients[conn] = pid
                    self.server_garbage_tracking[pid] = 0
                    self.server_garbage_received[pid] = 0
                    
                    # Send assigned ID
                    self._send_packet(conn, {'type': 'init', 'id': pid, 'max_players': max_players})
                except socket.timeout:
                    pass
                except Exception as e:
                    print(f"Accept error: {e}")

            # 2. Receive from clients
            rlist = list(self.clients.keys())
            if rlist:
                read_sockets, _, _ = select.select(rlist, [], [], 0.01)
                for sock in read_sockets:
                    data = self._recv_packet(sock)
                    if data:
                        pid = self.clients[sock]
                        with self.lock:
                            self.players[pid] = data
                    else:
                        # Disconnected
                        print(f"Client {self.clients[sock]} disconnected")
                        del self.clients[sock]

            # 3. Process Garbage Distribution
            self._process_garbage_distribution()

            # 4. Broadcast State
            with self.lock:
                # Inject garbage received info into each player's state
                broadcast_state = {}
                for pid, p_data in self.players.items():
                    p_data_copy = p_data.copy()
                    p_data_copy['total_garbage_received'] = self.server_garbage_received.get(pid, 0)
                    broadcast_state[pid] = p_data_copy
            
            for sock in self.clients:
                try:
                    self._send_packet(sock, {'type': 'state', 'data': broadcast_state})
                except:
                    pass # Handle disconnection in next loop

            time.sleep(0.016) # ~60 FPS

    def _process_garbage_distribution(self):
        import random
        with self.lock:
            active_ids = [0] + list(self.clients.values())
            # Filter only alive players? For now assume all connected are candidates
            # Ideally we should check 'game_over' flag in self.players
            
            alive_ids = [pid for pid in active_ids if pid in self.players and not self.players[pid].get('game_over', False)]
            if len(alive_ids) <= 1: return # No one to attack

            for pid in active_ids:
                if pid not in self.players: continue
                
                sent = self.players[pid].get('total_garbage_sent', 0)
                last = self.server_garbage_tracking.get(pid, 0)
                
                diff = sent - last
                if diff > 0:
                    self.server_garbage_tracking[pid] = sent
                    # Distribute 'diff' garbage
                    # Pick random target != pid
                    candidates = [t for t in alive_ids if t != pid]
                    if candidates:
                        target = random.choice(candidates)
                        self.server_garbage_received[target] = self.server_garbage_received.get(target, 0) + diff

    def _client_loop(self):
        while self.running:
            msg = self._recv_packet(self.client)
            if not msg:
                self.connected = False
                break
            
            # 'init' is now handled in join_game, but keep it just in case
            if msg['type'] == 'init':
                self.my_id = msg['id']
                print(f"Assigned ID: {self.my_id}")
            elif msg['type'] == 'start':
                self.game_started = True
            elif msg['type'] == 'state':
                with self.lock:
                    self.players = msg['data']
                    # Check for incoming garbage
                    if self.my_id in self.players:
                        server_recv = self.players[self.my_id].get('total_garbage_received', 0)
                        if server_recv > self.total_garbage_received:
                            # We have new garbage!
                            # This logic should be handled in main.py by checking get_garbage_diff()
                            pass

    def get_latest_data(self):
        with self.lock:
            return self.players.copy()

    def get_garbage_diff(self):
        """ Returns new garbage amount since last check """
        with self.lock:
            if self.my_id is not None and self.my_id in self.players:
                server_recv = self.players[self.my_id].get('total_garbage_received', 0)
                diff = server_recv - self.total_garbage_received
                if diff > 0:
                    self.total_garbage_received = server_recv
                    return diff
        return 0

    def close(self):
        self.running = False
        if self.client: self.client.close()


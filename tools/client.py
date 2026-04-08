import socket
import struct
import time
import numpy as np
from matplotlib import pyplot as plt
import math

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost", 8081))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
i = 1
sleep_factor = 0
photo_chunk_size = 4096 * 64
photo = True

actionPacket = struct.pack("ffffff??", 0.05, 0.8, 0.0, 0.0, 0.0, 0.0, False, photo)
s.sendall(actionPacket)

while True:
    start_time = time.time()
    data = s.recv(1024)
    # unpacked = struct.unpack("ffffffffffffffffffffffffffffff", data)
    total_data = b""
    if photo:
        s.sendall(bytes("OK", 'ascii'))
        data = s.recv(1024)
        size_data = data.decode('ascii')
        size = int(size_data[5:])
        # print(size, "\n\n\n\n\n\n\n")
        s.sendall(bytes("OK", 'ascii'))
        total_received = 0

    actionPacket = struct.pack("ffffff??", 0.1, 0.8, 0.0, 0.0, 0.0, 0.0, False, photo)
    s.sendall(actionPacket)
    if photo:
        while total_received < size:
            data = s.recv(min(photo_chunk_size, size - total_received))
            total_received += len(data)
            total_data += data
        data = np.frombuffer(total_data, dtype='B')
        image_dimension = int(math.sqrt(total_received/3))
        data = data.reshape((image_dimension, image_dimension, 3))
        #plt.imshow(data, interpolation='nearest')
        #plt.show()

    print("######", 1.0/(time.time() - start_time) * 0.05)
    time.sleep(sleep_factor)

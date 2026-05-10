import serial
import time
import math
import socket
import json

# UDP 配置
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PC_IP = "192.168.127.1"
PC_PORT = 9870

def send_debug_data(x,y):
    try:
        data = {
                "x":float(x),
                "y":float(y),
        }
        udp_sock.sendto(json.dumps(data).encode(), (PC_IP, PC_PORT))
    except Exception as e:
        print(f"Send debug data error: {e}")

ser=serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

t=0.0
last_send_time=time.time()
while True:
    now = time.time()
    if now - last_send_time >= 0.01:
        x = 10 * math.sin(t * 0.05) + 10  # 输出：0 ~ 20
        y = 10 * math.sin(t * 0.05) + 10  # 输出：0 ~ 20
        t += 1
        send_debug_data(x,y)
        # 组装数据包
        byte = [0x50]
        byte.append((int(x/20*0xFFFF)>>8) &0xFF)
        byte.append( int(x/20*0xFFFF)     &0xFF)
        byte.append((int(y/20*0xFFFF)>>8) &0xFF)
        byte.append( int(y/20*0xFFFF)     &0xFF)
        byte.append(0x0f)
        byte.append((byte[0]+byte[1]+byte[2]+byte[3]+byte[4]+byte[5])&0xFF)
        # 发送
        ser.write(bytes(byte))
        print(f'{byte}')
        last_send_time = now
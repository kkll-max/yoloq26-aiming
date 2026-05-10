import os
import cv2
import time
import serial
import json
import logging
import socket
import threading
import queue
import numpy as np
from collections import deque
from sort import AlphaBetaFilter
from hobot_dnn import pyeasy_dnn as dnn
from HIK_TEF_Driver_module import HIK_TEF_Driver


# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(threadName)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger('bin_detect')


# 常量定义
COLOR_NAMES = ["blue", "red", "none", "purple"]
CLASS_NAMES = ["G", "1", "2", "3", "4", "5", "O", "B"]
SIZE_NAMES  = ["s", "b"]
STRIDES     = [8, 16, 32]
TARGET_SIZE = (640, 640)
PAD_TOP     = 50
PAD_SIZE    = 640

# SORT 参数
TRACKER_MAX_AGE  = 3
MAX_VX         = 0.3     


#UDP协议
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PC_IP, PC_PORT = "192.168.127.1", 9870

def send_debug_data(raw_x1, raw_y1, raw_x_cmd, raw_y_cmd,vx,vy):
    try:
        data: dict = {
            "raw_x1":       float(raw_x1),
            "raw_y1":       float(raw_y1),
            "raw_x_cmd":   float(raw_x_cmd),
            "raw_y_cmd":   float(raw_y_cmd),
            "vx":          float(vx),
            "vy":          float(vy), 
        }
        udp_sock.sendto(json.dumps(data).encode(), (PC_IP, PC_PORT))
    except Exception as e:
        logger.warning(f"UDP send error: {e}")


# Grid 预计算
GRIDS = {}
for _s in STRIDES:
    _g = np.stack(np.indices((640 // _s, 640 // _s))[::-1], axis=-1)
    GRIDS[_s] = _g.reshape(-1, 2).astype(np.float32) + 0.5


# BGR → NV12
_nv12_buf   = None
_nv12_shape = None

def bgr2nv12(image):
    global _nv12_buf, _nv12_shape
    h, w = image.shape[:2]
    area = h * w
    yuv  = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(area * 3 // 2)
    uv   = yuv[area:].reshape(2, area // 4).T.reshape(area // 2)
    if _nv12_shape != image.shape:
        _nv12_buf   = np.empty(area * 3 // 2, dtype=np.uint8)
        _nv12_shape = image.shape
    _nv12_buf[:area] = yuv[:area]
    _nv12_buf[area:] = uv
    return _nv12_buf


# 特征图解析
def parse_outputs(outputs):
    features = {}
    for out in outputs:
        data = out.buffer
        _, H, W, C = data.shape
        stride = 640 // H
        if stride not in features:
            features[stride] = {}
        flat = data.reshape(-1, C)
        if   C == 4: features[stride]['box'] = flat
        elif C == 8: features[stride]['kpt'] = flat.reshape(-1, 4, 2)
        else:        features[stride]['cls'] = flat
    return features


# 后处理
_CONF_RAW = -np.log(1 / 0.45 - 1)

last_target_pos = None # (x, y)

def postprocess(features, conf_threshold=0.45, nms_threshold=0.45):
    global last_target_pos
    all_boxes, all_scores, all_ids, all_kpts = [], [], [], []

    for stride, feats in features.items():
        if not all(k in feats for k in ('box', 'cls', 'kpt')): continue
        cls_data = feats['cls']
        raw_sc = np.max(cls_data, axis=1)
        mask = raw_sc >= _CONF_RAW
        if not mask.any(): continue
        v_id = np.argmax(cls_data[mask], axis=1)
        v_sc = 1.0 / (1.0 + np.exp(-raw_sc[mask]))
        v_box, v_kpts = feats['box'][mask], feats['kpt'][mask]
        grid = GRIDS[stride][mask]
        xyxy = np.empty_like(v_box)
        xyxy[:, :2] = (grid - v_box[:, :2]) * stride
        xyxy[:, 2:] = (grid + v_box[:, 2:]) * stride
        decoded_kpts = (v_kpts[:, :, :2] + grid[:, None, :]) * stride
        all_boxes.append(xyxy); all_scores.append(v_sc)
        all_ids.append(v_id); all_kpts.append(decoded_kpts)

    if not all_boxes:
        last_target_pos = None
        return None

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    ids = np.concatenate(all_ids)
    kpts = np.concatenate(all_kpts)

    xywh = boxes.copy(); xywh[:, 2:] -= xywh[:, :2]
    indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf_threshold, nms_threshold)
    if not len(indices):
        last_target_pos = None
        return None

    flat = indices.flatten()
    
    best_score = -1.0
    best_i = -1

    # 权重配置
    W_AREA = 2   # 面积权重
    W_CENTER = 0.5  # 中心权重
    W_HISTORY = 0.3 # 历史粘性权重

    for i in flat:
        box = boxes[i]
        # 计算面积分
        area = (box[2] - box[0]) * (box[3] - box[1])
        area_score = area / (PAD_SIZE * PAD_SIZE) 
        
        # 计算中心距离分
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        dist_to_center = np.sqrt(((cx / PAD_SIZE) - 0.5)**2 + ((cy / PAD_SIZE) - 0.5)**2)
        center_score = 1.0 - (dist_to_center / 0.707) # 0.707 是中心到角点的最大距离

        # 筛选最优得分 
        history_score = 0
        if last_target_pos is not None:
            dist_to_last = np.sqrt((cx - last_target_pos[0])**2 + (cy - last_target_pos[1])**2)
            if dist_to_last < 50: 
                history_score = 1.0

        # 综合打分
        total_score = (area_score * W_AREA) + (center_score * W_CENTER) + (history_score * W_HISTORY)
        
        if total_score > best_score:
            best_score = total_score
            best_i = i

    # 更新历史位置
    final_box = boxes[best_i]
    last_target_pos = ((final_box[0] + final_box[2]) / 2, (final_box[1] + final_box[3]) / 2)
    kp = kpts[best_i].copy()
    polygon = np.array([[kp[0,0], kp[0,1]], [kp[3,0], kp[3,1]], [kp[2,0], kp[2,1]], [kp[1,0], kp[1,1]]], dtype=np.int32)
    det_id = int(ids[best_i])
    return {
        'box': final_box.astype(int),
        'confidence': float(scores[best_i]),
        'color_id': det_id // (len(CLASS_NAMES) * len(SIZE_NAMES)),
        'size_id': (det_id // len(CLASS_NAMES)) % len(SIZE_NAMES),
        'class_id': det_id % len(CLASS_NAMES),
        'polygon': polygon,
    }
# 串口异步发送
_serial_queue = queue.Queue(maxsize=2)

def _serial_worker(ser):
    while True:
        data = _serial_queue.get()
        if data is None:
            break
        try:
            ser.write(data)
        except Exception as e:
            logger.warning(f"Serial error: {e}")

def async_serial_write(data: bytes):
    try:
        _serial_queue.put_nowait(data)
    except queue.Full:
        pass


# 控制指令编码
def encode_cmd(sd_x, sd_y, raw_x, raw_y):
    sd_x = max(0.0, min(20.0, sd_x))
    sd_y = max(0.0, min(20.0, sd_y))
    ox   = int(sd_x / 20.0 * 0xFFFF)
    oy   = int(sd_y / 20.0 * 0xFFFF)
    fire = 0x0F if ((raw_x-0.5)**2 + (raw_y-0.5)**2)**0.5 < 0.05 else 0xF0
    b    = [0x50, (ox>>8)&0xFF, ox&0xFF, (oy>>8)&0xFF, oy&0xFF, fire]
    b.append(sum(b) & 0xFF)
    return bytes(b)


# 流水线队列
queue_cap     = queue.Queue(maxsize=1)
queue_infer   = queue.Queue(maxsize=1)
stop_event    = threading.Event()
trigger_event = threading.Event()


# 采集线程
def capture_thread(Video):
    try:
        Video.Photograph()
        while not stop_event.is_set():
            while not Video.isimagereturned():
                if stop_event.is_set():
                    while not Video.isimagereturned():
                        time.sleep(0.001)
                    Video.get_image()
                    return
                time.sleep(0.001)

            ret, frame = Video.get_image()
            ts = time.time()

            if ret != 0 or frame is None:
                logger.warning("capture: get_image failed ret=%s", ret)
                Video.Photograph()
                continue

            queue_cap.put((frame, ts))
            trigger_event.wait(timeout=1.0)
            trigger_event.clear()
            if not stop_event.is_set():
                Video.Photograph()

        while not Video.isimagereturned():
            time.sleep(0.001)
        Video.get_image()

    except Exception as e:
        logger.error(f"capture_thread exception: {e}")
        stop_event.set()


# 推理线程
def inference_thread(models, img_w, img_h):
    try:
        while not stop_event.is_set():
            try:
                frame, ts = queue_cap.get(timeout=0.2)
            except queue.Empty:
                continue

            frame_padded = frame[220:860,400:1040 ].copy()
            nv12=bgr2nv12(frame_padded)
            outputs     = models[0].forward(nv12)
            features    = parse_outputs(outputs)

            trigger_event.set()

            try: queue_infer.get_nowait()
            except queue.Empty: pass
            queue_infer.put((features, ts))

    except Exception as e:
        logger.error(f"inference_thread exception: {e}")
        stop_event.set()


# 主函数
if __name__ == "__main__":

    models = dnn.load('models/auto_aim_modified2.bin')

    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
    t_serial = threading.Thread(target=_serial_worker, args=(ser,), daemon=True)
    t_serial.start()

    Video = HIK_TEF_Driver.HIK_Video()
    ret, img_w, img_h = Video.Init_Video()
    logger.info(f"Camera: ret={ret}  {img_w}x{img_h}")

    # 初始化滤波器 
    filter_x = AlphaBetaFilter(alpha=0.6, beta=0.1) # alpha越小越稳，beta决定预测强度
    filter_y = AlphaBetaFilter(alpha=0.6, beta=0.1)
    # 状态变量初始化
    raw=False
    lost_count = 0
    history_len = 5
    vx, vy = 0.0, 0.0
    last_ts = time.time()
    fps_buf = [time.time()] * 10
    t_cap   = threading.Thread(target=capture_thread,   args=(Video,),
                               daemon=True, name="capture")
    t_infer = threading.Thread(target=inference_thread, args=(models, img_w, img_h),
                               daemon=True, name="infer")
    t_cap.start()
    t_infer.start()

    try:
        while True:
            try:
                features, ts = queue_infer.get(timeout=0.2)
            except queue.Empty:
                continue

            fps_buf.append(time.time())
            fps = 10.0 / (fps_buf[-1] - fps_buf.pop(0))
                
            obj = postprocess(features)
            has_det = obj is not None
            
            curr_dt = ts - last_ts
            last_ts = ts
            
            # 状态更新

            if has_det:
                lost_count = 0
                det_box = obj['box'].astype(float)
                raw=True
                raw_x0 = (det_box[0] + det_box[2]) / 2 / PAD_SIZE
                raw_y0 = (det_box[1] + det_box[3]) / 2 / PAD_SIZE

                # if filter_x.x_filt is not None:
                #     dist = np.sqrt((raw_x0 - filter_x.x_filt)**2 + (raw_y0 - filter_y.x_filt)**2)
                #     if dist > 0.2:  
                #         filter_x.reset()
                #         filter_y.reset()
                #         logger.info("Target Switched - Filter Reset")
                #         vx=0
                #         vy=0
                #滤波器更新
                raw_x1, vx = filter_x.update(raw_x0, curr_dt)
                raw_y1, vy = filter_y.update(raw_y0, curr_dt)
            else:
                lost_count += 1
                # 只有当滤波器初始化过，且丢失时间不长时才预测
                if filter_x.x_filt is not None and lost_count < TRACKER_MAX_AGE:
                    # 纯预测模式
                    vx, vy = filter_x.v_filt, filter_y.v_filt
                    raw_x1 = filter_x.x_filt + vx * curr_dt
                    raw_y1 = filter_y.x_filt + vy * curr_dt
                    # 更新滤波器内部状态以便下一帧继续预测 (注意衰减速度)
                    filter_x.x_filt = raw_x1
                    filter_y.x_filt = raw_y1
                    filter_x.v_filt *= 0.9  # 丢失目标时，预测速度逐帧衰减
                    filter_y.v_filt *= 0.9
                else:
                    if raw :
                        # 彻底丢失，回归中心或保持不动
                        raw_x1=raw_x1
                        raw_y1 =raw_y1
                        vx=vx
                        vy=vy
                    else:
                        raw_x1=0.5
                        raw_y1=0.5
                        vx, vy = 0.0, 0.0

            # 速度限幅 (防止过冲)
            vx = np.clip(vx, -MAX_VX, MAX_VX)
            vy = np.clip(vy, -MAX_VX, MAX_VX)

            # 前馈控制量计算
            process_latency = time.time() - ts
            look_ahead = 0.85*process_latency 
            print(f'look_ahead:{look_ahead}') 
            
            raw_x_cmd = raw_x1 + vx * look_ahead 
            raw_y_cmd = raw_y1 + vy * look_ahead+0.03

            # 指令编码与开火控制
            sd_x = (raw_x_cmd - 0.5) * 20.0 + 10.0
            sd_y = (raw_y_cmd - 0.5) * 20.0 + 10.0
            dist_to_center = ((raw_x_cmd-0.5)**2 + (raw_y_cmd-0.5)**2)**0.5
            is_fire_ready = has_det and (dist_to_center < 0.05)
            
            # 串口通信
            cmd = encode_cmd(sd_x, sd_y, raw_x_cmd, raw_y_cmd)
            if not is_fire_ready: 
                cmd = bytearray(cmd)
                cmd[5] = 0xF0 
                cmd[6] = sum(cmd[:6]) & 0xFF # 重新计算校验和
                cmd = bytes(cmd)

            async_serial_write(cmd)
            send_debug_data(
                raw_x1, raw_y1,
                raw_x_cmd, raw_y_cmd,vx, vy,
            )
            src  = "det" if has_det else "pred"
            fire = cmd[5] == 0x0F

            tag  = (f"[{COLOR_NAMES[obj['color_id']]} "
                    f"{SIZE_NAMES[obj['size_id']]} "
                    f"{CLASS_NAMES[obj['class_id']]}] "
                    f"conf={obj['confidence']:.2f}") if has_det else "[PREDICTED]"

            logger.info(
                f"FPS={fps:.1f}  {tag}  "
                f"filt=({raw_x1:.3f},{raw_y1:.3f})  "
                f"cmd=({raw_x_cmd:.3f},{raw_y_cmd:.3f})  "
                f"vx={vx:.4f}   "
                f"src={src}  fire={'YES' if fire else 'no'}"
                )


    except KeyboardInterrupt:
        logger.info("Interrupted.")

    finally:
        stop_event.set()
        trigger_event.set()
        t_cap.join(timeout=3)
        try:
            Video.Close_Video()
        except Exception:
            pass
        t_infer.join(timeout=2)
        try:
            _serial_queue.put_nowait(None)
        except queue.Full:
            pass
        t_serial.join(timeout=2)
        try:
            ser.close()
        except Exception:
            pass
        try:
            udp_sock.close()
        except Exception:
            pass
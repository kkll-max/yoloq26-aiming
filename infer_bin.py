import os
import cv2
import time
import serial
import json
import logging
import socket
import numpy as np
import bpu_yolov5_tools as tools
from hobot_dnn import pyeasy_dnn as dnn
from HIK_TEF_Driver_module import HIK_TEF_Driver

# sigmoid函数定义
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(img, target_size=(640, 640)):
    """预处理图像"""
    # 保持长宽比进行resize
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    
    # resize并填充, 保持RGB格式
    resized = cv2.resize(img, (nw, nh), cv2.INTER_NEAREST)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    new_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    new_img[:nh, :nw] = resized 
    
    #CHW格式
    new_img = np.expand_dims(new_img, axis=0).astype(np.float32) / 255.0
    
    return new_img, scale, 0, 0

# bgr2nv12
def bgr2nv12(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12

# 颜色和类别映射（根据您的模型调整）
COLOR_NAMES = ["blue", "red", "none", "purple"]
CLASS_NAMES = ["G", "1", "2", "3", "4", "5", "O", "B",]#哨兵，12345，前哨站，基地
SIZE_NAMES = ["s", "b"]
def draw_results(img, results):
    """在图像上绘制检测结果，使用多边形而不是矩形框"""
    for obj in results:
        polygon = obj['polygon']
        class_id = obj['class_id']
        color_id = obj['color_id']
        size_id = obj['size_id']
        confidence = obj['confidence']
        
        # 选择颜色，红色或蓝色
        color = (0, 255, 0)
        
        # 绘制多边形
        cv2.polylines(img, [polygon], True, color, 1)
        
        # 计算多边形质心用于放置标签
        centroid = np.mean(polygon, axis=0).astype(int)
        
        # 绘制类别和置信度，放在多边形上方
        label = f"{COLOR_NAMES[color_id]} {SIZE_NAMES[size_id]} {CLASS_NAMES[class_id]} : {confidence:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0]
        text_x = max(0, centroid[0] - text_size[0] // 2)
        text_y = max(15, centroid[1] - 10)
        
        # 绘制文字
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        
        # 可选：在多边形顶点处绘制小圆点
        for point in polygon:
            cv2.circle(img, tuple(point), 1, color, 1)
    
    return img

STRIDES = [8, 16, 32]
grids = {}
def postprocess(features, img_shape, scale, dx, dy, conf_threshold=0.65, nms_threshold=0.45):
    """后处理函数"""    
    conf_raw = -np.log(1 / conf_threshold - 1)

    detections = []
    for stride, feats in features.items():
        if not all(k in feats for k in ['box', 'cls', 'kpt']):
            continue
            
        box_data, cls_data, kpt_data = feats['box'], feats['cls'], feats['kpt']
        
        class_ids = np.argmax(cls_data, axis=1)
        scores = np.max(cls_data, axis=1)
        
        mask = scores >= conf_raw
        if not np.any(mask):
            continue
        
        v_id = class_ids[mask]
        v_scores = 1 / (1 + np.exp(-scores[mask]))
        v_box = box_data[mask]
        v_kpts = kpt_data[mask]
        grid = grids[stride][mask]

        # bbox decoding 
        xyxy = np.hstack([(grid - v_box[:, :2]), (grid + v_box[:, 2:])]) * stride
            
        # Keypoints decoding
        decoded_kpts = (v_kpts[:, :, :2] + grid[:, None, :]) * stride

        for box, score, kpts, id in zip(xyxy, v_scores, decoded_kpts, v_id):
            detections.append({'box': box, 'score': score, 'id': id,'kpts': kpts})

    final_res = []
    if detections:
        boxes = np.array([d['box'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
            
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores.tolist(), 
            conf_threshold, nms_threshold
        )
            
        if len(indices) > 0:
            for i in indices.flatten():
                det = detections[i]
                x1 = det['box'][0] / scale
                y1 = det['box'][1] / scale
                x2 = det['box'][2] / scale
                y2 = det['box'][3] / scale
                
                kpts = det['kpts'].copy()
                kpts[:, 0] /= scale
                kpts[:, 1] /= scale

                polygon = np.array([
                    [kpts[0, 0], kpts[0, 1]],  # 左上
                    [kpts[3, 0], kpts[3, 1]],  # 右上
                    [kpts[2, 0], kpts[2, 1]],  # 右下
                    [kpts[1, 0], kpts[1, 1]]   # 左下
                ], dtype=np.int32)
                    
                final_res.append({
                    'box': np.array([x1, y1, x2, y2], dtype=int),
                    'confidence': det['score'],
                    'color_id': det['id'] // (len(CLASS_NAMES) * len(SIZE_NAMES)),
                    'size_id': (det['id'] // len(CLASS_NAMES)) % len(SIZE_NAMES),
                    'class_id': det['id'] % len(CLASS_NAMES),
                    'polygon': polygon
                })
    
    return final_res

# UDP 配置
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PC_IP = "192.168.127.1"
PC_PORT = 9870

def send_debug_data(raw_x,raw_y,smooth_x,smooth_y,aim_x,aim_y,sd_x,sd_y):
    try:
        data = {
            "raw_x": float(raw_x),
            "raw_y": float(raw_y),
            "smooth_x" :float(smooth_x),
            "smooth_y" :float(smooth_y), 
            "aim_x" :float(aim_x),
            "aim_y" :float(aim_y),
            "sd_x" : float(sd_x),
            "sd_y" : float(sd_y),
        }
        udp_sock.sendto(json.dumps(data).encode(), (PC_IP, PC_PORT))
    except Exception as e:
        print(f"Send debug data error: {e}")

#日志配置
logging.basicConfig(
    level=logging.DEBUG, 
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger('bin_detect')

if __name__ == "__main__":

    #加载模型信息
    model_path='models/auto_aim_modified2.bin'
    models=dnn.load(model_path)

    # 串口初始化
    ser = serial.Serial("/dev/ttyS1", 115200, timeout=1)

    #grids初始化
    STRIDES = [8, 16, 32]
    grids = {} 
    for s in STRIDES:
        grid_h, grid_w = 640 // s, 640 // s
        grid = np.stack(np.indices((grid_h, grid_w))[::-1], axis=-1)
        grids[s] = grid.reshape(-1, 2).astype(np.float32) + 0.5

    # 创建摄像头对象
    Video = HIK_TEF_Driver.HIK_Video()
    ret, w, h = Video.Init_Video()
    print(ret)
    print("Image's height {} width {}.".format(h,w))

    # 启动首张拍摄
    Video.Photograph()

    # 时间序列
    time_list = [time.time() for i in range(10)]

    #raw_x,raw_y初始化
    last_time=time.time()
    last_raw_x=0
    last_raw_y=0
    try:
        while True:
            time0=time.time()
            time_list.append(time.time())
            print("FPS: ", 10/(time_list[-1]-time_list.pop(0)))

            begin_time=time.time()
            # 等待拍摄结束
            while not Video.isimagereturned(): # 等待拍摄完成
                pass
            ret, image_bgr = Video.get_image() # 获取图像
            logger.debug(f"获取图像：{1000*(time.time() - begin_time):.2f} ms")

            # 继续获取图像
            Video.Photograph()
            logger.debug(f"拍摄图片：{1000*(time.time() - begin_time):.2f} ms")

            #预处理
            image_bgr = cv2.copyMakeBorder(image_bgr, 50, 50, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            input_tensor, scale, dx, dy = preprocess(image_bgr)
            nv12_data = bgr2nv12(image_bgr)
            logger.debug(f"预处理2：{1000*(time.time() - begin_time):.2f} ms")

            # 模型推理
            outputs = models[0].forward(nv12_data)
            #logger.debug(f"模型推理：{1000*(time.time() - begin_time):.2f} ms")

            features = {}
            for out in outputs:
                data = out.buffer
                batch_size, H, W, C = data.shape
                stride = 640 // H
                if stride not in features:
                    features[stride] = {}
                if C == 4:
                    features[stride]['box'] =data.reshape(-1, 4)
                elif C == 8:
                    features[stride]['kpt'] = data.reshape(-1, 4, 2)
                else:
                    features[stride]['cls'] = data.reshape(-1, C)
                
            # 后处理
            results = postprocess(features, image_bgr.shape, scale, dx, dy)
            #logger.debug(f"后处理：{1000*(time.time() - begin_time):.2f} ms")
            if len(results)>0:
                print(f'results:{results}')
                results=[results[0]]
            #跟踪
            for obj in results:
                polygon = obj['polygon']
                class_id = obj['class_id']
                color_id = obj['color_id']
                size_id = obj['size_id']
                confidence = obj['confidence']
                center_x_pixel = np.mean(polygon[:, 0])
                center_y_pixel = np.mean(polygon[:, 1])
                print(f'{class_id},{color_id},{size_id}')
                raw_x = center_x_pixel / 640 -0.01
                raw_y = center_y_pixel / 640

                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time  

                if dt > 0.001:
                    vx = (raw_x - last_raw_x) / dt
                    vy = (raw_y - last_raw_y) / dt
                else:
                    vx = 0.0
                    vy = 0.0
                last_raw_x = raw_x
                last_raw_y = raw_y
                sd_x=(raw_x-0.5)*20+10
                sd_y=(raw_y-0.5)*20+10
                #send_debug_data(raw_x,raw_y,sd_x,sd_y)

                if sd_x>20:sd_x=20
                if sd_x<0:sd_x=0
                if sd_y>20:sd_y=20
                if sd_y<0:sd_y=0
                #logger.debug(f"解算：{1000*(time.time() - begin_time):.2f} ms")
                # 转码为云台控制指令
                byte = [0x50]
                byte.append((int(sd_x/20*0xFFFF)>>8) &0xFF)
                byte.append( int(sd_x/20*0xFFFF)     &0xFF)
                # byte.append( (int(vx*0xFFFF)>>8 )&0xFF)
                # byte.append( int(vy*0xFFFF)    &0xFF)
                byte.append((int(sd_y/20*0xFFFF)>>8) &0xFF)
                byte.append( int(sd_y/20*0xFFFF)     &0xFF)
                if ((raw_x-0.5)**2 + (raw_y-0.5)**2)**0.5 < 0.1:
                    byte.append(0x0F)
                    print("fire")
                else:
                    byte.append(0xF0)
                byte.append((byte[0]+byte[1]+byte[2]+byte[3]+byte[4]+byte[5])&0xFF)

                ser.write(bytes(byte))
                logger.debug(f"数据发送：{1000*(time.time() - begin_time):.2f} ms")
                print(byte)
                #[0x50, odx高8位, odx低8位, ody高8位, ody低8位, 射击标志, 校验和]
                print(f'dt:{1000*(time.time()-time0):.2f}ms')
    
    finally:
        Video.Close_Video()






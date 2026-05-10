import os
import cv2
import time
import logging
import numpy as np
import bpu_yolov5_tools as tools
from hobot_dnn import pyeasy_dnn as dnn

# sigmoid函数定义
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(img, target_size=(640, 640)):
    """预处理图像"""
    # 保持长宽比进行resize
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)

    if nw % 2 != 0:
        nw += 1
    if nh % 2 != 0:
        nh += 1

    resized = cv2.resize(img, (nw, nh), cv2.INTER_NEAREST)
    return resized, scale, 0, 0

# 颜色和类别映射（根据您的模型调整）
COLOR_NAMES = ["blue", "red", "none", "purple"]
CLASS_NAMES = ["G", "1", "2", "3", "4", "5", "O", "B",]
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
    """后处理函数，严格遵循C++代码逻辑"""    
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

logging.basicConfig(
    level=logging.DEBUG, 
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger('bin_detect')

if __name__ == "__main__":

    #加载模型信息
    model_path='models/auto_aim_modified2.bin'
    models=dnn.load(model_path)

    # 创建输出文件夹
    input_folder='video_img'
    output_folder='video_result'
    resized_folder='resized_img'
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #grids初始化
    STRIDES = [8, 16, 32]
    grids = {} 
    for s in STRIDES:
        grid_h, grid_w = 640 // s, 640 // s
        grid = np.stack(np.indices((grid_h, grid_w))[::-1], axis=-1)
        grids[s] = grid.reshape(-1, 2).astype(np.float32) + 0.5

    count=0
    # 处理每个图像
    for filename in os.listdir(input_folder):
        begin_time=time.time()
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            continue
        
        img_path = os.path.join(input_folder, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"无法读取图像: {img_path}")
            continue
        logger.debug(f"加载图片：{1000*(time.time() - begin_time):.2f} ms")

        #预处理
        resized, scale, dx, dy = preprocess(img_bgr)
        resized_path=os.path.join(resized_folder, filename)
        cv2.imwrite(resized_path, resized)
        nv12_data = tools.bgr2nv12(resized)
        logger.debug(f"预处理耗时：{1000*(time.time() - begin_time):.2f} ms")

        # 模型推理
        outputs = models[0].forward(nv12_data)
        logger.debug(f"模型推理：{1000*(time.time() - begin_time):.2f} ms")

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
        results = postprocess(features, img_bgr.shape, scale, dx, dy)
        logger.debug(f"后处理耗时：{1000*(time.time() - begin_time):.2f} ms")
        if len(results)>0:
            count+=1
        # 绘制结果
        result_img = draw_results(img_bgr.copy(), results)
        
        # 保存结果
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result_img)
        print(f"已处理并保存: {output_path}")
        print(f'count:{count}')

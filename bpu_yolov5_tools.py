# ==================================================
#
#     yolov5相关工具库
#       * Author: Leaf
#       * Date: 2024-01-26
#
# ==================================================
import cv2
import numpy as np

# 验证模型是否正常
def model_val(models):
    """
    正常模型示例（官方）：
    ------------------------------
    tensor type: NV12_SEPARATE
    data type: uint8
    layout: NCHW
    shape: (1, 3, 640, 640)
    ------------------------------
    tensor type: float32
    data type: float32
    layout: NHWC
    shape: (1, 80, 80, 255)
    ------------------------------
    tensor type: float32
    data type: float32
    layout: NHWC
    shape: (1, 40, 40, 255)
    ------------------------------
    tensor type: float32
    data type: float32
    layout: NHWC
    shape: (1, 20, 20, 255)
    ------------------------------
    """
    # 验证数据类型
    try:
        if models[0].inputs[0].properties.layout != 'NCHW' \
        or models[0].inputs[0].properties.shape[0] != 1 \
        or models[0].inputs[0].properties.shape[1] != 3 \
        or models[0].inputs[0].properties.shape[2] != models[0].inputs[0].properties.shape[3]:
            return -1 # 输入头错误
        
        if len(models[0].outputs) != 3:
            return -2 # 输出头数量错误

        for i in range(3):
            if models[0].outputs[i].properties.shape[0] != 1 \
            or models[0].outputs[i].properties.tensor_type != 'float32' \
            or models[0].outputs[i].properties.dtype != 'float32' \
            or models[0].outputs[i].properties.layout != 'NHWC':
                return -3 # 输出数据类型错误

        if models[0].outputs[0].properties.shape[1] != models[0].outputs[1].properties.shape[1]*2 \
        or models[0].outputs[1].properties.shape[1] != models[0].outputs[2].properties.shape[1]*2 \
        or models[0].outputs[0].properties.shape[1]*8 != models[0].inputs[0].properties.shape[2]:
            return -4 # 输出数据尺寸错误

        if models[0].outputs[0].properties.shape[3] != models[0].outputs[1].properties.shape[3] \
        or models[0].outputs[1].properties.shape[3] != models[0].outputs[2].properties.shape[3]:
            return -5 # 输出shape[3]不一致
        
    except:
        return -6 # 其他错误
    
    return 0

# 显示模型信息
def models_info_show(models):
    print('='*50)
    print('INPUT')
    print('-'*50)
    print("tensor type:", models[0].inputs[0].properties.tensor_type)
    print("data type:", models[0].inputs[0].properties.dtype)
    print("layout:", models[0].inputs[0].properties.layout)
    print("shape:", models[0].inputs[0].properties.shape)

    index = 0
    for output in models[0].outputs:
        print('='*50)
        print('OUTPUT',index)
        print('-'*50)
        index += 1
        print("tensor type:", output.properties.tensor_type)
        print("data type:", output.properties.dtype)
        print("layout:", output.properties.layout)
        print("shape:", output.properties.shape)

    print('='*50)

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

# 显示结果
def result_show(data,names):
    # 打印表头
    print('='*75)
    print('result')
    print('-'*75)
    print('{:<9}{:<9}{:<9}{:<9}{:<9}{:<9}{:<9}{:<9}'.format('ser','xmin','ymin','xmax','ymax','id','name','score'))
    # 遍历结果
    n = 0 # 序号
    for result in data:  
        n += 1
        bbox = result[:4] # 矩形框位置信息  
        score = result[5] # 置信度得分
        id_number = int(result[4]) # id  
        name = names[id_number] # 名称

        # 处理过长的name分行
        if len(name) > 7:
            _name = name[0:7]
            name = name[7:]
            ret = 1
        else:
            ret = 0
            _name = name
            name = ''

        print('{:<9d}{:<9.2f}{:<9.2f}{:<9.2f}{:<9.2f}{:<9d}{:<9}{:<9.2f}'.format(n, bbox[0], bbox[1], bbox[2], bbox[3], id_number, _name, score))

        while ret: # 分行处理
            print(' '*6*9,end='')
            if len(name) > 7:
                print('{:<9}'.format(name[:7]))
                name = name[7:]
            else:
                print(name)
                break

    print('='*75)

# 获取颜色
def get_color(id, class_number):
    code = int(id / class_number * 256 ** 3)
    b = code % 256
    g = (code // 256) % 256
    r = (code // 256 // 256)

    return (b, 255-g, r)

# 绘制可视化结果
def result_draw(_img,
                data, 
                names):

    img = _img
    # 遍历结果
    for result in data:
        # 获取信息
        bbox = result[:4] # 矩形框位置信息  
        score = result[5] # 置信度得分
        id_number = int(result[4]) # id  
        name = names[id_number] # 名称

        # 绘制矩形框
        cv2.rectangle(img, 
                      (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[2]), int(bbox[3])), 
                      get_color(id_number,len(names)), 5)

        # 获取文字大小
        (text_width, text_height), baseline = cv2.getTextSize(name + ' {:.2f}'.format(score), 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                                            1.2, 
                                                            3)

        # 计算文字背景的矩形坐标
        start_point = (int(bbox[0]), int(bbox[1]) - text_height - baseline)
        end_point = (int(bbox[0]) + text_width, int(bbox[1]))

        # 绘制矩形填充背景
        cv2.rectangle(img, start_point, end_point, get_color(id_number,len(names)), thickness=cv2.FILLED)

        # 绘制文字
        cv2.putText(img, name + ' {:.2f}'.format(score), (int(bbox[0]), int(bbox[1]) - (baseline>>1)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    return img

    
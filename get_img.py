import argparse
import os
import time
import cv2
from HIK_TEF_Driver_module import HIK_TEF_Driver


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--count',  type=int,   default=100,           help='采集帧数')
    p.add_argument('--out',    type=str,   default='video_img',    help='保存目录')
    p.add_argument('--prefix', type=str,   default='frame',        help='文件名前缀')
    p.add_argument('--ext',    type=str,   default='jpg',          help='图片格式 jpg/png')
    p.add_argument('--quality',type=int,   default=95,             help='jpg 质量 1-100')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    Video = HIK_TEF_Driver.HIK_Video()
    ret, img_w, img_h = Video.Init_Video()
    if ret != 0:
        print(f"[ERROR] 相机初始化失败 ret={ret}")
        return
    else:
        print(f"[INFO] 相机初始化成功  {img_w}x{img_h}")

    #  编码参数 
    if args.ext.lower() == 'jpg':
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    elif args.ext.lower() == 'png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  
    else:
        encode_params = []

    saved   = 0
    skipped = 0
    t_start = time.time()

    print(f"[INFO] 开始采集 {args.count} 帧 → {args.out}/")

    try:
        Video.Photograph()

        while saved < args.count:
            # 等待当前帧就绪
            timeout = time.time() + 2.0
            while not Video.isimagereturned():
                if time.time() > timeout:
                    print(f"[WARN] 等待帧超时，跳过 (saved={saved})")
                    skipped += 1
                    break
                time.sleep(0.001)
            else:
                ret, frame = Video.get_image()
                if ret != 0 or frame is None:
                    print(f"[WARN] get_image 失败 ret={ret}，跳过")
                    skipped += 1
                else:
                    fname = os.path.join(
                        args.out,
                        f"{args.prefix}_{saved:04d}.{args.ext}"
                    )
                    cv2.imwrite(fname, frame, encode_params)
                    saved += 1
                    if saved % 10 == 0 or saved == args.count:
                        elapsed = time.time() - t_start
                        fps     = saved / elapsed if elapsed > 0 else 0
                        print(f"[INFO] {saved}/{args.count}  FPS={fps:.1f}  跳过={skipped}")

            # 触发下一帧
            Video.Photograph()

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")

    finally:
        elapsed = time.time() - t_start
        print(f"\n[INFO] 完成  已保存={saved}  跳过={skipped}  耗时={elapsed:.1f}s")
        print(f"[INFO] 保存路径: {args.out}/")
        try:
            Video.Close_Video()
        except Exception as e:
            print(f"[WARN] 关闭相机异常: {e}")


if __name__ == "__main__":
    main()

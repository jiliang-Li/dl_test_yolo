# python3 
# -*- coding: utf-8 -*-
# @Time    : 2023/43/7 14:14
# @Author  : Li·Jiliang
# @File    : tsvPrediction.py
# @Software: PyCharm
from io import BytesIO
import base64
import cv2
import numpy as np
import requests
from PIL import Image
from numpy import ndarray
from concurrent.futures import ProcessPoolExecutor


# ------------------------------ opencv 和 base64 ------------------------------ #
def cv2_base64(image, **kwargs):
    base64_str = cv2.imencode('.jpg', image)[1].tobytes()
    base64_str = base64.b64encode(base64_str)
    return base64_str


def base64_cv2(base64_str, **kwargs):
    imgString = base64.b64decode(base64_str)
    nparr = np.frombuffer(imgString, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


# ------------------------------ opencv 和 bytes ------------------------------ #
def cv2bytes(img: ndarray, **kwargs):
    imgbytes = cv2.imencode(".jpg", img)[1]  # 编码为jpg格式
    imgbytes = imgbytes.tobytes()  # 转为bytes
    return imgbytes


def bytes2cv2(imgbytes, **kwargs):
    imgbytes = np.asarray(bytearray(imgbytes), dtype="uint8")  # 转为numpy数组
    img = cv2.imdecode(imgbytes, cv2.IMREAD_COLOR)  # 解码为opencv图像
    return img


def cv2pil2bytes2(img: ndarray, **kwargs):
    """
    opencv转为pil再转为bytes
        1、将BGR的opencv图像转为RGB的opencv图像
        2、将RGB的opencv图像转为PIL图像
        3、将PIL图像转为bytes
    Args:
        img:
    Returns:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    # print(pil_img)
    imgByteArr = BytesIO()
    # "JPEG"压缩会导致分类不准确
    # pil_img.save(imgByteArr, format="PNG")
    pil_img.save(imgByteArr, format="JPEG", quality=90)
    # imgByteArr.seek(0)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


# ------------------------------ PIL 和 bytes ------------------------------ #
def image2byte(image: Image, **kwargs):
    """
    PIL RGB Image
    :param image:
    :return:
    """
    imgByteArr = BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def bytes2image(imgbytes, **kwargs):
    img = Image.open(BytesIO(imgbytes))
    return img


def client(name="", nums=1):
    frame = cv2.imread('imgs/000001.jpg')
    frame = cv2.resize(frame, (1280, 640))
    ip = "192.168.6.49"
    port = 6650
    model_name = "yolo5"
    url = f"http://{ip}:{port}/predictions/{model_name}"

    jpg_data = cv2bytes(frame, quality=99)
    params = {
        "codec": "cv2bytes",
        "size": "[640,640]",
    }

    for i in range(nums):
        r = requests.post(
            url,
            params=params,
            data=jpg_data
        )
        print(r.status_code, r.text)
    return 1

def main(test_num=1, test_workers=4):
    """
    多进程
    :param test_num:
    :param test_workers:
    :return:
    """
    with ProcessPoolExecutor(max_workers=test_workers) as executor:
        futures = []
        for i in range(test_workers):
            futures.append(executor.submit(client, name=f"client-{i}", nums=test_num))
        for future in futures:
            future.result()


if __name__ == '__main__':
    # from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter as ForMat
    # parser = ArgumentParser(formatter_class=ForMat)
    # parser.add_argument("-n", "--nums", default=1, type=int, help="test iter nums")
    # parser.add_argument("-ws", "--workers", default=1, type=int, help="test workers")
    # args = parser.parse_args()
    # main(args.nums, args.workers)
    client('test1', 1)
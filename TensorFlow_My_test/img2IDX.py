# encoding: utf-8

import numpy as np
import struct
import ctypes

from PIL import Image
import binascii
import matplotlib.pyplot as plt

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data
#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()

if __name__=="__main__":
    filename = "./figure.png"
    data = ImageToMatrix(filename)
    #创建IDX文件
    offset = 0
    fmt_header = '>iiii'
    img_size = 28 * 28
    fmt_img = '>' + str(img_size) + 'B'
    # print struct.calcsize(fmt_header)
    buf = ctypes.create_string_buffer(struct.calcsize(fmt_header)+struct.calcsize(fmt_img))

    #写入描述训练集的第一行
    struct.pack_into(fmt_header,buf , offset ,2051 , 1 , 28 , 28)
    #偏移量按字节算
    offset = offset+struct.calcsize(fmt_header)
    print offset
    #然后写入一张图片
    data = data.reshape(1,img_size)
    print data
    #将线性矩阵转化为数组
    arr = data.getA1()

    struct.pack_into(fmt_img, buf, offset , *arr)
    # print binascii.hexlify(buf)
    with open("./my-idx3-ubyte","wb") as f:
        f.write(buf)
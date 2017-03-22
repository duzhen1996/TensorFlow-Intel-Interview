# encoding: utf-8
import gzip

if __name__=="__main__":
    # inputString = raw_input("请输入要判别的文件:")
    # print inputString
    with open('./figure.png', 'rb') as fRead:
        with gzip.open('./figure.gz', 'wb') as f:
            f.writelines(fRead)


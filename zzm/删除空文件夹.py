import os
from PIL import Image
import shutil

path1 = r'E:\武大\计算机\deep_learning\light-weight-refinenet-master\train_file1\labels'
labellist = os.listdir(path1)
path3=r'E:\武大\计算机\deep_learning\light-weight-refinenet-master\新建文件夹'
path3=os.path.abspath(path3)
i = 0

for item in labellist:
    sonpath=os.path.join(path1,item)
    lst=os.listdir(sonpath)
    if lst==[]:
        shutil.move(sonpath,path3)
        print("No i ".format(i) + sonpath)
        i = i + 1



print("total ", i)








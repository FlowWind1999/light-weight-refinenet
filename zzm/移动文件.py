import os
import shutil

path1 = r'E:\武大\计算机\deep_learning\light-weight-refinenet-master\train_file1\labels'
path2 = r'E:\武大\计算机\deep_learning\light-weight-refinenet-master\train_file1\images'
labellist = os.listdir(path1)
imagelist = os.listdir(path2)
path3=r'E:\武大\计算机\deep_learning\light-weight-refinenet-master\新建文件夹'
path3=os.path.abspath(path3)
i=0

for item in imagelist:
    n_item=item[0:-7]+"label.png"
    if n_item not in labellist:
        imagepath=os.path.join(path2,item)
        #image=Image.open(imagepath)
        shutil.move(imagepath,path3)
        print("No {} is ".format(i+1)+n_item)
        i=i+1

print("total ",i)








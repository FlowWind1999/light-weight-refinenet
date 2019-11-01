import os
import shutil

path1 = r'E:\武大\计算机\deep_learning\light-weight-refinenet-master\train_file1\labels'
path2 = r'E:\武大\计算机\deep_learning\light-weight-refinenet-master\train_file1\images'

path3 = r'E:\武大\计算机\deep_learning\light-weight-refinenet-master\train_file1\label'
#path3 = os.path.abspath(path3)
labellist = os.listdir(path3)
i = 0

for item in labellist:
    itempath=os.path.join(path3,item)

    image=os.path.join(itempath,item+"_img.png")
    label=os.path.join(itempath,item+"_label.png")

    shutil.move(image, path2)
    shutil.move(label, path1)

    print("No i ".format(i),image)

    i=i+1

print("total is ", i)








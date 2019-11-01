# _*_ coding:utf-8 _*_
import os
import os.path

ImageFilePath = r'../train_file1/first_label'
def getFilesAbsolutelyPath(ImageFilePath):
    currentfiles = os.listdir(ImageFilePath)
    filesVector = []
    for file_name in currentfiles:
        fullPath = os.path.join(ImageFilePath, file_name)
        filesVector.append(fullPath)
    return filesVector

filePathVector = getFilesAbsolutelyPath(ImageFilePath)
pngFile = []

for filename in filePathVector:
    pngFile.append(filename)

pngquantPath = 'labelme_json_to_dataset '

for filename in pngFile:
    os.system(pngquantPath + filename)
    print(filename)
print('game over!!!')


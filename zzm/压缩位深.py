import os
import os.path
from PIL import Image

ImageFilePath = r'../datasets/nyud/新建文件夹'
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
    im = Image.open(filename)
    if im.mode != "P":
        pngFile.append(filename)
        print(im.mode)

pngquantPath = r'.\pngquant\pngquant.exe -f --ext .png --quality 80-100  '

for filename in pngFile:
    os.system(pngquantPath + filename)
    print(filename)
print('game over!!!')


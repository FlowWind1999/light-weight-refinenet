import os

'''
批量重命名文件夹中的图片文件
'''

class BatchRename():
    def __init__(self):
        self.path = 'E:/武大/计算机/deep_learning/light-weight-refinenet-master/datasets/nyud/train_images'
    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i1 = '00000'
        i2 = '0000'
        i3 = '000'
        i4 = '00'
        i=0
        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                if (i<10):
                    dst = os.path.join(os.path.abspath(self.path), str(i1)+str(i) + '.png')
                elif (i<100):
                    dst = os.path.join(os.path.abspath(self.path), str(i2) +str(i) + '.png')
                elif (i < 1000):
                    dst = os.path.join(os.path.abspath(self.path), str(i3) + str(i) +'.png')
                elif (i < 10000):
                    dst = os.path.join(os.path.abspath(self.path), str(i4) + str(i) +'.png')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d pngs' % (total_num, i))
if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()

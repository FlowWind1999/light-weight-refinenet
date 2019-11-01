import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

def getid(i):
    i=i-1122
    if i<10:
        return './change_png/00000{}.png'.format(i)
    elif i<100:
        return './change_png/0000{}.png'.format(i)
    elif i<1000:
        return './change_png/000{}.png'.format(i)
    else:
        return './change_png/00{}.png'.format(i)


def change(image):
    shape = image.shape
    a = shape[1]
    b = shape[2]
    for i in range(a):
        for j in range(b):
            if (image[0][i][j] >= 0.4 and image[1][i][j]):
                image[0][i][j] = 128 / 255
                image[1][i][j] = 0
                image[2][i][j] = 0
            else:
                image[0][i][j] = 0
                image[1][i][j] = 0
                image[2][i][j] = 0
    image = transforms.ToPILImage()(image)
    return image


dataset = ImageFolder('datasets/nyud/',transform=transforms.ToTensor())
print(dataset)
for i in range(1122,1122+1122):
    name = getid(i)
    print(name)
    change(dataset[i][0]).save(name)

print("game over!!!")



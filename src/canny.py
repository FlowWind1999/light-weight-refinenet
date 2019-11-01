import cv2
import numpy as np

img=cv2.imread('../000000.png')
v1=cv2.Canny(img,80,150)
#v2=cv2.Canny(img,50,100)
#res=np.hstack((v1,v2))
cv2.imwrite(r'../result/edge/000000.png', v1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
cv2.imshow('v1',v1)
cv2.waitKey(0)
cv2.destroyAllWindows()
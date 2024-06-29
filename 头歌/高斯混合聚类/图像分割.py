from PIL import Image
import numpy as np
from sklearn.mixture import GaussianMixture

#******** Begin *********#

#读取图片，转为ndarray，变形为(-1,3)
im=Image.open('./step3/image/test.jpg')
img=np.array(im)
img_reshape=img.reshape(-1,3)  #RGB三通道即为3个特征，每个像素代表一个样本

gmm=GaussianMixture(3)
pred=gmm.fit_predict(img_reshape)

pred=pred.reshape(img.shape[0],img.shape[1])  #将预测结果变形为原图片的形状
res=np.zeros_like(img)   #(h,w,3)

#找到对应的像素点，根据聚类的结果将该像素点的RGB三通道值赋值为相应的颜色
res[pred==0,:]=[255,255,0]  #黄色
res[pred==1,:]=[0,0,255]    #蓝色
res[pred==2,:]=[0,255,0]    #绿色

#由于 img 为 ndarray,将其转成 Image 类型才能使用 save 函数实现保存图片的功能
im=Image.fromarray(res.astype('uint8'))
im.save('./step3/dump/result.jpg')
#********* End *********#
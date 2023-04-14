from unittest.mock import patch
import numpy as np
import kornia
import torch
import random
import torch.nn as nn
from utils import transformmasks

def colorJitter(colorJitter, img_mean, data = None, target = None, s=0.25):
    # s is the strength of colorjitter
    #colorJitter
    if not (data is None):
        if data.shape[1]==3:
            if colorJitter > 0.2:
                img_mean, _ = torch.broadcast_tensors(img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3), data)
                seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s,contrast=s,saturation=s,hue=s))
                data = (data+img_mean)/255
                data = seq(data)
                data = (data*255-img_mean).float()
    return data, target

def gaussian_blur(blur, data = None, target = None):
    if not (data is None):
        if data.shape[1]==3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15,1.15)
                kernel_size_y = int(np.floor(np.ceil(0.1 * data.shape[2]) - 0.5 + np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(np.floor(np.ceil(0.1 * data.shape[3]) - 0.5 + np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target

def flip(flip, data = None, target = None):
    #Flip
    if flip == 1:
        if not (data is None): data = torch.flip(data,(3,))#np.array([np.fliplr(data[i]).copy() for i in range(np.shape(data)[0])])
        if not (target is None):
            target = torch.flip(target,(2,))#np.array([np.fliplr(target[i]).copy() for i in range(np.shape(target)[0])])
    return data, target

def cowMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask, data = torch.broadcast_tensors(mask, data)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        data = (stackedMask*torch.cat((data[::2],data[::2]))+(1-stackedMask)*torch.cat((data[1::2],data[1::2]))).float()
    if not (target is None):
        stackedMask, target = torch.broadcast_tensors(mask, target)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        target = (stackedMask*torch.cat((target[::2],target[::2]))+(1-stackedMask)*torch.cat((target[1::2],target[1::2]))).float()
    return data, target

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target

def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target


def normalize(MEAN, STD, data = None, target = None):
    #Normalize
    if not (data is None):
        if data.shape[1]==3:
            STD = torch.Tensor(STD).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            MEAN = torch.Tensor(MEAN).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            STD, data = torch.broadcast_tensors(STD, data)
            MEAN, data = torch.broadcast_tensors(MEAN, data)
            data = ((data-MEAN)/STD).float()
    return data, target
################################# mymix

    #
    


#################### classmix
def classmix_mask(pred):
    # get mask
    classes = torch.unique(pred)
    classes = classes[classes!= 255] # 250 for city 淡黄色轮廓
    nclasses = classes.shape[0]
    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda() ## .long() long or float tensor
    mask = transformmasks.generate_class_mask(pred, classes).unsqueeze(0).cuda()
    return mask

def classmix_mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])# mix with next image
            ## data[(i + 1) % data.shape[0]]
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                            torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target
####################

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]: #[4,512,512] and [4,3,512,512]
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])]) # mix with next image
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target


def ricap_mix(  w_list, h_list, data = None, label1=None, label2=None ):
    #
    I_x, I_y = data.size()[2:]
    batch = data.shape[0]
    mixed_img = []
    l1_mixed_logit = []
    l2_mixed_logit = []
    for i in range(batch): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = w_list[i]
        h_ = h_list[i]
        cropped_images = {} # dict
        l1_logit = {}
        l2_logit = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
            x_k = np.random.randint(0, I_x-w_[k]+1) # 满足裁剪宽度的随机坐标
            y_k = np.random.randint(0, I_y-h_[k]+1)
            
            cropped_images[k] = data[ start, :, x_k:x_k+w_[k], y_k:y_k+h_[k]] # 生成4个crop
            l1_logit[k] = label1[ start, :, x_k:x_k+w_[k], y_k:y_k+h_[k]]
            l2_logit[k] = label2[ start, :, x_k:x_k+w_[k], y_k:y_k+h_[k]]
            start+=1
            start%=4 # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                               ( torch.cat((cropped_images[0],cropped_images[1]),1),
                                torch.cat((cropped_images[2],cropped_images[3]),1) ), 2 ) )  
        l1_mixed_logit.append( torch.cat( 
                               ( torch.cat((l1_logit[0],l1_logit[1]),1),
                                torch.cat((l1_logit[2],l1_logit[3]),1) ), 2 )) 
        l2_mixed_logit.append( torch.cat( 
                               ( torch.cat((l2_logit[0],l2_logit[1]),1),
                                torch.cat((l2_logit[2],l2_logit[3]),1) ), 2 )) 
    
    mixed_img = torch.stack(mixed_img)
    l1_mixed_logit = torch.stack(l1_mixed_logit)
    l2_mixed_logit = torch.stack(l2_mixed_logit)
    
    return mixed_img, l1_mixed_logit, l2_mixed_logit # 默认在cuda中
def mosaic_mix( data=None, label1=None, label2=None ):
    b, c, h, w = data.shape
    im_size = h
    mosaic_border = [-im_size // 2, -im_size // 2]
    s_mosaic = im_size * 2
    mixed_img = []
    l1_mixed_logit = []
    l2_mixed_logit = []
    for i in range(b): # batch size
        yc, xc = (int(np.random.uniform(-x, s_mosaic + x)) for x in mosaic_border)
        start = i # 记录起始位置
        for k in range(4): # 混合数量
            if k==0:   # top left
                large_img = torch.full((3,s_mosaic,s_mosaic),0).float().cuda() # base image with 4 tiles  torch.full can fill 114
                logit1 = torch.full(( 21, s_mosaic, s_mosaic), 0).float().cuda() # 默认为cpu 需要指定cuda
                logit2 = torch.full(( 21, s_mosaic, s_mosaic), 0).float().cuda()
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif k == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s_mosaic), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s_mosaic , yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif k == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s_mosaic ), min(s_mosaic , yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)    
                
            large_img[ :,y1a:y2a, x1a:x2a] = data[start,:,y1b:y2b, x1b:x2b]    # img4[ymin:ymax, xmin:xmax] 使用start 以防 batchsize 大小 修改
            logit1[ :,y1a:y2a, x1a:x2a] = label1[start,:,y1b:y2b, x1b:x2b]  
            logit2[ :,y1a:y2a, x1a:x2a] = label2[start,:,y1b:y2b, x1b:x2b]  
            start +=1
            start %=b
        mixed_img.append(large_img)
        l1_mixed_logit.append(logit1)
        l2_mixed_logit.append(logit2)
    unsup_imgs_mixed = nn.functional.interpolate(torch.stack(mixed_img,), size=(h,w), mode='nearest', )      # if bilinear  像素的预测值会变化 可能会产生有害信息 align_corner          
    logit_cons_tea_1 = nn.functional.interpolate(torch.stack(l1_mixed_logit,), size=(h,w), mode='nearest', )                
    logit_cons_tea_2 = nn.functional.interpolate(torch.stack(l2_mixed_logit,), size=(h,w), mode='nearest', )  
        
    return unsup_imgs_mixed, logit_cons_tea_1, logit_cons_tea_2      


def my_mosaic_mix( data=None, label1=None, label2=None ):
    b, c, h, w = data.shape
    im_size = h
    mosaic_border = [-im_size // 2, -im_size // 2]
    s_mosaic = im_size * 2
    mixed_img = []
    l1_mixed_logit = []
    l2_mixed_logit = []
    for i in range(b): # batch size
        yc, xc = (int(np.random.uniform(-x, s_mosaic + x)) for x in mosaic_border)
        start = i # 记录起始位置
        for k in range(4): # 混合数量
            if k==0:   # top left
                large_img = torch.full((3,s_mosaic,s_mosaic),0).float().cuda() # base image with 4 tiles  torch.full can fill 114
                logit1 = torch.full(( 21, s_mosaic, s_mosaic), 0).float().cuda() # 默认为cpu 需要指定cuda
                logit2 = torch.full(( 21, s_mosaic, s_mosaic), 0).float().cuda()
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif k == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s_mosaic), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s_mosaic , yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif k == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s_mosaic ), min(s_mosaic , yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)    
                
            large_img[ :,y1a:y2a, x1a:x2a] = data[start,:,y1b:y2b, x1b:x2b]    # img4[ymin:ymax, xmin:xmax] 使用start 以防 batchsize 大小 修改
            logit1[ :,y1a:y2a, x1a:x2a] = label1[start,:,y1b:y2b, x1b:x2b]  
            logit2[ :,y1a:y2a, x1a:x2a] = label2[start,:,y1b:y2b, x1b:x2b]  
            start +=1
            start %=b
        mixed_img.append(large_img)
        l1_mixed_logit.append(logit1)
        l2_mixed_logit.append(logit2)
    unsup_imgs_mixed = nn.functional.interpolate(torch.stack(mixed_img,), size=(h,w), mode='bilinear', align_corners=False)                
    logit_cons_tea_1 = nn.functional.interpolate(torch.stack(l1_mixed_logit,), size=(h,w), mode='bilinear', align_corners=False)                
    logit_cons_tea_2 = nn.functional.interpolate(torch.stack(l2_mixed_logit,), size=(h,w), mode='bilinear', align_corners=False)  
        
    return unsup_imgs_mixed, logit_cons_tea_1, logit_cons_tea_2      

def my_mix(data=None, label1=None, label2=None):
    pass

    #return unsup_imgs_mixed, logit_cons_tea_1, logit_cons_tea_2
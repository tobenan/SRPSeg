from unittest.mock import patch
import numpy as np
#import kornia
import torch
import random
import torch.nn as nn
from utils import transformmasks
import cv2
from utils import visualize
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
def classmix_mask(pred):
    # get mask
    classes = torch.unique(pred)
    classes = classes[classes!= 255] # 250 for city
    nclasses = classes.shape[0]
    ignore_classes_num = np.random.randint(nclasses) # 1类为0个 2类为0-1个 3类为0-2个
    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda() ## .long() long or float tensor
    
    ignore_classes_pos_list = torch.randperm(nclasses) # 打乱类别索引顺序
    ignore_classes_idx = classes[ ignore_classes_pos_list[:ignore_classes_num] ] # 选num个索引 取对应类idx
    ignore_classes_idx = torch.cat((ignore_classes_idx,torch.tensor([0]).cuda())) # 把背景类加回来
    mask = transformmasks.generate_class_mask(pred, classes).unsqueeze(0).cuda()
    return mask



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

def my_mix(data, softmax_pred, ps_label_1, ps_label_2, hard_id=None):
    num, c, I_y, I_x = data.shape   
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(0.3, 0.3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(0.3, 0.3)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
            
    if hard_id is not None:# 没有限制num
        # if k==0:
        cat_x = int(np.round(I_x * np.random.beta(3, 3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(3, 3)))# 19
        
        aug_pos1=(hard_id+1)%num
        aug_pos2=(hard_id+2)%num
        aug_pos3=(hard_id+3)%num
        
        cat_position_x[hard_id] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ] # 修改cat分布
        cat_position_y[hard_id] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos1] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]# same
        cat_position_y[aug_pos1] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos2] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# rotate
        cat_position_y[aug_pos2] = [ I_y-cat_y, I_y-cat_y, cat_y, cat_y ]
        
        cat_position_x[aug_pos3] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# mirror
        cat_position_y[aug_pos3] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]    
        
    '''
    # hard sample detect
    
    if engine.distributed and (engine.local_rank == 0):
        #test= torch.zeros((4,3,512,512)).cuda()
        #tb_logger.add_graph(model, test)
        #tb_logger.add_image("img",entropy)
        tb_logger.add_histogram( 'batch_entropy', entropy, epoch * config.niters_per_epoch + idx) 
        #for i in range(num):
        #    tb_logger.add_histogram( 'entropy'+ str(i), entropy[i], epoch * config.niters_per_epoch + idx) 
        imgnum = 4*(epoch * config.niters_per_epoch + idx)    
        for g in range(num):
            tb_logger.add_histogram( 'entropy', entropy[g], imgnum ) 
            imgnum+=1
        tb_logger.flush()
    '''
    
    max_probs, argmax_pred = torch.max(softmax_pred, dim=1) #  each pixel max probs value\class int64 
    # pad_list= (128,128,128,128)
    # pad_img=nn.ConstantPad2d(img) 
    #### 生成 cate list 
    
    #beta = 0.3
                                        
    mixed_img = []
    l1_mixed_label = []
    l2_mixed_label = []
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        l1_label = {}
        l2_label = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                
            '''if i==hard_id : # 如果是后续循环且要切的图像是hard图像 就直接赋值hard_img并跳过后续 如果是 5 6 7 0 则 不好赋值
                cropped_images[k] = hard_img
                l1_label[k] = hard_label1
                l2_label[k] = hard_label2
                continue
            '''
            img = data[start]   # 取起始图片
            pred = argmax_pred[start] # int64 [512,512]
            label1 = ps_label_1[start]
            label2 = ps_label_2[start]
            
            w=w_[k] # 310
            h=h_[k] # 0
            # pad
            pad_w_list = [int(w/2),w-int(w/2)] # 避免 w为奇数 除不尽的情况 如果为1 或0怎么处理 为0 后续则不取该图像
            pad_h_list = [int(h/2),h-int(h/2)]
            padding = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) # pad with 0
            pad_img = padding(img,) # pad img
            # padding_label = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) 
            ## 不需要pad with 255，这里黑边就是背景 pred是经过通道argmax出来的 label1 label2是预测出来的，
            pad_label1 = padding(label1) # pad label
            pad_label2 = padding(label2)
            # class guide
            #classes = torch.unique(pred) # 众数
            #classes = classes[classes!= 255] # 250 for city 淡黄色轮廓   按通道算argmax 那就不可能有255
            #if classes.shape[0]>2:
            #    continue
            mask = pred==0 # false 为前景 true为背景
            #fg_pixle_num= torch.sum(~mask) # 
            #fg_ratio = fg_pixle_num / ( unsup_imgs.shape[2]*unsup_imgs.shape[3]) # 0.99
            #if ratio>0.7 or ratio<0.1 如果没有前景 如果只是小块区域 怎么处理
            #    continue
            #imask = pred==0
            #if not imask.equal(mask): pass
            #print(mask.type)
            row, _ =torch.min(mask,axis=1) # 修改 _, row
            col, _ =torch.min(mask,axis=0)
            row = row.int()
            col = col.int()
            mn_row=torch.argmin(row)
            mx_row=row.shape[0]-torch.argmin(row.flip(0))-1 # torch 不支持反向索引 [::-1]
            mn_col=torch.argmin(col)
            mx_col=col.shape[0]-torch.argmin(col.flip(0))-1
            
            center_y = torch.randint( int(h/2) + mn_row, int(h/2) + mx_row + 1,[] ) # get center point 均匀分布 还是正态分布
            center_x = torch.randint( int(w/2) + mn_col, int(w/2) + mx_col + 1, [] ) 
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2)
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
            
            if hard_id is not None and i == hard_id and start == hard_id: # 在hard_id个混合图像 的开始图像为困难图像时 储存 hard_area
                hard_img = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label1 = pad_label1[ new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label2 = pad_label2[ new_min_row:new_max_row, new_min_col:new_max_col]  
                                                                                                            
            cropped_images[k] = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            l1_label[k] = pad_label1[ new_min_row:new_max_row, new_min_col:new_max_col]
            l2_label[k] = pad_label2[ new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        l1_mixed_label.append( torch.cat( 
                            ( torch.cat((l1_label[0],l1_label[1]),1),
                                torch.cat((l1_label[2],l1_label[3]),1) ), 0 )) 
        l2_mixed_label.append( torch.cat( 
                            ( torch.cat((l2_label[0],l2_label[1]),1),
                                torch.cat((l2_label[2],l2_label[3]),1) ), 0 )) 
        
    if hard_id is not None:
        mixed_img[ aug_pos1 ][ :, 0:cat_y, 0:cat_x ] = hard_img
        l1_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label1
        l2_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label2
        
        mixed_img[ aug_pos2 ][ :, I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label2
        
        mixed_img[ aug_pos3 ][ :, 0:cat_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label2
                                    
    unsup_imgs_mixed = torch.stack(mixed_img)
    ps_label_1 = torch.stack(l1_mixed_label)
    ps_label_2 = torch.stack(l2_mixed_label)
    if hard_id is not None:
        return unsup_imgs_mixed, ps_label_1, ps_label_2 ,aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y
    return unsup_imgs_mixed, ps_label_1, ps_label_2

def my_mix_weight(data, softmax_pred, ps_label_1, ps_label_2, ps_weight_1, ps_weight_2, beta=0.3, hard_id=None):
    num, c, I_y, I_x = data.shape   
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(beta, beta)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(beta, beta)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
    
    max_probs, argmax_pred = torch.max(softmax_pred, dim=1) #  each pixel max probs value\class int64 
    # pad_list= (128,128,128,128)
    # pad_img=nn.ConstantPad2d(img) 
    #### 生成 cate list 
    #beta = 0.3
    mixed_img = []
    l1_mixed_label = []
    l2_mixed_label = []
    l1_mixed_weight = []
    l2_mixed_weight = []
    
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        l1_label = {}
        l2_label = {}
        l1_weight = {}
        l2_weight = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                
            '''if i==hard_id : # 如果是后续循环且要切的图像是hard图像 就直接赋值hard_img并跳过后续 如果是 5 6 7 0 则 不好赋值
                cropped_images[k] = hard_img
                l1_label[k] = hard_label1
                l2_label[k] = hard_label2
                continue
            '''
            img = data[start]   # 取起始图片
            pred = argmax_pred[start] # int64 [512,512]
            label1 = ps_label_1[start]
            label2 = ps_label_2[start]
            weight1 = ps_weight_1[start]
            weight2 = ps_weight_2[start]
            
            w=w_[k] # 310
            h=h_[k] # 0
            # pad
            pad_w_list = [int(w/2),w-int(w/2)] # 避免 w为奇数 除不尽的情况 如果为1 或0怎么处理 为0 后续则不取该图像
            pad_h_list = [int(h/2),h-int(h/2)]
            padding = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) # pad with 0
            pad_img = padding(img,) # pad img
            padding_label = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 255) 
            ## 不需要pad with 255，这里黑边就是背景 pred是经过通道argmax出来的 label1 label2是预测出来的，
            pad_label1 = padding_label(label1) # pad label
            pad_label2 = padding_label(label2)
            pad_weight_1 = padding(weight1) # pad 0
            pad_weight_2 = padding(weight2)
            # class guide
            #classes = torch.unique(pred) # 众数
            #classes = classes[classes!= 255] # 250 for city 淡黄色轮廓   按通道算argmax 那就不可能有255
            #if classes.shape[0]>2:
            #    continue
            mask = pred==0 # false 为前景 true为背景
            #fg_pixle_num= torch.sum(~mask) # 
            #fg_ratio = fg_pixle_num / ( unsup_imgs.shape[2]*unsup_imgs.shape[3]) # 0.99
            #if ratio>0.7 or ratio<0.1 如果没有前景 如果只是小块区域 怎么处理
            #    continue
            #imask = pred==0
            #if not imask.equal(mask): pass
            #print(mask.type)
            row, _ =torch.min(mask,axis=1) # 修改 _, row
            col, _ =torch.min(mask,axis=0)
            row = row.int()
            col = col.int()
            mn_row=torch.argmin(row)
            mx_row=row.shape[0]-torch.argmin(row.flip(0))-1 # torch 不支持反向索引 [::-1]
            mn_col=torch.argmin(col)
            mx_col=col.shape[0]-torch.argmin(col.flip(0))-1
            
            center_y = torch.randint( int(h/2) + mn_row, int(h/2) + mx_row + 1,[] ) # get center point 均匀分布 还是正态分布
            center_x = torch.randint( int(w/2) + mn_col, int(w/2) + mx_col + 1, [] ) 
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2)
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
                                                                                                            
            cropped_images[k] = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            l1_label[k] = pad_label1[ new_min_row:new_max_row, new_min_col:new_max_col]
            l2_label[k] = pad_label2[ new_min_row:new_max_row, new_min_col:new_max_col]
            l1_weight[k] = pad_weight_1[ new_min_row:new_max_row, new_min_col:new_max_col]
            l2_weight[k] = pad_weight_2[ new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        l1_mixed_label.append( torch.cat( 
                            ( torch.cat((l1_label[0],l1_label[1]),1),
                                torch.cat((l1_label[2],l1_label[3]),1) ), 0 )) 
        l2_mixed_label.append( torch.cat( 
                            ( torch.cat((l2_label[0],l2_label[1]),1),
                                torch.cat((l2_label[2],l2_label[3]),1) ), 0 )) 
        l1_mixed_weight.append( torch.cat( 
                            ( torch.cat((l1_weight[0],l1_weight[1]),1),
                                torch.cat((l1_weight[2],l1_weight[3]),1) ), 0 )) 
        l2_mixed_weight.append( torch.cat( 
                            ( torch.cat((l2_weight[0],l2_weight[1]),1),
                                torch.cat((l2_weight[2],l2_weight[3]),1) ), 0 )) 
                                    
    unsup_imgs_mixed = torch.stack(mixed_img)
    ps_label_1 = torch.stack(l1_mixed_label)
    ps_label_2 = torch.stack(l2_mixed_label)
    weight_1 = torch.stack(l1_mixed_weight)
    weight_2 = torch.stack(l2_mixed_weight)
    if hard_id is not None:
        return unsup_imgs_mixed, ps_label_1, ps_label_2 ,aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y
    return unsup_imgs_mixed, ps_label_1, ps_label_2, weight_1, weight_2



def my_mix_nopad(data, softmax_pred, ps_label_1, ps_label_2, hard_id=None):
    num, c, I_y, I_x = data.shape   
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(0.3, 0.3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(0.3, 0.3)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
    
    max_probs, argmax_pred = torch.max(softmax_pred, dim=1) #  each pixel max probs value\class int64 
    # pad_list= (128,128,128,128)
    # pad_img=nn.ConstantPad2d(img) 
    #### 生成 cate list 
    
    #beta = 0.3
                                        
    mixed_img = []
    l1_mixed_label = []
    l2_mixed_label = []
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        l1_label = {}
        l2_label = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                
            img = data[start]   # 取起始图片
            pred = argmax_pred[start] # int64 [512,512]
            label1 = ps_label_1[start]
            label2 = ps_label_2[start]
            
            w=w_[k] # 310
            h=h_[k] # 0
            
            mask = pred==0 # false 为前景 true为背景
            
            row, _ =torch.min(mask,axis=1) # 修改 _, row
            col, _ =torch.min(mask,axis=0)
            row = row.int()
            col = col.int()
            mn_row=torch.argmin(row)
            mx_row=row.shape[0]-torch.argmin(row.flip(0))-1 # torch 不支持反向索引 [::-1]
            mn_col=torch.argmin(col)
            mx_col=col.shape[0]-torch.argmin(col.flip(0))-1
            
            center_y = torch.randint(  mn_row,  mx_row + 1,[] ) # get center point 均匀分布 还是正态分布
            center_x = torch.randint(  mn_col,  mx_col + 1, [] ) 
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2) # 采用减法防止奇数
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
            
            #边界条件
            if new_min_row<0:
                new_min_row=0
                new_max_row=h
            if new_max_row>512:
                new_max_row=512
                new_min_row=512-h
            if new_min_col<0:
                new_min_col=0
                new_max_col=w
            if new_max_col>512:
                new_max_col=512
                new_min_col=512-w
                                                                                                            
            cropped_images[k] = img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            l1_label[k] = label1[ new_min_row:new_max_row, new_min_col:new_max_col]
            l2_label[k] = label2[ new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        l1_mixed_label.append( torch.cat( 
                            ( torch.cat((l1_label[0],l1_label[1]),1),
                                torch.cat((l1_label[2],l1_label[3]),1) ), 0 )) 
        l2_mixed_label.append( torch.cat( 
                            ( torch.cat((l2_label[0],l2_label[1]),1),
                                torch.cat((l2_label[2],l2_label[3]),1) ), 0 )) 
                                    
    unsup_imgs_mixed = torch.stack(mixed_img)
    ps_label_1 = torch.stack(l1_mixed_label)
    ps_label_2 = torch.stack(l2_mixed_label)

    return unsup_imgs_mixed, ps_label_1, ps_label_2

def my_mix_sup(data, label):
    num, c, I_y, I_x = data.shape   
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(0.3, 0.3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(0.3, 0.3)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
    #### 生成 cate list 
    #beta = 0.3                                       
    mixed_img = []
    mixed_label = []
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        crop_label = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                
            '''if i==hard_id : # 如果是后续循环且要切的图像是hard图像 就直接赋值hard_img并跳过后续 如果是 5 6 7 0 则 不好赋值
                cropped_images[k] = hard_img
                l1_label[k] = hard_label1
                l2_label[k] = hard_label2
                continue
            '''
            img = data[start]   # 取起始图片
            gt = label[start]
            
            w=w_[k] # 310
            h=h_[k] # 0
            # pad
            pad_w_list = [int(w/2),w-int(w/2)] # 避免 w为奇数 除不尽的情况 如果为1 或0怎么处理 为0 后续则不取该图像
            pad_h_list = [int(h/2),h-int(h/2)]
            padding = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) # pad with 0 当使用gt时 应padwith 255
            pad_img = padding(img,) # pad img
            padding_label = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 255) 
            ## 不需要pad with 255，这里黑边就是背景 pred是经过通道argmax出来的 label1 label2是预测出来的，
            pad_gt = padding_label(gt) # pad label
            # class guide
            #classes = torch.unique(pred) # 众数
            #classes = classes[classes!= 255] # 250 for city 淡黄色轮廓   按通道算argmax 那就不可能有255
            #if classes.shape[0]>2:
            #    continue
            mask = gt==0 # false 为前景 true为背景
            #fg_pixle_num= torch.sum(~mask) # 
            #fg_ratio = fg_pixle_num / ( unsup_imgs.shape[2]*unsup_imgs.shape[3]) # 0.99
            #if ratio>0.7 or ratio<0.1 如果没有前景 如果只是小块区域 怎么处理
            #    continue
            #imask = pred==0
            #if not imask.equal(mask): pass
            #print(mask.type)
            row, _ =torch.min(mask,axis=1) # 修改 _, row
            col, _ =torch.min(mask,axis=0)
            row = row.int()
            col = col.int()
            mn_row=torch.argmin(row)
            mx_row=row.shape[0]-torch.argmin(row.flip(0))-1 # torch 不支持反向索引 [::-1]
            mn_col=torch.argmin(col)
            mx_col=col.shape[0]-torch.argmin(col.flip(0))-1
            
            center_y = torch.randint( int(h/2) + mn_row, int(h/2) + mx_row + 1,[] ) # get center point 均匀分布 还是正态分布
            center_x = torch.randint( int(w/2) + mn_col, int(w/2) + mx_col + 1, [] ) 
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2)
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
                                                                                                            
            cropped_images[k] = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            crop_label[k] = pad_gt[ new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        mixed_label.append( torch.cat( 
                            ( torch.cat((crop_label[0],crop_label[1]),1),
                                torch.cat((crop_label[2],crop_label[3]),1) ), 0 )) 
                                    
    sup_imgs_mixed = torch.stack(mixed_img)
    gts_label = torch.stack(mixed_label)

    return sup_imgs_mixed, gts_label

def my_mix_nopad_sup(data, label, hard_id=None):
    num, c, I_y, I_x = data.shape   
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(0.3, 0.3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(0.3, 0.3)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
            
    if hard_id is not None:# 没有限制num
        # if k==0:
        cat_x = int(np.round(I_x * np.random.beta(3, 3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(3, 3)))# 19
        
        aug_pos1=(hard_id+1)%num
        aug_pos2=(hard_id+2)%num
        aug_pos3=(hard_id+3)%num
        
        cat_position_x[hard_id] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ] # 修改cat分布
        cat_position_y[hard_id] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos1] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]# same
        cat_position_y[aug_pos1] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos2] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# rotate
        cat_position_y[aug_pos2] = [ I_y-cat_y, I_y-cat_y, cat_y, cat_y ]
        
        cat_position_x[aug_pos3] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# mirror
        cat_position_y[aug_pos3] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]    
        
    '''
    # hard sample detect
    
    if engine.distributed and (engine.local_rank == 0):
        #test= torch.zeros((4,3,512,512)).cuda()
        #tb_logger.add_graph(model, test)
        #tb_logger.add_image("img",entropy)
        tb_logger.add_histogram( 'batch_entropy', entropy, epoch * config.niters_per_epoch + idx) 
        #for i in range(num):
        #    tb_logger.add_histogram( 'entropy'+ str(i), entropy[i], epoch * config.niters_per_epoch + idx) 
        imgnum = 4*(epoch * config.niters_per_epoch + idx)    
        for g in range(num):
            tb_logger.add_histogram( 'entropy', entropy[g], imgnum ) 
            imgnum+=1
        tb_logger.flush()
    '''
    
    #### 生成 cate list 
    #beta = 0.3
                                        
    mixed_img = []
    mixed_label = []
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        crop_label = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                
            '''if i==hard_id : # 如果是后续循环且要切的图像是hard图像 就直接赋值hard_img并跳过后续 如果是 5 6 7 0 则 不好赋值
                cropped_images[k] = hard_img
                l1_label[k] = hard_label1
                l2_label[k] = hard_label2
                continue
            '''
            img = data[start]   # 取起始图片
            gt = label[start] # int64 [512,512]
            
            w=w_[k] # 310
            h=h_[k] # 0
            
            # class guide
            #classes = torch.unique(pred) # 众数
            #classes = classes[classes!= 255] # 250 for city 淡黄色轮廓   按通道算argmax 那就不可能有255
            #if classes.shape[0]>2:
            #    continue
            mask = gt==0 # false 为前景 true为背景
            #fg_pixle_num= torch.sum(~mask) # 
            #fg_ratio = fg_pixle_num / ( unsup_imgs.shape[2]*unsup_imgs.shape[3]) # 0.99
            #if ratio>0.7 or ratio<0.1 如果没有前景 如果只是小块区域 怎么处理
            #    continue
            #imask = pred==0
            #if not imask.equal(mask): pass
            #print(mask.type)
            row, _ =torch.min(mask,axis=1) # 修改 _, row
            col, _ =torch.min(mask,axis=0)
            row = row.int()
            col = col.int()
            mn_row=torch.argmin(row)
            mx_row=row.shape[0]-torch.argmin(row.flip(0))-1 # torch 不支持反向索引 [::-1]
            mn_col=torch.argmin(col)
            mx_col=col.shape[0]-torch.argmin(col.flip(0))-1
            
            center_y = torch.randint(  mn_row,  mx_row + 1,[] ) # get center point 均匀分布 还是正态分布
            center_x = torch.randint(  mn_col,  mx_col + 1, [] ) 
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2) # 采用减法防止奇数
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
            
            #边界条件
            if new_min_row<0:
                new_min_row=0
                new_max_row=h
            if new_max_row>512:
                new_max_row=512
                new_min_row=512-h
            if new_min_col<0:
                new_min_col=0
                new_max_col=w
            if new_max_col>512:
                new_max_col=512
                new_min_col=512-w

            if hard_id is not None and i == hard_id and start == hard_id: # 在hard_id个混合图像 的开始图像为困难图像时 储存 hard_area
                hard_img = img[ :, new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label = label[ new_min_row:new_max_row, new_min_col:new_max_col]
                                                                                                            
            cropped_images[k] = img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            crop_label[k] = gt[ new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        mixed_label.append( torch.cat( 
                            ( torch.cat((crop_label[0],crop_label[1]),1),
                                torch.cat((crop_label[2],crop_label[3]),1) ), 0 )) 
        
        
    if hard_id is not None:
        mixed_img[ aug_pos1 ][ :, 0:cat_y, 0:cat_x ] = hard_img
        mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label
        
        mixed_img[ aug_pos2 ][ :, I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_img
        mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label

        mixed_img[ aug_pos3 ][ :, 0:cat_y, I_x-cat_x:I_x ] = hard_img
        mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label
                                    
    sup_imgs_mixed = torch.stack(mixed_img)
    gt_mixed = torch.stack(mixed_label)
    if hard_id is not None:
        return sup_imgs_mixed, gt_mixed, aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y
    return sup_imgs_mixed, gt_mixed


def my_mix_logit(data, softmax_pred, ps_label_1, ps_label_2, beta=0.3, hard_id=None): # ps_label_12 as logit
    num, c, I_y, I_x = data.shape   
    #beta=0.3 # 23 beta参数设置为0.3 0.3 beta=2
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(beta, beta)))# 23 beta参数设置为0.3 0.3 beta=2
        cat_y = int(np.round(I_y * np.random.beta(beta, beta)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
            
    if hard_id is not None:# 没有限制num
        # if k==0:
        cat_x = int(np.round(I_x * np.random.beta(3, 3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(3, 3)))# 19
        
        aug_pos1=(hard_id+1)%num
        aug_pos2=(hard_id+2)%num
        aug_pos3=(hard_id+3)%num
        
        cat_position_x[hard_id] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ] # 修改cat分布
        cat_position_y[hard_id] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos1] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]# same
        cat_position_y[aug_pos1] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos2] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# rotate
        cat_position_y[aug_pos2] = [ I_y-cat_y, I_y-cat_y, cat_y, cat_y ]
        
        cat_position_x[aug_pos3] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# mirror
        cat_position_y[aug_pos3] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]    
        
    '''
    # hard sample detect
    
    if engine.distributed and (engine.local_rank == 0):
        #test= torch.zeros((4,3,512,512)).cuda()
        #tb_logger.add_graph(model, test)
        #tb_logger.add_image("img",entropy)
        tb_logger.add_histogram( 'batch_entropy', entropy, epoch * config.niters_per_epoch + idx) 
        #for i in range(num):
        #    tb_logger.add_histogram( 'entropy'+ str(i), entropy[i], epoch * config.niters_per_epoch + idx) 
        imgnum = 4*(epoch * config.niters_per_epoch + idx)    
        for g in range(num):
            tb_logger.add_histogram( 'entropy', entropy[g], imgnum ) 
            imgnum+=1
        tb_logger.flush()
    '''
    
    max_probs, argmax_pred = torch.max(softmax_pred, dim=1) #  each pixel max probs value\class int64 
    # pad_list= (128,128,128,128)
    # pad_img=nn.ConstantPad2d(img) 
    #### 生成 cate list 
    
    #beta = 0.3
                                        
    mixed_img = []
    l1_mixed_label = []
    l2_mixed_label = []
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        l1_label = {}
        l2_label = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                

            img = data[start]   # 取起始图片
            pred = argmax_pred[start] # int64 [512,512]
            label1 = ps_label_1[start]
            label2 = ps_label_2[start]
            
            w=w_[k] # 310
            h=h_[k] # 0
            # pad
            pad_w_list = [int(w/2),w-int(w/2)] 
            pad_h_list = [int(h/2),h-int(h/2)]
            padding = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) # pad with 0
            pad_img = padding(img,) # pad img
            padding_label = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 255) 
          
            pad_label1 = padding_label(label1) # pad label
            pad_label2 = padding_label(label2)
            # class guide
            #classes = torch.unique(pred) # 众数
            #classes = classes[classes!= 255] # 250 for city 淡黄色轮廓  
            #if classes.shape[0]>2:
            #    continue
            mask = pred==0 # false 为前景 true为背景
            #fg_pixle_num= torch.sum(~mask) # 
            #fg_ratio = fg_pixle_num / ( unsup_imgs.shape[2]*unsup_imgs.shape[3]) # 0.99
            #if ratio>0.7 or ratio<0.1 
            #    continue
            #imask = pred==0
            #if not imask.equal(mask): pass
            #print(mask.type)
            row, _ =torch.min(mask,axis=1) # 修改 _, row
            col, _ =torch.min(mask,axis=0)
            row = row.int()
            col = col.int()
            mn_row=torch.argmin(row)
            mx_row=row.shape[0]-torch.argmin(row.flip(0))-1 # torch 不支持反向索引 [::-1]
            mn_col=torch.argmin(col)
            mx_col=col.shape[0]-torch.argmin(col.flip(0))-1
            
            center_y = torch.randint( int(h/2) + mn_row, int(h/2) + mx_row + 1,[] ) # get center point 均匀分布 还是正态分布
            center_x = torch.randint( int(w/2) + mn_col, int(w/2) + mx_col + 1, [] ) 
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2)
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
            
            if hard_id is not None and i == hard_id and start == hard_id: # 在hard_id个混合图像 的开始图像为困难图像时 储存 hard_area
                hard_img = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label1 = pad_label1[:, new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label2 = pad_label2[:, new_min_row:new_max_row, new_min_col:new_max_col]  
                                                                                                            
            cropped_images[k] = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            l1_label[k] = pad_label1[:, new_min_row:new_max_row, new_min_col:new_max_col]
            l2_label[k] = pad_label2[:, new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        l1_mixed_label.append( torch.cat( 
                            ( torch.cat((l1_label[0],l1_label[1]),2),
                                torch.cat((l1_label[2],l1_label[3]),2) ), 1 )) 
        l2_mixed_label.append( torch.cat( 
                            ( torch.cat((l2_label[0],l2_label[1]),2),
                                torch.cat((l2_label[2],l2_label[3]),2) ), 1 )) 
        
    if hard_id is not None:
        mixed_img[ aug_pos1 ][ :, 0:cat_y, 0:cat_x ] = hard_img
        l1_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label1
        l2_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label2
        
        mixed_img[ aug_pos2 ][ :, I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label2
        
        mixed_img[ aug_pos3 ][ :, 0:cat_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label2
                                    
    unsup_imgs_mixed = torch.stack(mixed_img)
    ps_label_1 = torch.stack(l1_mixed_label)
    ps_label_2 = torch.stack(l2_mixed_label)
    if hard_id is not None:
        return unsup_imgs_mixed, ps_label_1, ps_label_2 ,aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y
    return unsup_imgs_mixed, ps_label_1, ps_label_2

def my_mix_logit_nopad(data, softmax_pred, ps_label_1, ps_label_2, hard_id=None): # ps_label_12 as logit
    num, c, I_y, I_x = data.shape   
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(0.3, 0.3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(0.3, 0.3)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
            
    if hard_id is not None:# 没有限制num
        # if k==0:
        cat_x = int(np.round(I_x * np.random.beta(3, 3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(3, 3)))# 19
        
        aug_pos1=(hard_id+1)%num
        aug_pos2=(hard_id+2)%num
        aug_pos3=(hard_id+3)%num
        
        cat_position_x[hard_id] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ] # 修改cat分布
        cat_position_y[hard_id] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos1] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]# same
        cat_position_y[aug_pos1] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos2] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# rotate
        cat_position_y[aug_pos2] = [ I_y-cat_y, I_y-cat_y, cat_y, cat_y ]
        
        cat_position_x[aug_pos3] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# mirror
        cat_position_y[aug_pos3] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]    
        
    '''
    # hard sample detect
    
    if engine.distributed and (engine.local_rank == 0):
        #test= torch.zeros((4,3,512,512)).cuda()
        #tb_logger.add_graph(model, test)
        #tb_logger.add_image("img",entropy)
        tb_logger.add_histogram( 'batch_entropy', entropy, epoch * config.niters_per_epoch + idx) 
        #for i in range(num):
        #    tb_logger.add_histogram( 'entropy'+ str(i), entropy[i], epoch * config.niters_per_epoch + idx) 
        imgnum = 4*(epoch * config.niters_per_epoch + idx)    
        for g in range(num):
            tb_logger.add_histogram( 'entropy', entropy[g], imgnum ) 
            imgnum+=1
        tb_logger.flush()
    '''
    
    max_probs, argmax_pred = torch.max(softmax_pred, dim=1) #  each pixel max probs value\class int64 
    # pad_list= (128,128,128,128)
    # pad_img=nn.ConstantPad2d(img) 
    #### 生成 cate list 
    
    #beta = 0.3
                                        
    mixed_img = []
    l1_mixed_label = []
    l2_mixed_label = []
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        l1_label = {}
        l2_label = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                
            '''if i==hard_id : # 如果是后续循环且要切的图像是hard图像 就直接赋值hard_img并跳过后续 如果是 5 6 7 0 则 不好赋值
                cropped_images[k] = hard_img
                l1_label[k] = hard_label1
                l2_label[k] = hard_label2
                continue
            '''
            img = data[start]   # 取起始图片
            pred = argmax_pred[start] # int64 [512,512]
            label1 = ps_label_1[start]
            label2 = ps_label_2[start]
            
            w=w_[k] # 310
            h=h_[k] # 0
            
            # class guide
            #classes = torch.unique(pred) # 众数
            #classes = classes[classes!= 255] # 250 for city 淡黄色轮廓   按通道算argmax 那就不可能有255
            #if classes.shape[0]>2:
            #    continue
            mask = pred==0 # false 为前景 true为背景
            #fg_pixle_num= torch.sum(~mask) # 
            #fg_ratio = fg_pixle_num / ( unsup_imgs.shape[2]*unsup_imgs.shape[3]) # 0.99
            #if ratio>0.7 or ratio<0.1 如果没有前景 如果只是小块区域 怎么处理
            #    continue
            #imask = pred==0
            #if not imask.equal(mask): pass
            #print(mask.type)
            row, _ =torch.min(mask,axis=1) # 修改 _, row
            col, _ =torch.min(mask,axis=0)
            row = row.int()
            col = col.int()
            mn_row=torch.argmin(row)
            mx_row=row.shape[0]-torch.argmin(row.flip(0))-1 # torch 不支持反向索引 [::-1]
            mn_col=torch.argmin(col)
            mx_col=col.shape[0]-torch.argmin(col.flip(0))-1
            
            center_y = torch.randint( int(h/2) + mn_row, int(h/2) + mx_row + 1,[] ) # get center point 均匀分布 还是正态分布
            center_x = torch.randint( int(w/2) + mn_col, int(w/2) + mx_col + 1, [] ) 
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2)
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
            
            #边界条件
            if new_min_row<0:
                new_min_row=0
                new_max_row=h
            if new_max_row>512:
                new_max_row=512
                new_min_row=512-h
            if new_min_col<0:
                new_min_col=0
                new_max_col=w
            if new_max_col>512:
                new_max_col=512
                new_min_col=512-w
            
            if hard_id is not None and i == hard_id and start == hard_id: # 在hard_id个混合图像 的开始图像为困难图像时 储存 hard_area
                hard_img = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label1 = pad_label1[:, new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label2 = pad_label2[:, new_min_row:new_max_row, new_min_col:new_max_col]  
                                                                                                            
            cropped_images[k] = img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            l1_label[k] = label1[:, new_min_row:new_max_row, new_min_col:new_max_col]
            l2_label[k] = label2[:, new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        l1_mixed_label.append( torch.cat( 
                            ( torch.cat((l1_label[0],l1_label[1]),2),
                                torch.cat((l1_label[2],l1_label[3]),2) ), 1 )) 
        l2_mixed_label.append( torch.cat( 
                            ( torch.cat((l2_label[0],l2_label[1]),2),
                                torch.cat((l2_label[2],l2_label[3]),2) ), 1 )) 
        
    if hard_id is not None:
        mixed_img[ aug_pos1 ][ :, 0:cat_y, 0:cat_x ] = hard_img
        l1_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label1
        l2_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label2
        
        mixed_img[ aug_pos2 ][ :, I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label2
        
        mixed_img[ aug_pos3 ][ :, 0:cat_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label2
                                    
    unsup_imgs_mixed = torch.stack(mixed_img)
    ps_label_1 = torch.stack(l1_mixed_label)
    ps_label_2 = torch.stack(l2_mixed_label)
    if hard_id is not None:
        return unsup_imgs_mixed, ps_label_1, ps_label_2 ,aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y
    return unsup_imgs_mixed, ps_label_1, ps_label_2

def my_mix_halfclass(data, softmax_pred, ps_label_1, ps_label_2, hard_id=None):
    num, c, I_y, I_x = data.shape   
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(0.3, 0.3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(0.3, 0.3)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
            
    if hard_id is not None:# 没有限制num
        # if k==0:
        cat_x = int(np.round(I_x * np.random.beta(3, 3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(3, 3)))# 19
        
        aug_pos1=(hard_id+1)%num
        aug_pos2=(hard_id+2)%num
        aug_pos3=(hard_id+3)%num
        
        cat_position_x[hard_id] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ] # 修改cat分布
        cat_position_y[hard_id] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos1] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]# same
        cat_position_y[aug_pos1] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos2] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# rotate
        cat_position_y[aug_pos2] = [ I_y-cat_y, I_y-cat_y, cat_y, cat_y ]
        
        cat_position_x[aug_pos3] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# mirror
        cat_position_y[aug_pos3] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]    
        
    '''
    # hard sample detect
    
    if engine.distributed and (engine.local_rank == 0):
        #test= torch.zeros((4,3,512,512)).cuda()
        #tb_logger.add_graph(model, test)
        #tb_logger.add_image("img",entropy)
        tb_logger.add_histogram( 'batch_entropy', entropy, epoch * config.niters_per_epoch + idx) 
        #for i in range(num):
        #    tb_logger.add_histogram( 'entropy'+ str(i), entropy[i], epoch * config.niters_per_epoch + idx) 
        imgnum = 4*(epoch * config.niters_per_epoch + idx)    
        for g in range(num):
            tb_logger.add_histogram( 'entropy', entropy[g], imgnum ) 
            imgnum+=1
        tb_logger.flush()
    '''
    
    max_probs, argmax_pred = torch.max(softmax_pred, dim=1) #  each pixel max probs value\class int64 
    # pad_list= (128,128,128,128)
    # pad_img=nn.ConstantPad2d(img) 
    #### 生成 cate list 
    
    #beta = 0.3
                                        
    mixed_img = []
    l1_mixed_label = []
    l2_mixed_label = []
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        l1_label = {}
        l2_label = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                
            '''if i==hard_id : # 如果是后续循环且要切的图像是hard图像 就直接赋值hard_img并跳过后续 如果是 5 6 7 0 则 不好赋值
                cropped_images[k] = hard_img
                l1_label[k] = hard_label1
                l2_label[k] = hard_label2
                continue
            '''
            img = data[start]   # 取起始图片
            pred = argmax_pred[start] # int64 [512,512]
            label1 = ps_label_1[start]
            label2 = ps_label_2[start]
            
            w=w_[k] # 310
            h=h_[k] # 0
            
            # if w*h==0:
            #     cropped_images[k] = torch.tensor([]).cuda()
            #     l1_label[k] = torch.tensor([]).long().cuda()
            #     l2_label[k] = torch.tensor([]).long().cuda()
            #     start+=1
            #     start%=num # start 代表 要裁剪的图像索引
            #     continue# 等于不取该图像区域 节约计算
            # pad
            pad_w_list = [int(w/2),w-int(w/2)] # 避免 w为奇数 除不尽的情况 如果为1 或0怎么处理 为0 后续则不取该图像
            pad_h_list = [int(h/2),h-int(h/2)]
            padding = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) # pad with 0
            pad_img = padding(img,) # pad img
            padding_label = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 255) 
            ## 是否需要pad with 255，这里黑边就是背景 pred是经过通道argmax出来的 此处是否会影响精度
            pad_label1 = padding_label(label1) # pad label
            pad_label2 = padding_label(label2)
            
            
            #     classes = torch.unique(pred)
            #     classes = classes[classes!= 255] # 250 for city 淡黄色轮廓
            #     nclasses = classes.shape[0]
            #     classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda() ## .long() long or float tensor
            #     mask = transformmasks.generate_class_mask(pred, classes).unsqueeze(0).cuda()
            #     return mask
            # def generate_class_mask(pred, classes):
            #     pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2)) # broadcast([1,512,512],[10,1,1]) -> [10,512,512],[10,512,512]
            #     N = pred.eq(classes).sum(0) # 若都为空，则0维度求和后从空变为0
            #     return N
            
            
            # class guide
            classes = torch.unique(pred) # 众数 获取类别数
            classes = classes[classes!=0] # 前景类
            nclasses = classes.shape[0] # 获取前景类别数 一般都包括背景吧
            if nclasses>0:
                ignore_classes_num = np.random.randint(nclasses) # 1类为0个 2类为0-1个 3类为0-2个
                ignore_classes_pos_list = torch.randperm(nclasses) # 打乱类别索引顺序
                ignore_classes_idx = classes[ ignore_classes_pos_list[:ignore_classes_num] ] # 选num个索引 取对应类idx
                ignore_classes_idx = torch.cat((ignore_classes_idx,torch.tensor([0]).cuda())) # 把背景类加回来
            else:
                ignore_classes_idx = classes # 此时类中只有0
            ignore_mask = transformmasks.generate_class_mask(pred, ignore_classes_idx) # 获得背景类与将要删除类的mask
            
            #classes = classes[classes!= 255] # 250 for city 淡黄色轮廓   按通道算argmax 那就不可能有255
            #if classes.shape[0]>2:
            #    continue
            # mask = pred==0 # false 为前景 true为背景
            #fg_pixle_num= torch.sum(~mask) # 
            #fg_ratio = fg_pixle_num / ( unsup_imgs.shape[2]*unsup_imgs.shape[3]) # 0.99
            #if ratio>0.7 or ratio<0.1 如果没有前景 如果只是小块区域 怎么处理
            #    continue
            #imask = pred==0
            #if not imask.equal(mask): pass
            #print(mask.type)
            #维度是从外层到内层数 axis=0代表最外边维度 axis=1代表内部维度 一维向量只有列维度
            # axis=1 为横向 axis=0为纵向
            row, _ =torch.min(ignore_mask,axis=1) #如果预测图像中没有目标 也就是矩阵全一 如何处理 最小的就是1
            col, _ =torch.min(ignore_mask,axis=0)
            mn_row=torch.argmin(row) # 
            mx_row=row.shape[0]-torch.argmin(row.flip(0))-1 # torch 不支持反向索引 [::-1]
            mn_col=torch.argmin(col)
            mx_col=col.shape[0]-torch.argmin(col.flip(0))-1
            
            center_y = torch.randint( int(h/2) + mn_row, int(h/2) + mx_row + 1,[] ) # 注意h为奇数的处理 
            #并且为了避免总是中心点总在边界 可以换一个分布函数 get center point 均匀分布 还是正态分布
            center_x = torch.randint( int(w/2) + mn_col, int(w/2) + mx_col + 1, [] ) 
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2)
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
            
            if hard_id is not None and i == hard_id and start == hard_id: # 在hard_id个混合图像 的开始图像为困难图像时 储存 hard_area
                hard_img = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label1 = pad_label1[ new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label2 = pad_label2[ new_min_row:new_max_row, new_min_col:new_max_col]  
                                                                                                            
            cropped_images[k] = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            l1_label[k] = pad_label1[ new_min_row:new_max_row, new_min_col:new_max_col]
            l2_label[k] = pad_label2[ new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        l1_mixed_label.append( torch.cat( 
                            ( torch.cat((l1_label[0],l1_label[1]),1),
                                torch.cat((l1_label[2],l1_label[3]),1) ), 0 )) 
        l2_mixed_label.append( torch.cat( 
                            ( torch.cat((l2_label[0],l2_label[1]),1),
                                torch.cat((l2_label[2],l2_label[3]),1) ), 0 )) 
        
    if hard_id is not None:
        mixed_img[ aug_pos1 ][ :, 0:cat_y, 0:cat_x ] = hard_img
        l1_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label1
        l2_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label2
        
        mixed_img[ aug_pos2 ][ :, I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label2
        
        mixed_img[ aug_pos3 ][ :, 0:cat_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label2
                                    
    unsup_imgs_mixed = torch.stack(mixed_img)
    ps_label_1 = torch.stack(l1_mixed_label)
    ps_label_2 = torch.stack(l2_mixed_label)
    if hard_id is not None:
        return unsup_imgs_mixed, ps_label_1, ps_label_2 ,aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y
    return unsup_imgs_mixed, ps_label_1, ps_label_2


########## saliency
def saliency_bbox(img, lam):#[100,3,32,32]
    """ generate saliency box by lam """
    size = img.size() #[3,512,512]
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam) # 
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # force fp32 when convert to numpy
    img = img.type(torch.float32)
    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    
    temp_img = img.cpu().numpy().transpose(1, 2, 0) # [32,32,3]
    saliency = cv2.saliency.StaticSaliencyFineGrained_create() #
    (success, saliencyMap) = saliency.computeSaliency(temp_img) #[512,512]
    saliencyMap = (saliencyMap * 255).astype("uint8")
    maximum_indices = np.unravel_index( np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    # argmax得到最大值的faltten位置，在转化为行列位置
    # 效果不好
    y = maximum_indices[0] # 行数
    x = maximum_indices[1] # 列数
    # bbx1 = np.clip(x - cut_w // 2, 0, W) 记得调换x和y y为行数 x为列数
    # bby1 = np.clip(y - cut_h // 2, 0, H)
    # bbx2 = np.clip(x + cut_w // 2, 0, W)
    # bby2 = np.clip(y + cut_h // 2, 0, H)
    return y,x,saliencyMap #bbx1, bby1, bbx2, bby2
def mymix_saliency(data, ps_label_1, ps_label_2, hard_id=None):
    # 删除pred
    num, c, I_y, I_x = data.shape #    
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(0.3, 0.3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(0.3, 0.3)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
            
    if hard_id is not None:# 没有限制num
        # if k==0:
        cat_x = int(np.round(I_x * np.random.beta(3, 3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(3, 3)))# 19
        
        aug_pos1=(hard_id+1)%num
        aug_pos2=(hard_id+2)%num
        aug_pos3=(hard_id+3)%num
        
        cat_position_x[hard_id] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ] # 修改cat分布
        cat_position_y[hard_id] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos1] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]# same
        cat_position_y[aug_pos1] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos2] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# rotate
        cat_position_y[aug_pos2] = [ I_y-cat_y, I_y-cat_y, cat_y, cat_y ]
        
        cat_position_x[aug_pos3] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# mirror
        cat_position_y[aug_pos3] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]    
    
    # max_probs, argmax_pred = torch.max(softmax_pred, dim=1) #  each pixel max probs value\class int64 
    # pad_list= (128,128,128,128)
    # pad_img=nn.ConstantPad2d(img) 
    #### 生成 cate list 
    
    # beta = 0.3
    # 获取每个样本的att最值点
    
    ###  
    lam = np.random.beta( 0.3, 0.3)
    saliency_pos=[]
    saliency_maps=[]
    for single_img in data:
        y,x, saliency_map = saliency_bbox( single_img, lam)
        saliency_pos.append([y,x])
        saliency_maps.append(saliency_map)
    saliency_maps = np.stack(saliency_maps, axis=0)    
    # [4,2,1]                                   
    mixed_img = []
    l1_mixed_label = []
    l2_mixed_label = []
    
    
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        l1_label = {}
        l2_label = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                
            '''if i==hard_id : # 如果是后续循环且要切的图像是hard图像 就直接赋值hard_img并跳过后续 如果是 5 6 7 0 则 不好赋值
                cropped_images[k] = hard_img
                l1_label[k] = hard_label1
                l2_label[k] = hard_label2
                continue
            '''
            img = data[start]   # 取起始图片
            #pred = argmax_pred[start] # int64 [512,512]
            label1 = ps_label_1[start]
            label2 = ps_label_2[start]
            #beta = 0.3
            #lam = np.random.beta(beta, beta)
            #top_k = min(max(1, int(att_grid * lam)), att_grid) 初步选为最高值
           
            
            w=w_[k] # 310
            h=h_[k] # 0
            # pad
            pad_w_list = [int(w/2), int(w/2)] # 避免 w为奇数 除不尽的情况 如果为1 或0怎么处理 为0 后续则不取该图像
            pad_h_list = [int(h/2), int(h/2)]
            padding = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) # pad with 0
            pad_img = padding(img,) # pad img
            # padding_label = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) 
            ## 不需要pad with 255，这里黑边就是背景 pred是经过通道argmax出来的
            pad_label1 = padding(label1) # pad label
            pad_label2 = padding(label2)
            
            center_y = saliency_pos[start][0]+int(h/2) # 当前idx 需要加上pad的行列数
            center_x = saliency_pos[start][1]+int(w/2)
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2)
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
            
            if hard_id is not None and i == hard_id and start == hard_id: # 在hard_id个混合图像 的开始图像为困难图像时 储存 hard_area
                hard_img = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label1 = pad_label1[ new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label2 = pad_label2[ new_min_row:new_max_row, new_min_col:new_max_col]  
                                                                                                            
            cropped_images[k] = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            l1_label[k] = pad_label1[ new_min_row:new_max_row, new_min_col:new_max_col]
            l2_label[k] = pad_label2[ new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        l1_mixed_label.append( torch.cat( 
                            ( torch.cat((l1_label[0],l1_label[1]),1),
                                torch.cat((l1_label[2],l1_label[3]),1) ), 0 )) 
        l2_mixed_label.append( torch.cat( 
                            ( torch.cat((l2_label[0],l2_label[1]),1),
                                torch.cat((l2_label[2],l2_label[3]),1) ), 0 )) 
        
    if hard_id is not None:
        mixed_img[ aug_pos1 ][ :, 0:cat_y, 0:cat_x ] = hard_img
        l1_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label1
        l2_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label2
        
        mixed_img[ aug_pos2 ][ :, I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label2
        
        mixed_img[ aug_pos3 ][ :, 0:cat_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label2
                                    
    unsup_imgs_mixed = torch.stack(mixed_img)
    ps_label_1 = torch.stack(l1_mixed_label)
    ps_label_2 = torch.stack(l2_mixed_label)
    if hard_id is not None:
        return unsup_imgs_mixed, ps_label_1, ps_label_2 ,saliency_maps, aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y
    return unsup_imgs_mixed, ps_label_1, ps_label_2, saliency_maps

############ attentive
def mymix_attentive(data, features, ps_label_1, ps_label_2, hard_id=None):
    # 删除pred
    num, c, I_y, I_x = data.shape #    
    for k in range(num):
        cat_x = int(np.round(I_x * np.random.beta(0.3, 0.3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(0.3, 0.3)))# 195
        if k==0:
            cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
            cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
        else:
            cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
            cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )
            
    if hard_id is not None:# 没有限制num
        # if k==0:
        cat_x = int(np.round(I_x * np.random.beta(3, 3)))# 23 beta参数设置为0.3 0.3
        cat_y = int(np.round(I_y * np.random.beta(3, 3)))# 19
        
        aug_pos1=(hard_id+1)%num
        aug_pos2=(hard_id+2)%num
        aug_pos3=(hard_id+3)%num
        
        cat_position_x[hard_id] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ] # 修改cat分布
        cat_position_y[hard_id] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos1] = [ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]# same
        cat_position_y[aug_pos1] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]
        
        cat_position_x[aug_pos2] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# rotate
        cat_position_y[aug_pos2] = [ I_y-cat_y, I_y-cat_y, cat_y, cat_y ]
        
        cat_position_x[aug_pos3] = [ I_x-cat_x, cat_x, I_x-cat_x, cat_x ]# mirror
        cat_position_y[aug_pos3] = [ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]    
    
    # max_probs, argmax_pred = torch.max(softmax_pred, dim=1) #  each pixel max probs value\class int64 
    # pad_list= (128,128,128,128)
    # pad_img=nn.ConstantPad2d(img) 
    #### 生成 cate list 
    
    # beta = 0.3
    # 获取每个样本的att最值点
    
    ###  如何处理feature 使用多通道avg？ 空间注意力？
    #features[4，256，128，128]
    score = features.mean(1).unsqueeze(1) # [4,128,128]  平均池化  还是 最大池化？
    score = nn.functional.interpolate( score, size=(I_y,I_x), mode='bilinear', align_corners=False) 
    score = score.squeeze(1) # [4,512,512]
    #att_idx= torch.argmax(score)  # 返回flatten的idx
    att_size = I_x # 获取列的宽度
    top_k=1
    _, att_idx = score.view(num, att_size**2 ).topk(top_k)
    att_idx = torch.cat([
        (torch.div(att_idx,att_size,rounding_mode='floor') ).unsqueeze(1), #[100,1,27] 除数 实际是反flatten 代表行数
        (att_idx  % att_size).unsqueeze(1),], dim=1) #[100,1,27] 余数 代表列数 [4，2] 如果设计为tok 则为[4,2,tok]
    # [4,2,1]                                   
    mixed_img = []
    l1_mixed_label = []
    l2_mixed_label = []
    
    
    #### 对应crop位置
    for i in range(num): # every img in batch has a concate position / i 选择混合的图片起始位置
        w_ = cat_position_x[i] # w list
        h_ = cat_position_y[i] # h list
        cropped_images = {} # 
        l1_label = {}
        l2_label = {}
        start = i #记录起始位置
        for k in range(4): # ricap need 4 img 由于每个裁剪点的位置是随机生成的 因此 图像与label要同步处理 / k代表 裁剪图像的存储顺序
                
            '''if i==hard_id : # 如果是后续循环且要切的图像是hard图像 就直接赋值hard_img并跳过后续 如果是 5 6 7 0 则 不好赋值
                cropped_images[k] = hard_img
                l1_label[k] = hard_label1
                l2_label[k] = hard_label2
                continue
            '''
            img = data[start]   # 取起始图片
            #pred = argmax_pred[start] # int64 [512,512]
            label1 = ps_label_1[start]
            label2 = ps_label_2[start]
            #beta = 0.3
            #lam = np.random.beta(beta, beta)
            #top_k = min(max(1, int(att_grid * lam)), att_grid) 初步选为最高值
           
            
            w=w_[k] # 310
            h=h_[k] # 0
            # pad
            pad_w_list = [int(w/2), int(w/2)] # 避免 w为奇数 除不尽的情况 如果为1 或0怎么处理 为0 后续则不取该图像
            pad_h_list = [int(h/2), int(h/2)]
            padding = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) # pad with 0
            pad_img = padding(img,) # pad img
            # padding_label = nn.ConstantPad2d((pad_w_list[0],pad_w_list[1],pad_h_list[0],pad_h_list[1]), 0) 
            ## 不需要pad with 255，这里黑边就是背景 pred是经过通道argmax出来的
            pad_label1 = padding(label1) # pad label
            pad_label2 = padding(label2)
            
            center_y = att_idx[start,0]+int(h/2) # 当前idx 需要加上pad的行列数
            center_x = att_idx[start,1]+int(w/2)
                                                                                                            
            # 截取区域
            new_max_row = center_y + (h-int(h/2))
            new_min_row = center_y - int(h/2)
            new_max_col = center_x + (w-int(w/2)) # 不需减一 因为后续提取位置会加一 
            new_min_col = center_x - int(w/2)
            
            if hard_id is not None and i == hard_id and start == hard_id: # 在hard_id个混合图像 的开始图像为困难图像时 储存 hard_area
                hard_img = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label1 = pad_label1[ new_min_row:new_max_row, new_min_col:new_max_col]
                hard_label2 = pad_label2[ new_min_row:new_max_row, new_min_col:new_max_col]  
                                                                                                            
            cropped_images[k] = pad_img[ :, new_min_row:new_max_row, new_min_col:new_max_col]# 生成4个crop
            l1_label[k] = pad_label1[ new_min_row:new_max_row, new_min_col:new_max_col]
            l2_label[k] = pad_label2[ new_min_row:new_max_row, new_min_col:new_max_col]
            
            start+=1
            start%=num # start 代表 要裁剪的图像索引
        
        mixed_img.append(      torch.cat( 
                            ( torch.cat((cropped_images[0],cropped_images[1]),2),
                                torch.cat((cropped_images[2],cropped_images[3]),2) ), 1 ) )  
        l1_mixed_label.append( torch.cat( 
                            ( torch.cat((l1_label[0],l1_label[1]),1),
                                torch.cat((l1_label[2],l1_label[3]),1) ), 0 )) 
        l2_mixed_label.append( torch.cat( 
                            ( torch.cat((l2_label[0],l2_label[1]),1),
                                torch.cat((l2_label[2],l2_label[3]),1) ), 0 )) 
        
    if hard_id is not None:
        mixed_img[ aug_pos1 ][ :, 0:cat_y, 0:cat_x ] = hard_img
        l1_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label1
        l2_mixed_label[ aug_pos1 ][ 0:cat_y, 0:cat_x ] = hard_label2
        
        mixed_img[ aug_pos2 ][ :, I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos2 ][ I_y-cat_y:I_y, I_x-cat_x:I_x ] = hard_label2
        
        mixed_img[ aug_pos3 ][ :, 0:cat_y, I_x-cat_x:I_x ] = hard_img
        l1_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label1
        l2_mixed_label[ aug_pos3 ][ 0:cat_y, I_x-cat_x:I_x ] = hard_label2
                                    
    unsup_imgs_mixed = torch.stack(mixed_img)
    ps_label_1 = torch.stack(l1_mixed_label)
    ps_label_2 = torch.stack(l2_mixed_label)
    if hard_id is not None:
        return unsup_imgs_mixed, ps_label_1, ps_label_2 ,aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y
    return unsup_imgs_mixed, ps_label_1, ps_label_2

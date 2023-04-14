#!/usr/bin/env python3
# encoding: utf-8
try:
    from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')
import torch
import numpy as np
from PIL import Image
from utils.visualize import tensor2im
import os
import cv2
import argparse
import numpy as np

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x) # [0]

class SemanticSegmentationTarget:
    """ Gets a binary spatial mask and a category,
        And return the sum of the category scores,
        of the pixels in the mask. """

    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()
    
def gen_cam(model, input_tensor, pred_tensor,label=None,margin=None):
    ## 定义模型    
    cam_model = SegmentationModelOutputWrapper(model)
    numpy_img = tensor2im(input_tensor[0],cam=True) # [512, 512，3] numpy shapew为float32
    ignore_label = 255 # voc
    if label is not None:
        # label=label.cpu().numpy()
        #ignore_mask = (label.not_equal(ignore_label))# ignore 的 pixel值=0
        #class_mask = (label * ignore_mask)
        #ignore_mask = label!=ignore_label
        #
        class_mask = label
        class_category, count = np.unique(class_mask, return_counts=True)
        class_category = class_category[(0<class_category)&(class_category<255)]
        print("---------------", class_category)
        targets = []
        both_images = []
        if np.size(class_category)>1:
            print(class_category)
        for class_category in class_category:# 多个类别时
            
            class_mask_uint8 = 255 * np.uint8( class_mask == class_category)
            class_mask_float = np.float32( class_mask == class_category) # 只取对应类别mask做损失计算，不需要忽略255
            rgb_img = numpy_img[margin[0]:(numpy_img.shape[0] - margin[1]),
                      margin[2]:(numpy_img.shape[1] - margin[3]), :]       # 裁剪
            rgb_img = rgb_img *255 # 乘以 255 必须在uint8之前，不然先用uint8会使得小数都变为1
            rgb_img = Image.fromarray(rgb_img.astype(np.uint8),"RGB") # 转化为图像 二维
            class_mask_uint8 = class_mask_uint8[ margin[0]:(class_mask_uint8.shape[0] - margin[1]),
                    margin[2]:(class_mask_uint8.shape[1] - margin[3])]
            both_images.append( np.hstack(( rgb_img, np.repeat(class_mask_uint8[:, :, None], 3, axis=-1))) )  # 数据类型必须为uint8  
            targets.append( SemanticSegmentationTarget(class_category, class_mask_float))
            #Image.fromarray(both_images).save( save_path + '/' + 'both' + str(0) + '.png')# 数据类型必须为uint8 PIL才能转换
            
    else:        
        # pred_tensor =  
        # 转到cpu上计算输出
        
        
        normalized_masks = torch.softmax( pred_tensor, dim=0).cpu() # [21, 512, 512]

        sem_classes = [
            '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        # dataloader.get_class_names
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

        class_category = [ sem_class_to_idx["dog"] ]
        class_mask = normalized_masks.argmax(axis=0).detach().cpu().numpy() #[512,512]
        #class_category, count = np.unique(class_mask, return_counts=True)
        class_mask_uint8 = 255 * np.uint8( class_mask == class_category)
        class_mask_float = np.float32( class_mask == class_category)
        
        
        
        rgb_img = numpy_img[ margin[0]:(rgb_img.shape[0] - margin[1]),
                    margin[2]:(rgb_img.shape[1] - margin[3])]       
        class_mask_uint8 = class_mask_uint8[ margin[0]:(class_mask_uint8.shape[0] - margin[1]),
                    margin[2]:(class_mask_uint8.shape[1] - margin[3])]
        both_images = [np.hstack(( rgb_img, np.repeat(class_mask_uint8[:, :, None], 3, axis=-1)))]
        targets = [SemanticSegmentationTarget(class_category, class_mask_float)]
    
    target_layers = [cam_model.model.branch1.head.last_conv[-2]]           
    #targets = None
    '''
    if args.target_category:
        grad_cam_v = pkg_resources.get_distribution('grad_cam').version
        if digit_version(grad_cam_v) >= digit_version('1.3.7'):
            from pytorch_grad_cam.utils.model_targets import \
                ClassifierOutputTarget
            targets = [ClassifierOutputTarget(c) for c in args.target_category]
        else:
            targets = args.target_category
    '''                        
    
    with GradCAM(model=cam_model, # GradCAMPlusPlus
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor= input_tensor,
                        targets=targets)[0, :] #[512,512] 在model eval时会生成两个输出 因此在base_cam.py处修改
        #rgb_img = tensor2im(input_tensor[0], cam=True) # 重新生成未被裁剪的 0-1取值范围的图
        cam_image = show_cam_on_image(numpy_img, grayscale_cam, use_rgb=True, image_weight=0.7) #[512,512,3]
        cam_image = cam_image[ margin[0]:(cam_image.shape[0] - margin[1]),
                    margin[2]:(cam_image.shape[1] - margin[3]),:] # crop
        #Image.fromarray(cam_image ).save( save_path+ '/' + 'cam' + str(0) + '.png')  
    return both_images, cam_image
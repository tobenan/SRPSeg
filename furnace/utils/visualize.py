import numpy as np
import cv2
import scipy.io as sio

from engine.logger import get_logger
from utils.pyt_utils import ensure_dir
import os.path as osp
import torch
from PIL import Image
import numpy as np

logger =get_logger()

def set_img_color(colors, background, img, pred, gt, show255=False):
    for i in range(0, len(colors)):
        if i != background:
            img[np.where(pred == i)] = colors[i]
    if show255:
        img[np.where(gt==background)] = 255
    return img

def show_prediction(colors, background, img, pred, gt):
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred, gt)
    final = np.array(im)
    return final

def show_img(colors, background, img, clean, gt, *pds):
    im1 = np.array(img, np.uint8)
    #set_img_color(colors, background, im1, clean, gt)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        im = np.array(img, np.uint8)
        # pd[np.where(gt == 255)] = 255
        set_img_color(colors, background, im, pd, gt)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, gt, True)
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    return final

def get_colors(class_num):
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1,3)) * 255).tolist()[0])

    return colors

def get_ade_colors():
    colors = sio.loadmat('./color150.mat')['colors']
    colors = colors[:,::-1,]
    colors = np.array(colors).astype(int).tolist()
    colors.insert(0,[0,0,0])

    return colors

def print_iou(iu, mean_pixel_acc, cpa, recall, mpa, class_names=None, show_no_back=False, no_print=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % (cls, iu[i] * 100, "cpa:", cpa[i]*100, "recall", recall[i]*100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append('----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IU', mean_IU * 100, 'mean_IU_no_back', 
                                                        mean_IU_no_back*100,'mean_pixel_ACC',mean_pixel_acc*100, 'mpa', mpa*100))
    else:
        print(mean_pixel_acc)
        lines.append('----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IU', mean_IU * 100,'mean_pixel_ACC',mean_pixel_acc*100))
    lines.append('IU: ' + str(iu.flatten().tolist()) + 'CPA: ' + str(cpa.flatten().tolist()) + 'RECALL: ' + str(recall.flatten().tolist()))# 输出iu
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line

def tensor2im(input_image, cam=False,imtype=np.uint8 ):
    """"
    Parameters:
        input_image (tensor) --  输入的tensor, 维度为CHW, 注意这里没有batch size的维度
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485, 0.456, 0.406] # dataLoader中设置的mean参数，需要从dataloader中拷贝过来
    std = [0.229, 0.224, 0.225]  # dataLoader中设置的std参数，需要从dataloader中拷贝过来
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): # 如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
            
    if  image_numpy.shape[0] == 1:  # grayscale to RGB
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        
    if len(image_numpy.shape)==3: # 如果是三维图像 就反标准化  如果是mask 就跳过
        for i in range(len(mean)): # 反标准化，乘以方差，加上均值
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    else:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    if cam: return image_numpy
    image_numpy = image_numpy * 255 #反归一化 ToTensor(),从[0,1]转为[0,255]
     
    return Image.fromarray(image_numpy.astype(imtype),"RGB")

def get_class_colors(*args):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)]) # 八位二进制数
    N = 21
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    class_colors = cmap.tolist()
    return class_colors[1:]

def pred2im(pred):
    result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
    class_colors = get_class_colors()
    palette_list = list(np.array(class_colors).flat)
    if len(palette_list) < 768:
        palette_list += [0] * (768 - len(palette_list))
    result_img.putpalette(palette_list)
    return result_img
    
##### visual
def save_augimage(augimg, savedir,imtype='mix_img'):
    # type can be img, mask, label, mix_img, mix_label
    ensure_dir(savedir)
    for i in range(augimg.shape[0]):
        img = augimg[i]
        image = tensor2im(img)
        logger.info("image saved at {}".format(savedir))
        image.save( savedir + '/' + imtype + str(i) + '.jpg')

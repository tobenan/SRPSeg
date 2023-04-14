from __future__ import division
from ast import Pass, Return, arg
import os.path as osp
import os
from pickletools import uint8
from random import random
from re import S
import sys
import time
import argparse
import math
from tkinter import W
from types import ClassMethodDescriptorType
from matplotlib.pyplot import xkcd
#from sqlalchemy import false
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import VOC
from utils.init_func import init_weight, group_weight
from utils import transformmasks, transformsgpu
from utils.visualize import save_augimage

from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import ensure_dir, get_logger
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
#from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

parser = argparse.ArgumentParser()
#os.environ['MASTER_PORT'] = '169111'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

from custom_collate import SegCollate
collate_fn = SegCollate()

logger = get_logger(log_dir = config.log_dir, log_file=config.log_file) ### log 必须放在engine类前，作为其余logger的父类

with Engine(custom_parser=parser) as engine:
    
    cudnn.benchmark = True ## optimize training sped 
    args = parser.parse_args()
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, \
                                                   unsupervised=False, collate_fn=collate_fn)
    unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(engine, VOC, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn)

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb_logger = SummaryWriter(log_dir=tb_dir) # tb log
        engine.link_tb(tb_dir, generate_tb_dir)
        

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    criterion_csst = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)

    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_, # gpu 同步点
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # set the lr
    base_lr = config.lr # 0.0025
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    # define two optimizers
    params_list_l = []
    params_list_l = group_weight(params_list_l, model.branch1.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_r = torch.optim.SGD(params_list_r,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    # set decay_group weight_decay=0.0001 
    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model, device_ids=engine.devices) # model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer_l=optimizer_l, optimizer_r=optimizer_r)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    print('begin train')
    
    for epoch in range(engine.state.epoch, config.nepochs):#
        if engine.distributed:
            train_sampler.set_epoch(epoch)
            unsupervised_train_sampler.set_epoch(epoch) #  set random seed

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]' 

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)

        sum_loss_sup = 0
        sum_loss_sup_r = 0
        sum_cps = 0
        sum_area = 0
        
        if engine.local_rank==0:
            logger.info("--------{}--------".format(args.mix))
            
        entropy_file_path = "/media/ders/GDH/TorchSemiSeg/exp.voc/voc8.res50v3+.CPS+CutMix/entropy/entropy_{}.txt".format(epoch)
        hard_sample_file_path = "/media/ders/GDH/TorchSemiSeg/exp.voc/voc8.res50v3+.CPS+CutMix/hard_sample/hard_sample_{}.txt".format(epoch)
        
        
        ''' supervised part '''
        for idx in pbar:
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            minibatch = dataloader.next()
            unsup_minibatch = unsupervised_dataloader.next()
            print("num_threads: %d" % torch.get_num_threads())
            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs = unsup_minibatch['data'] #float32
            # to_device
            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_imgs = unsup_imgs.cuda(non_blocking=True)
            hard_id = None
            if args.mix is None:
                _, pred_sup_l = model(imgs, step=1)
                _, pred_unsup_l = model(unsup_imgs, step=1)
                _, pred_sup_r = model(imgs, step=2)
                _, pred_unsup_r = model(unsup_imgs, step=2)

                ### cps loss ###
                pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
                pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
                _, max_l = torch.max(pred_l, dim=1)
                _, max_r = torch.max(pred_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()
                cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)    
                
            # unsupervised loss on model/branch#1
            else: 
                with torch.no_grad(): ###
                    # Estimate the pseudo-label with branch#1 & supervise branch#2
                    _, logits_u_tea_1 = model(unsup_imgs, step=1) # [4,21,512,512]
                    logits_u_tea_1 = logits_u_tea_1.detach()

                    # Estimate the pseudo-label with branch#2 & supervise branch#1
                    _, logits_u_tea_2 = model(unsup_imgs, step=2)
                    logits_u_tea_2 = logits_u_tea_2.detach()
                    
                    
                               
                    unsup_imgs = unsup_imgs.detach()
                    num = unsup_imgs.shape[0] # get batch size

                    if args.mix == 'classmix':
            
                        k = np.random.choice(1, 1) # 需要优化 随机选择一个模型预测 制作mask
                        pick_model_pred= eval("logits_u_tea_"+str(int(k+1))) # eval("logits_u_tea_"+str(int(k+1)))
                        softmax_pred = torch.softmax(pick_model_pred, dim=1) # logit to softmax float32
                        max_probs, argmax_pred = torch.max(softmax_pred, dim=1) #  each pixel max probs value\class int64
                    
                        for i in range(num):
                            img = unsup_imgs[i]       
                            pred = argmax_pred[i] # int64 [512,512]
                            if i == 0:
                                mask = transformsgpu.classmix_mask(pred)
                            else:
                                mask = torch.cat((mask, transformsgpu.classmix_mask(pred))) # cat must be tuple 
                        
                        unsup_imgs_mixed, _ = transformsgpu.classmix_mix(mask.float(), data=unsup_imgs) # in classmix  target=none
                        logit_cons_tea_1, _ = transformsgpu.classmix_mix(mask.float(), data=logits_u_tea_1) 
                        logit_cons_tea_2, _ = transformsgpu.classmix_mix(mask.float(), data=logits_u_tea_2)

                    elif args.mix == 'cutmix':
                        
                        import mask_gen
                        mask_param = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                            random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                            prop_by_area=not config.cutmix_boxmask_by_size, within_bounds=not config.cutmix_boxmask_outside_bounds,
                                            invert=not config.cutmix_boxmask_no_invert)
                        mask = mask_param.generate_params(num, unsup_imgs.shape[2:4]).squeeze(1).astype(np.float32)  # [4,1,512,512] np.float64 -> [4,512,512] np.float32
                        mask = torch.as_tensor(mask).cuda(non_blocking=True) # require_grad=false
                        unsup_imgs_mixed, _ = transformsgpu.mix( mask, data=unsup_imgs ) # 
                        logit_cons_tea_1, _ = transformsgpu.mix( mask, data=logits_u_tea_1 ) 
                        logit_cons_tea_2, _ = transformsgpu.mix( mask, data=logits_u_tea_2 )
                            
                    elif args.mix == 'cutout':
                    
                        import mask_gen
                        mask_param = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                            random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                            prop_by_area=not config.cutmix_boxmask_by_size, within_bounds=not config.cutmix_boxmask_outside_bounds,
                                            invert=not config.cutmix_boxmask_no_invert)
                        mask = mask_param.generate_params(num, unsup_imgs.shape[2:4]).astype(np.float32) # [4,1,512,512]
                        mask_np = np.squeeze(mask, axis=1)
                        mask = torch.as_tensor(mask).cuda(non_blocking=True) # 存储为图片时 需要squeeze 因为多了一个通道
                        unsup_imgs_mixed = (1-mask) * unsup_imgs   ## 
                        logit_cons_tea_1 = (1-mask) * logits_u_tea_1
                        logit_cons_tea_2 = (1-mask) * logits_u_tea_2
                        
                    elif args.mix == 'mixup': # gridmask hide and seek
                        pass
                    elif args.mix == 'ricap': # 过度正则化 导致欠拟合 需要调整学习率
                        
                        I_x, I_y = unsup_imgs.shape[2:]
                        beta = 0.3
                        for k in range(num):
                            cat_x = int(np.round(I_x * np.random.beta(0.3, 0.3)))# 23 beta参数设置为0.3 0.3
                            cat_y = int(np.round(I_y * np.random.beta(0.3, 0.3)))# 195
                            if k==0:
                                cat_position_x = [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]]
                                cat_position_y = [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]]
                            else:
                                cat_position_x = np.append( cat_position_x, [[ cat_x, I_x-cat_x, cat_x, I_x-cat_x ]], axis=0 )
                                cat_position_y = np.append( cat_position_y, [[ cat_y, cat_y, I_y-cat_y, I_y-cat_y ]] ,axis=0 )

                        unsup_imgs_mixed, logit_cons_tea_1, logit_cons_tea_2 = transformsgpu.ricap_mix( cat_position_x, cat_position_y, data = unsup_imgs, label1=logits_u_tea_1, label2=logits_u_tea_2 )
                                                
                    elif args.mix == 'mosaic':
                        
                        unsup_imgs_mixed, logit_cons_tea_1, logit_cons_tea_2 = transformsgpu.mosaic_mix( data = unsup_imgs, label1=logits_u_tea_1, label2=logits_u_tea_2 )
                    
                    elif args.mix == 'my_mosaic':
                        
                        unsup_imgs_mixed, logit_cons_tea_1, logit_cons_tea_2 = transformsgpu.my_mosaic_mix( data = unsup_imgs, label1=logits_u_tea_1, label2=logits_u_tea_2 )
                        
                    elif args.mix == 'mymix':
                        # 1拼接点正态分布 改分布 改lr 改损失条调权重 改引导信息 改信息引导方式 
                        # 2如何利用指导信息 如果前景占比大 要做什么操作 抠出难分区域也要计算损失 占比小要做什么操作 前景类分布 前景类频率众数 前景类别名字
                        # 3是否要分阶段 增强 先训练五个epoch 再class guide 
                        # 4 set omp 
                        # 5 规范代码  基尼指数 信息增益比 决策树 
                        # 6 显著性区域、注意力（通道、空间注意力） attentive cutmix 选取7x7中前6个注意力最高点 transmix 也是 在损失处 叠加注意力
                        # 7 先实现 难分区域损失计算 如果batch的 熵之间差别不大 正常增强
                        # 如果差较大 则 以高熵区域为中心 画出固定大小区域 并储存下来 与别的区域进行比较算loss
                        '''class SpatialAttention(nn.Module):
                            def __init__(self, kernel_size=7):
                                super(SpatialAttention, self).__init__()
                                assert kernel_size in (3,7), "kernel size must be 3 or 7"
                                padding = 3 if kernel_size == 7 else 1

                                self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
                                self.sigmoid = nn.Sigmoid()

                            def forward(self, x):
                                avgout = torch.mean(x, dim=1, keepdim=True)
                                maxout, _ = torch.max(x, dim=1, keepdim=True)
                                x = torch.cat([avgout, maxout], dim=1)
                                x = self.conv(x)
                                return self.sigmoid(x)
                        '''
                        
                        _, ps_label_1 = torch.max(logits_u_tea_1, dim=1)
                        ps_label_1 = ps_label_1.long()
                        _, ps_label_2 = torch.max(logits_u_tea_2, dim=1)
                        ps_label_2 = ps_label_2.long() # [4,512,512]             
                                                                        
                        pick = np.random.choice(1, 1) # 需要优化 随机选择一个模型预测 制作mask
                        pick_model_pred= eval("logits_u_tea_"+str(int(pick+1))) # eval("logits_u_tea_"+str(int(k+1)))
                        softmax_pred = torch.softmax(pick_model_pred, dim=1) # logit to softmax float32 [4,21,512,512]
                        entropy = -torch.sum(softmax_pred * torch.log(softmax_pred + 1e-10), dim=1) # [4,512,512]                       
                        
                        enresults = open(entropy_file_path, 'a')
                        hdresults = open(hard_sample_file_path, 'a')
                        if idx==0: # save entropy
                            enresults.write('epoch:{}, start \n'.format(epoch))
                            hdresults.write('epoch:{}, start \n'.format(epoch))
                        enresults.write("batch:{}, 中位数 batch_entropy:{}, Q3 :{} \n".format(idx,torch.quantile( entropy,q=0.5),torch.quantile( entropy,q=0.75)))
                        
                        if epoch==0:
                            hardest_ratio = 0.1
                            for hard_batch_id in range(num):
                                
                                high_entropy_sum =torch.sum(entropy[hard_batch_id]>1.0)
                                high_entropy_ratio = high_entropy_sum/( entropy.shape[1]*entropy.shape[2])
                                
                                if high_entropy_ratio>0.1:
                                    logger.info("hard sample detected!, id:{} ".format( unsup_minibatch["fn"][hard_batch_id]))
                                    hdresults.write('hard_sample: ' + str(unsup_minibatch["fn"][hard_batch_id]) + ',中位数: ' + str(torch.quantile( entropy[hard_batch_id],q=0.5)) +'\n')
                                    enresults.flush()
                                if hardest_ratio<high_entropy_ratio: # get hard_id
                                    hard_id= hard_batch_id
                                    hardest_ratio = high_entropy_ratio
                        hdresults.close()
                        enresults.close()
                        I_x, I_y = unsup_imgs.shape[2:]   
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
                        I_x, I_y = unsup_imgs.shape[2:]
                        beta = 0.3
                                                            
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
                                
                                img = unsup_imgs[start]   # 取起始图片
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
                                ## 不需要pad with 255，这里黑边就是背景 pred是经过通道argmax出来的
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
                                _, row=torch.min(mask,axis=1)
                                _, col=torch.min(mask,axis=0)
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
                              
                    elif args.mix == 'supermix':
                        pass
                    # save AugImage
                    #save_augimage(unsup_imgs, config.visual_img_dir, imtype= args.mix + '_img') # [4,3,512,512]
                    #save_augimage(mask_np, config.visual_img_dir, imtype = args.mix + '_mask') # [4,512,512]
                    #save_augimage(unsup_imgs_mixed, config.visual_img_dir, imtype= args.mix + '_mixed_img')

                # loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])
                # acc = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])
                # Mix teacher predictions using same mask
                # It makes no difference whether we do this with logits or probabilities as
                # the mask pixels are either 1 or 0
                
                #_, ps_label_1 = torch.max(logit_cons_tea_1, dim=1)
                #ps_label_1 = ps_label_1.long()
                #_, ps_label_2 = torch.max(logit_cons_tea_2, dim=1)
                #ps_label_2 = ps_label_2.long()

                # Get student#1 prediction for mixed image
                _, logits_cons_stu_1 = model(unsup_imgs_mixed, step=1)
                # Get student#2 prediction for mixed image
                _, logits_cons_stu_2 = model(unsup_imgs_mixed, step=2)
                if hard_id is not None:
                    pick_model_stu_pred= eval("logits_cons_stu_"+str(int(pick+1)))
                    pick_model_label=eval("hard_label"+str(int(pick+1))).unsqueeze(0)
                    
                    area1 = pick_model_stu_pred[aug_pos1][ :, 0:cat_y, 0:cat_x ].unsqueeze(0)
                    area2 = pick_model_stu_pred[aug_pos2][ :, I_y-cat_y:I_y, I_x-cat_x:I_x ].unsqueeze(0)
                    area3 = pick_model_stu_pred[aug_pos3][ :, 0:cat_y, I_x-cat_x:I_x ].unsqueeze(0)
                    area_loss = criterion(area1, pick_model_label)+criterion(area2, pick_model_label)+criterion(area3, pick_model_label)# 用其中一个的预测是否合理
                    dist.all_reduce(area_loss, dist.ReduceOp.SUM)
                    area_loss = area_loss / engine.world_size
                else: area_loss=torch.tensor(0) 
                   
                cps_loss = criterion(logits_cons_stu_1, ps_label_2) + criterion(logits_cons_stu_2, ps_label_1) 
                dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
                cps_loss = cps_loss / engine.world_size
                cps_loss = cps_loss * config.cps_weight

            
            # supervised loss on both models
            print(torch.cuda.memory_summary())
            torch.cuda.empty_cache() #### 
            print(torch.cuda.memory_summary())
            
            if args.mix is None:
                sup_pred_l = pred_sup_l
                sup_pred_r = pred_sup_r
            else:
                _, sup_pred_l = model(imgs, step=1)
                _, sup_pred_r = model(imgs, step=2)

            loss_sup = criterion(sup_pred_l, gts)
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size

            loss_sup_r = criterion(sup_pred_r, gts)
            dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
            loss_sup_r = loss_sup_r / engine.world_size
            
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            # print(len(optimizer.param_groups))
            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr
            optimizer_r.param_groups[0]['lr'] = lr
            optimizer_r.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_r.param_groups)):
                optimizer_r.param_groups[i]['lr'] = lr

            loss = loss_sup + loss_sup_r + cps_loss + area_loss
            loss.backward()
            optimizer_l.step()
            optimizer_r.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_cps=%.4f' % cps_loss.item() \
                        + ' loss_area=%.4f' % area_loss.item()

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_cps += cps_loss.item()
            sum_area += area_loss.item()
            pbar.set_description(print_str, refresh=False)
            logger.info( print_str )
            end_time = time.time()
        
        if engine.distributed and (engine.local_rank == 0):
            tb_logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
            tb_logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
            tb_logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)
            tb_logger.add_scalar('train_loss_area', sum_area / len(pbar), epoch)

        if (epoch > config.nepochs // 6) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
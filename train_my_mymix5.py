from __future__ import division
from ast import Pass, Return
import os.path as osp
import os
from pickletools import uint8
import sys
import time
import argparse
import math
import matplotlib

#from sqlalchemy import false
from tqdm import tqdm
import numpy as np
from PIL import Image
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
from utils.visualize import save_augimage,tensor2im,pred2im
from engine.cam import gen_cam

from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import ensure_dir, get_logger
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
#from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter
import gc 

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
    criterion_pseudo = nn.CrossEntropyLoss(reduction='none', ignore_index=255)

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
    
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
            unsupervised_train_sampler.set_epoch(epoch) #  set random seed
        gc.collect()  # 清理内存
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
            #print("num_threads: %d" % torch.get_num_threads())
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
                area_loss=torch.tensor(0).float().cuda() 
            # unsupervised loss on model/branch#1
            else: 
                with torch.no_grad(): ###
                    # Estimate the pseudo-label with branch#1 & supervise branch#2
                    v3plus, logits_u_tea_1 = model(unsup_imgs, step=1) # [4,21,512,512]
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
                        _, ps_label_1 = torch.max(logit_cons_tea_1, dim=1)
                        ps_label_1 = ps_label_1.long()
                        _, ps_label_2 = torch.max(logit_cons_tea_2, dim=1)
                        ps_label_2 = ps_label_2.long()
                            
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
                        
                        # mosaic = kornia.RandomMosaic((512, 512), data_keys=["input", "bbox_xyxy"])
                        # boxes = torch.tensor([[
                        #     [70, 5, 150, 100],
                        #      [60, 180, 175, 220],
                        #     ]]).repeat(8, 1, 1)
                        # input = torch.randn(8, 3, 224, 224)
                        # out = mosaic(input, boxes)
                        # out[0].shape, out[1].shape
                        # (torch.Size([8, 3, 300, 300]), torch.Size([8, 8, 4]))
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
                        # entropy = -torch.sum(softmax_pred * torch.log(softmax_pred + 1e-10), dim=1) # [4,512,512]                       
                        
                        # enresults = open(entropy_file_path, 'a')
                        # hdresults = open(hard_sample_file_path, 'a')
                        # if idx==0: # save entropy
                        #     enresults.write('epoch:{}, start \n'.format(epoch))
                        #     hdresults.write('epoch:{}, start \n'.format(epoch))
                        # enresults.write("batch:{}, 中位数 batch_entropy:{}, Q3 :{} \n".format(idx,torch.quantile( entropy,q=0.5),torch.quantile( entropy,q=0.75)))
                        
                        # hardest_ratio = 0.1
                        # for hard_batch_id in range(num):
                        #     high_entropy_sum =torch.sum(entropy[hard_batch_id]>1.0)
                        #     high_entropy_ratio = high_entropy_sum/( entropy.shape[1]*entropy.shape[2])
                            
                        #     if high_entropy_ratio>0.1:
                        #         logger.info("hard sample detected!, id:{} ".format( unsup_minibatch["fn"][hard_batch_id]))
                        #         hdresults.write('hard_sample: ' + str(unsup_minibatch["fn"][hard_batch_id]) + ',中位数: ' + str(torch.quantile( entropy[hard_batch_id],q=0.5)) +'\n')
                        #         enresults.flush()
                        #     if epoch>40 and hardest_ratio<high_entropy_ratio: # get hard_id
                        #         hard_id= hard_batch_id
                        #         hardest_ratio = high_entropy_ratio
                                    
                        # del pick_model_pred,entropy,
                        # hdresults.close()
                        # enresults.close()
                        if hard_id is not None:
                            unsup_imgs_mixed, ps_label_1, ps_label_2, aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y  \
                            =transformsgpu.my_mix( unsup_imgs, softmax_pred, ps_label_1, ps_label_2, hard_id=hard_id)
                        else: 
                            # unsup_imgs_mixed, ps_label_1, ps_label_2 = transformsgpu.my_mix( unsup_imgs, \
                            # softmax_pred, ps_label_1, ps_label_2, hard_id=hard_id) 
                            unsup_imgs_mixed, ps_logit_1, ps_logit_2 = transformsgpu.my_mix_logit( unsup_imgs, \
                            softmax_pred, logits_u_tea_1, logits_u_tea_2, beta=0.3, hard_id=hard_id) 
                        
                        #imgs, gts = transformsgpu.my_mix_sup(imgs,gts)# sup mymix 对有标签数据进行混合增强 效果非常差
                        
                        ####### pslabel entropy fliter    
                        def pseudo_weight(pse_outputs, a=0.9, k=0.2):# a 0.5 0.6 0.4 0.3 0.7 k=0.2
                            #print('----------------',a,k)
                            pse_outputs = F.softmax(pse_outputs, dim=1)
                            pseudo_label = torch.max(pse_outputs, dim=1)[1].long()
                            uncertainty = -1.0 * torch.sum(pse_outputs * torch.log(pse_outputs + 1e-8), dim=1)
                            unc_flatten = torch.flatten(uncertainty, 1)
                            max_unc = torch.max(unc_flatten, dim=1)[0].reshape(-1, 1, 1)
                            min_unc = torch.min(unc_flatten, dim=1)[0].reshape(-1, 1, 1)
                            unc = (uncertainty - min_unc) / (max_unc - min_unc)
                            one = torch.ones_like(unc)
                            #unc_weight = torch.where(unc_weight >= a, one, unc_weight * 1 / a)
                            unc_weight = torch.where(unc <= a, one, (1-unc)/(1-a + 1e-8)) # 熵低于a的 权重为一 其余像素的权重线性递减
                            k_ones = k*one
                            unc_weight = torch.where(unc_weight <= k, k_ones, unc_weight)# 像素的权重小于0.1的 提升到0.1 避免 丢失了最混乱像素的贡献 对应的b 为1-k（1-a）                           
                            #unc_weight = torch.where(unc_weight >= a, one, unc_weight)
                            ### 不过滤 只提升hard class 过滤有些许效果
                            return pseudo_label, unc_weight
                                            
                        # class weight
                        def voc_class_weight(pl,weight, l=1):    
                            # 为难类且weight大于0.6的可以升级为1
                            #a = 0.55 # 0.6= 2 * 0.3 0.3+a/2
                            one = torch.ones_like(weight)
                            zero = torch.zeros_like(weight)
                            #chairhigh_weight = torch.where(weight >= 0.4, one, weight) #
                            #higher_weight = torch.where(weight >= a, one, weight)
                            bicycle_weight = torch.where(pl==2 , one, zero)  
                            bottle_weight = torch.where(pl==5, one, zero)
                            chair_weight = torch.where(pl==9, one, zero) 
                            diningtable_weight = torch.where(pl==11, one, zero)
                            pottedplant_weight = torch.where(pl==16, one, zero) 
                            class_weight = bicycle_weight + bottle_weight + chair_weight + diningtable_weight + pottedplant_weight
                            hardclass_weight = torch.where(class_weight>0, class_weight, weight)# one to weight 
                            return hardclass_weight
                        
                        def voc_class_weight_1(pl,weight):    
                            # 为难类且weight大于0.6的可以升级为1
                            a = 0.3 # 0.6= 2 * 0.3
                            one = torch.ones_like(weight)
                            zero = torch.zeros_like(weight)
                            bicycle_weight = torch.where(pl==3 , one*2, zero)  
                            chair_weight = torch.where(pl==10, one*4, zero) 
                            sofa_weight = torch.where(pl==19, one*2, zero) 
                            class_weight = bicycle_weight + chair_weight + sofa_weight
                            hardclass_weight = torch.where(class_weight>0, class_weight, one) 
                            return hardclass_weight
                        
                        ps_label_1, weight_1 = pseudo_weight(ps_logit_1)
                        # low_weight_1_sum = torch.sum(weight_1)
                        weight_1 = voc_class_weight(ps_label_1, weight_1)
                        #weight_1_sum = torch.sum(weight_1)
                        ps_label_2, weight_2 = pseudo_weight(ps_logit_2)
                        weight_2 = voc_class_weight(ps_label_2, weight_2)
                        #weight_2_sum = torch.sum(weight_2)
                        # weight_1 = torch.ones_like(ps_label_1)
                        # weight_2 = torch.ones_like(ps_label_2)
                        
                    elif args.mix == 'mymix_halfclass':

                        _, ps_label_1 = torch.max(logits_u_tea_1, dim=1)
                        ps_label_1 = ps_label_1.long()
                        _, ps_label_2 = torch.max(logits_u_tea_2, dim=1)
                        ps_label_2 = ps_label_2.long() # [4,512,512]             
                                                                        
                        pick = np.random.choice(1, 1) # 需要优化 随机选择一个模型预测 制作mask
                        pick_model_pred= eval("logits_u_tea_"+str(int(pick+1))) # eval("logits_u_tea_"+str(int(k+1)))
                        softmax_pred = torch.softmax(pick_model_pred, dim=1) # logit to softmax float32 [4,21,512,512]

                        if hard_id is not None:
                            unsup_imgs_mixed, ps_label_1, ps_label_2, aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y  \
                            =transformsgpu.my_mix_halfclass( unsup_imgs, softmax_pred, ps_label_1, ps_label_2, hard_id=hard_id)
                        else: 
                            unsup_imgs_mixed, ps_label_1, ps_label_2 =transformsgpu.my_mix_halfclass( unsup_imgs, \
                            softmax_pred, ps_label_1, ps_label_2, hard_id=hard_id) 
                    
                    elif args.mix == 'mymix_saliency':
                        
                        _, ps_label_1 = torch.max(logits_u_tea_1, dim=1)
                        ps_label_1 = ps_label_1.long()
                        _, ps_label_2 = torch.max(logits_u_tea_2, dim=1)
                        ps_label_2 = ps_label_2.long() # [4,512,512]             
                                                                        
                        pick = np.random.choice(1, 1) # 需要优化 随机选择一个模型预测 制作mask
                        pick_model_pred= eval("logits_u_tea_"+str(int(pick+1))) # eval("logits_u_tea_"+str(int(k+1)))
                        softmax_pred = torch.softmax(pick_model_pred, dim=1) # logit to softmax float32 [4,21,512,512]
 
                        if hard_id is not None:
                            unsup_imgs_mixed, ps_label_1, ps_label_2, saliency_maps, aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y  \
                            =transformsgpu.mymix_saliency( unsup_imgs, ps_label_1, ps_label_2, hard_id=hard_id)
                        else: 
                            unsup_imgs_mixed, ps_label_1, ps_label_2 , saliency_maps=transformsgpu.mymix_saliency( unsup_imgs, \
                             ps_label_1, ps_label_2, hard_id=hard_id) 
                            
                    elif args.mix == 'mymix_attentive':
                        _, ps_label_1 = torch.max(logits_u_tea_1, dim=1)
                        ps_label_1 = ps_label_1.long()
                        _, ps_label_2 = torch.max(logits_u_tea_2, dim=1)
                        ps_label_2 = ps_label_2.long() # [4,512,512]             
                                                                        
                        if hard_id is not None:
                            unsup_imgs_mixed, ps_label_1, ps_label_2, aug_pos1,aug_pos2,aug_pos3, cat_x, cat_y  \
                            =transformsgpu.mymix_attentive( unsup_imgs, v3plus, ps_label_1, ps_label_2, hard_id=hard_id)
                        else: 
                            unsup_imgs_mixed, ps_label_1, ps_label_2 =transformsgpu.mymix_attentive( unsup_imgs, \
                             v3plus, ps_label_1, ps_label_2, hard_id=hard_id)   
                               
                    
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
                    #if pick==0:
                    #    pick2=2
                    #else: pick2=1
                    #pick_model_label=eval("hard_label"+str(int(pick2))).unsqueeze(0)# 方案二比较对应label 效果不好
                    #pick_model_label=eval("logits_cons_stu_"+str(int(pick2)))# 方案三 用预测后的logit再算一个label
                    #hard_label = pick_model_label[hard_id][:, 0:cat_y, 0:cat_x ].unsqueeze(0)
                    #_, hard_label = torch.max( hard_label, dim=1 )
                    
                    hard_label = pick_model_stu_pred[hard_id][ :, 0:cat_y, 0:cat_x ].unsqueeze(0)
                    _, hard_label = torch.max( hard_label,dim=1 )
                    
                    area1 = pick_model_stu_pred[aug_pos1][ :, 0:cat_y, 0:cat_x ].unsqueeze(0)
                    area2 = pick_model_stu_pred[aug_pos2][ :, I_y-cat_y:I_y, I_x-cat_x:I_x ].unsqueeze(0)
                    area3 = pick_model_stu_pred[aug_pos3][ :, 0:cat_y, I_x-cat_x:I_x ].unsqueeze(0)
                    area_loss = criterion(area1, hard_label)+criterion(area2, hard_label)+criterion(area3, hard_label)# 用其中一个的预测是否合理                   
                    
                    #dist.all_reduce(area_loss, dist.ReduceOp.SUM) # 放在此处就会wait
                    #area_loss = area_loss / engine.world_size
                    del area1,area2,area3, pick_model_stu_pred
                else: 
                    area_loss=torch.tensor(0).float().cuda() # 
                dist.all_reduce(area_loss, dist.ReduceOp.SUM) # 多卡必须都有该代码
                area_loss = area_loss / engine.world_size
                area_loss = area_loss * 0.001 # 改权重 0.5 1 0.25
                #torch.cuda.empty_cache()
                
                # cps_loss = torch.div(torch.sum(criterion_pseudo(logits_cons_stu_1, ps_label_2)*weight_2), weight_2_sum) + \
                #            torch.div(torch.sum(criterion_pseudo(logits_cons_stu_2, ps_label_1)*weight_1), weight_1_sum) #+area_loss 
                cps_loss = torch.mean( criterion_pseudo(logits_cons_stu_1, ps_label_2)*weight_2 ) + \
                           torch.mean( criterion_pseudo(logits_cons_stu_2, ps_label_1)*weight_1 ) #+area_loss  还可以加 sce loss
                #cps_loss = criterion(logits_cons_stu_1, ps_label_2) + criterion(logits_cons_stu_2, ps_label_1) #+area_loss 
                ## 查看sum是否相同 直接加torch.mean()
                dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
                cps_loss = cps_loss / engine.world_size
                cps_loss = cps_loss * config.cps_weight
            
            # supervised loss on both models
            #print(torch.cuda.memory_summary())
            #torch.cuda.empty_cache() #### 
            #print(torch.cuda.memory_summary())
            #print("num_threads: %d" % torch.get_num_threads())
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

            loss = loss_sup + loss_sup_r + cps_loss  + area_loss
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
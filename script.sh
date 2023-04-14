#!/usr/bin/env bash
nvidia-smi
export CUDA_VISIBLE_DEVICES="4,5 "
export volna="/media/ders/GDH/TorchSemiSeg/"
export NGPUS=2 ### bash 中不要有空格
export OUTPUT_PATH="/media/ders/GDH/TorchSemiSeg/exp.voc/voc8.res50v3+.CPS+CutMix/output/mymix_23.2.22_only_pad255_beta0.4"
# 2 1 0.3
export snapshot_dir=$OUTPUT_PATH

export batch_size=8 # 8
export learning_rate=0.0025 #0.0025
export snapshot_iter=1
#
python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=28010 train_my_mymix7_beta.py  --mix mymix
#-c /media/ders/GDH/TorchSemiSeg/exp.voc/voc8.res50v3+.CPS+CutMix/output/mymix1_beta3/snapshot/epoch-19.pth --port 15010
export TARGET_DEVICE=$[$NGPUS-1]
python eval_more.py -e 22-34 -d 0 -one #-$TARGET_DEVICE  #--save_path $OUTPUT_PATH/results   #
# following is the command for debug
# export NGPUS=1
# export batch_size=2
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1
# cd /media/ders/GDH/TorchSemiSeg/exp.voc/voc8.res50v3+.CPS+CutMix/
# conda activate gdh
# nohup bash script.sh > mymix_23.2.22_only_pad255_beta0.2.out 2>&1 &   
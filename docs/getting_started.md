## Getting Started

### Data Preparation 
##### Download the data (VOC, Cityscapes) and pre-trained models from  [OneDrive link](https://pkueducn-my.sharepoint.com/:f:/g/personal/pkucxk_pku_edu_cn/EtjNKU0oVMhPkOKf9HTPlVsBIHYbACel6LSvcUeP4MXWVg?e=139icd): 

```
DATA/
|-- city
|-- pascal_voc
|-- pytorch-weight
|   |-- resnet50_v1c.pth
|   |-- resnet101_v1c.pth
```


### Training && Inference on PASCAL VOC:

```shell
$ cd pwd
$ bash script.sh
```

### Different Partitions
To try other data partitions beside 1/8, you just need to change two variables in `config.py`:
```python
C.labeled_ratio = 8
C.nepochs = 34
```


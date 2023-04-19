# A Lightest CNN Network Architecture

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- thop
- tensorboardX

## Training
```
# Start training with: 
python train.py

# Start testing with: 
python test.py
```

## Experimental setting：
```
Epoch = 200  
Batchsize = 128  
Learning rate = 0.1  
momentum = 0.9  
weight_decay = 4e-5
```

## Loss and error curves：

![](data/loss.png)



## Accuracy
| Model          |       | Params(M) | FLOPs(M)  | Acc(%) |
| -------------- | ---   | -------   | ----      | ----   |
| ShuffleNet V1  | G=2   |  2.05     | 90.34     | 90.28  |
| ShuffleNet V2  | 1.0x  |  1.26     | 46.13     | 92.25  |
| MobileNet V1   | 1.0x  |  3.22     | 178.06    | 93.15  |
| MobileNet V2   | 1.0x  |  2.30     | 94.60     | 93.90  |
| MobileNet V3   | Small |  1.68     | 18.49     | 91.64  |
| MobileNet V3   | Large |  4.21     | 68.67     | 94.03  |
| GhostNet       | R=3   |  3.63     | 35.20     | 91.13  |
| GatherNet      | 1.0x  |  0.24     | 21.01     | 90.60  |

## ------------------------------------------
## In GatherNet/dataset/

There Are Ocular Surface Disease Images Dataset (OSD Dataset).

Including 467 normal and 486 abnormal images showing 12 types of ocular surface diseases.
trainset:testset=7:3 (667:286)

Use this OSD dataset, please cite the article: 

[1]Chen R,Zeng W,Fan W,et al.Automatic Recognition of Ocular Surface Diseases on Smartphone Images 
Using Densely Connected Convolutional Networks[C].2021 43rd Annual International Conference of the 
IEEE Engineering in Medicine & Biology Society (EMBC).IEEE,2021:2786-2789.

## OSD Experiments
The lightest CNN network was played on the ocular surface disease dataset.

## OSD Accuracy
| Model          |       | Params(M) | FLOPs(B)  | Acc(%) |
| -------------- | ---   | -------   | ----      | ----   |
| ShuffleNet V2  | 1.0x  |  1.26     | 1.04      | 83.22  |
| MobileNet V1   | 1.0x  |  3.21     | 4.06      | 83.92  |
| MobileNet V2   | 1.0x  |  2.23     | 2.20      | 87.41  |
| GhostNet       | R=2   |  3.62     | 0.82      | 86.71  |
| GatherNet      | 1.0x  |  0.24     | 1.88      | 90.56  |

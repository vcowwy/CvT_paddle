### 目前进展
目前在准备进行反向对齐了。
注：没有使用分布式。

# 引言
这是 [CvT: Introducing Convolutions to Vision Transformers CVT: 将卷积引入Tansformer](https://arxiv.org/abs/2103.15808) 的正式实现.
我们提出了一种新的结构，称为卷积视觉Transformer(CVT),它通过在VIT中引入卷积来实现两种设计的最佳效果，从而提高了视觉转换器VIT的性能和效率。这是通过两个主要修改来实现的：
包含新卷积token嵌入的转换器层次结构，以及利用卷积映射的卷积Tansformer块。这些变化将卷积神经网络CNN的理想特性引入VIT体系结构（例如平移、缩放和失真不变性），同时保持转换器的优点（例如动态注意、全局上下文和更好的泛化）。
我们通过大量实验验证了CVT，结果表明，与ImageNet-1K上的其它视觉转换器和RESNET相比，该方法实现了最先进的性能，具有更少的参数和更低的触发器。
此外，在对较大的数据集（比如ImageNet-22K）进行预训练并对下游任务进行微调时，可以保持性能提升。
我们的CVT-W24经过ImageNet-22k的预训练，在ImageNet-1k数据集上获得了87.7%的顶级精度。
最后，我们的结果表明，位置编码，一个在现有的视觉转换器的关键组成部分，可以在我们的模型中安全地删除，简化了高分辨率视觉任务的设计

![](figures/pipeline.svg)

# 主要结果
## 在 ImageNet-1k 上预训练的模型
| 模型  | 分辨率 | 参数 | GFLOPs | Top-1 |
|--------|------------|-------|--------|-------|
| CvT-13 | 224x224    | 20M   | 4.5    | 81.6  |
| CvT-21 | 224x224    | 32M   | 7.1    | 82.5  |
| CvT-13 | 384x384    | 20M   | 16.3   | 83.0  |
| CvT-32 | 384x384    | 32M   | 24.9   | 83.3  |

## 在 ImageNet-22k 上预训练的模型
| 模型   | 分辨率 | 参数 | GFLOPs | Top-1 |
|---------|------------|-------|--------|-------|
| CvT-13  | 384x384    | 20M   | 16.3   | 83.3  |
| CvT-32  | 384x384    | 32M   | 24.9   | 84.9  |
| CvT-W24 | 384x384    | 277M  | 193.2  | 87.6  |

你可以从我们的 [model zoo](https://1drv.ms/u/s!AhIXJn_J-blW9RzF3rMW7SsLHa8h?e=blQ0Al) 上下载所有模型.


# 快速启动
## 安装
假设您已安装Pytorch和torchvision, 如果未安装，请按照 [officiall instruction](https://pytorch.org/) 首先安装它们. 
使用cmd安装所有依赖项:

``` sh
python -m pip install -r requirements.txt --user -q
```

此代码是使用 pytorch 1.7.1 开发和测试的， pytorch 的其它版本没有经过全面测试.

## 数据准备
请准备以下数据:

``` sh
|-DATASET
  |-imagenet
    |-train
    | |-class1
    | | |-img1.jpg
    | | |-img2.jpg
    | | |-...
    | |-class2
    | | |-img3.jpg
    | | |-...
    | |-class3
    | | |-img4.jpg
    | | |-...
    | |-...
    |-val
      |-class1
      | |-img5.jpg
      | |-...
      |-class2
      | |-img6.jpg
      | |-...
      |-class3
      | |-img7.jpg
      | |-...
      |-...
```


## 运行
每个实验都由 yaml 配置文件定义， 该文件保存在 `experiments` 目录下.  `experiments` 目录具有如下结构:

``` sh
experiments
|-{DATASET_A}
| |-{ARCH_A}
| |-{ARCH_B}
|-{DATASET_B}
| |-{ARCH_A}
| |-{ARCH_B}
|-{DATASET_C}
| |-{ARCH_A}
| |-{ARCH_B}
|-...
```

我们提供了一个 `run.sh` 脚本，用于在本地机器上运行作业。

``` sh
Usage: run.sh [run_options]
Options:
  -g|--gpus <1> - number of gpus to be used
  -t|--job-type <aml> - job type (train|test)
  -p|--port <9000> - master port
  -i|--install-deps - If install dependencies (default: False)
```

### 本地机器上训练

``` sh
bash run.sh -g 8 -t train --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml
```

你还可以通过命令行修改配置参数。例如，如果要将 lr rate 更改为0.1， 可以运行以下命令:
``` sh
bash run.sh -g 8 -t train --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml TRAIN.LR 0.1
```

Notes:
- 默认情况下，checkpoint, model, and log files 将保存在 OUTPUT/{dataset}/{training config}.

### 测试预训练模型
``` sh
bash run.sh -t test --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml TEST.MODEL_FILE ${PRETRAINED_MODLE_FILE}
```


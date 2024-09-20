# Better Synth 比赛开发套件

1. 依赖安装
- 推荐使用 conda 环境
```shell
conda create -n dj python=3.10
conda activate dj

bash install.sh
```

2. 比赛资源下载
- 下载基础模型，种子/微调/评测数据集
- 基础模型与微调数据集均存放于训练目录中指定位置
- 种子数据集存放于`input`目录
- 评测数据集存放于`toolkit/eval`目录
```shell
bash download.sh
```

3. 数据处理与合成
- 比赛要求使用 [data-juicer](https://github.com/modelscope/data-juicer) 基于上一步中下载的**种子数据集**进行数据处理与合成
- 数据处理与合成相关代码请存放于`solution`目录中，并在`solution/requirements.txt`中添加所需的依赖
- 最后也请在`solution/readme`中详细介绍所使用的数据处理与合成方案
- 处理后的数据集需按照如下结构存放：
```
📦 output/
├── 📂 processed_data/
│       ├── 📂 synthetic_images/ (if any)
│       └── 📄 processed_data.jsonl
│ ...
```
- `processed_data.jsonl`需为标准的`JSONL`格式，例如：
```json lines
{"images": ["images/00237/002375592.jpg"], "text": "<image>\nadorable pink and gray elephant themed party favour boxes with tissue fillers <|__dj__eoc|>", "id": "002375592"}
{"images": ["images/00199/001999195.jpg"], "text": "<image>\nbreccinano adult dog food for all ages with turkey, lamb and venisi <|__dj__eoc|>", "id": "001999195"}
...
```

4. 执行模型训练/推理
- 模型训练与推理
```shell
cd toolkit/

# 请根据自身需求修改训练脚本train_mgm_2b_stage_1.sh内的参数
# 您只能修改以下范围内的参数

############################################################################
########################### Editable Part Begins ###########################
############################################################################
# 可修改参数范围
############################################################################
############################ Editable Part Ends ############################
############################################################################

# 修改完毕后执行训练与推理脚本
bash train_mgm_2b_stage_1.sh
```
- 训练与推理结束后，会在`output`目录中产出训练好的模型以及评测集推理结果
  - 训练后的模型存放于：`output/training_dirs`
  - 推理结果存放于：`output/eval_results`

5. 线上赛提交结果

线上赛只需要提交solution，训练脚本，训练日志（pretrain和finetuning），以及评测推理结果 
- 请将数据处理方案、训练及推理脚本、训练日志、推理结果打包成一个 zip 文件，上传至天池平台进行评测。
- 为保证提交的规范性，务必遵循以下文件打包结构并提交以下所需的文件，请勿添加额外的顶级目录。

```text
submit.zip
├── solution
│   ├── readme                                  ########## 介绍您的算法设计和执行流程 ########## 
│   ├── requirements.txt                        ########## 第三方 pip 依赖库 ########## 
│   └── ...
└── output
    ├── train.sh                                ########## 训练脚本 ##########
    ├── training_dirs
    │   ├── MGM-2B-Pretrain-xxxx                ########## 预训练好的模型 ##########
    │   │   └── pretrain.log                    ########## 预训练日志 ##########
    │   └── MGM-2B-Finetune-xxxx								########## 微调好的模型 ########## 	
    │       └── finetune.log                    ########## 微调训练日志 ##########
    └── eval_results                            ########## 推理结果 ##########
        └── MGM-2B-Finetune-xxxx
```

6. 线下赛提交结果

- 请将数据处理方案、训练与推理脚本、处理与合成的数据、模型checkpoint、推理结果等打包为一个zip文件，上传到天池平台进行评测
```shell
zip -r submit.zip solution/ output/
```
- 为保证提交的规范性，务必遵循以下文件打包结构并提交以下所需文件，请勿添加额外的顶层目录
```text
submit.zip
├── solution
│   ├── readme                                  ########## 介绍您的算法设计和执行流程 ########## 
│   ├── requirements.txt                        ########## 第三方 pip 依赖库 ########## 
│   ├── ...
└── output
    ├── train.sh                                ########## 训练脚本 ########## 
    ├── processed_data
    │   ├── synthetic_images/                   ########## 合成的图片数据（如有） ########## 
    │   └── processed_data.jsonl                ########## 用于训练的处理与合成的数据文件 ##########
    ├── training_dirs
    │   ├── MGM-2B-Pretrain-xxxx                ########## 预训练好的模型 ##########
    │   │   ├── pretrain.log                    ########## 预训练日志 ########## 
    │   │   └── mm_projector.bin
    │   └── MGM-2B-Finetune-xxxx				########## 微调好的模型 ########## 	
    │       ├── finetune.log                    ########## 微调训练日志 ##########	  
    │       └── model-xxx.safetensors  
    └── eval_results                            ########## 推理结果 ##########
        └── MGM-2B-Finetune-xxxx
```

### 10k Baseline
10k Baseline是为了给初级选手体验和参与比赛的一个快速基线过程。与标准比赛流程不同，要体验和走通10k快速基线流程的选手们可以把标准流程中的以下步骤文件替换为带有`_10k_baseline`后缀的版本，包括：
1. 比赛资源下载时，运行脚本为：`download_10k_baseline.sh`
2. 训练模型时，需修改和运行的训练脚本为：`toolkit/train_mgm_2b_stage_1_10k_baseline.sh`
此外，套件中还提供了一个Jupyter Notebook来帮助选手快速从10k基线中熟悉比赛基本流程，并提交baseline，可参考`10k_quick_baseline.ipynb`
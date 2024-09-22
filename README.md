# Better_Synth_challenge

## 赛题解读：

Better Synth 是一项以数据为中心的挑战赛，考察如何合成与清洗图文数据以在多模态大模型上取得更优的图片理解能力。  
本次比赛基于 Mini-Gemini 模型进行训练，只关注于预训练（模态间对齐）阶段的数据合成与清洗，指令微调阶段为固定数据集。  
要求参赛者基于种子数据集进行数据合成与清洗，产出一份基于种子数据集的更高质量、更多样性的数据集，并在给定计算约束下进行训练。

### 数据约束：

- 种子数据集数据量（预训练阶段）为400k，预训练数据量约束为200k。  
- 因为训练参数是固定的，本次挑战的关键在于数据的质量，这是影响预训练效果的关键。

### 数据分析：
经过前期多次试验发现image-text similarity 和 image-text matching的指标比较关键。
![myplot092001.png](..%2Fmyplot092001.png)
![myplot090202.png](..%2Fmyplot090202.png)
![myplot092003.png](..%2Fmyplot092003.png)
![myplot092005.png](..%2Fmyplot092005.png)
### 整体方案：

1. 数据处理的策略方案主要是：清洗→合成→清洗的过程。
2. 因为一开始的出发点是图生文的想法，并没想到文生图的方案（后续可以尝试结合起来）。
   - 经过前期观察和分析，看到存在很多分辨率较差以及无意义的图片，所以利用 `image_nsfw_filter` 和 `image_aesthetics_filter` 这两个算子过滤掉大部分无意义和不会带来太多收益的图片。但是这两个算子也会“误杀”部分高质量的情况。
   - 宁缺毋滥，因为脏数据给模型训练会带来更多负收益，所以尽量保留高质量的图片，减少低质量图片。
   
   ![005522915.jpg](..%2Fimages%2Ftest2%2F005522915.jpg)![002691620.jpg](..%2Fimages%2Ftest2%2F002691620.jpg)
3. 经过过滤后，剩下的数据远少于200k，所以需要利用不够100k的数据合成更多的数据。
   - 初赛使用 Qwen-vl （复赛使用 Qwen2-VL-2B-Instruct-AWQ）通过调整 prompt 来生成数据，一张图片对应多条文本描述。目的是引入多样化的描述，帮助模型学习到更广泛的图像和文本之间的关系，使模型不仅局限于简单的图文对齐，还能理解不同风格、不同角度的描述，从而提升模型的泛化能力。

### prompt 例子：

```
"请帮忙分析这张图片，可以帮助模型提高对图像内容的理解能力，包括对象识别、场景分析和整体内容，用英文回答，简洁，切勿捏造。（可加入例子作为提示，目的是控制文本长度）\n"
"输出格式参考如下（每句话加上序号）：\n"
"1.。。。。\n"
"2.。。。。\n"
"3.。。。。\n"
```

### 数据对比：

| 文本描述                                                                                         | image_text_matching_score | image_text_similarity |
|------------------------------------------------------------------------------------------------|----------------------------|-----------------------|
| signature candle with reed diffuser in white                                                    | 0.996043086                | 0.312515765           |
| The image shows a set of home fragrance products, specifically a reed diffuser and a signature candle. | 0.977473736                | 0.3182971478          |
| The reed diffuser is contained in a glass bottle with a silver lid and a black ribbon tied around it. The bottle has a label with a purple and white design. | 0.999578059                | 0.34042412            |
| The signature candle is also in a glass container, with a similar design to the reed diffuser's packaging. It has a black ribbon tied around it as well. | 0.999629498                | 0.290428698           |
| Both items are presented in a clear plastic box, which adds to the overall aesthetic appeal of the product set. | 0.749893427                | 0.239513278           |


4. 利用 `image_text_matching_score` 和 `image_text_similarity` 过滤分数过低的数据。

## 运行：

**step 1. 依赖安装**

```shell
conda create -n dj python=3.10
conda activate dj

bash install.sh
```

**step 2. 比赛资源下载**


```shell
bash download.sh
```

**step1 准备好数据**

我们建议在这一步先做一个基本的过滤, 比如只过滤掉异常图片或者caption相关性过低的图片.
便于更精准的聚类. 直接聚类也可以.

```
cd toolkit/data-juicer
python tools/process_data.py --config ./configs/xx.yaml
```

**step2 将数据聚类为簇**

将脚本参数修改为自己的数据集路径

```shell
python solution/data_process/preprocess_data.py # 修改输入文件以适配聚类输入
bash solution/cluster/cluster.sh
```

**step3 数据合成**

由于复赛数据量较小, 我们进行了全量合成, 然后再筛选出2.5k数据double一下作为训练数据.
可以根据实际需要 先进行step4, 然后把需要合成的数据筛选出来进行合成.
注意修改里面的相关文件路径
```
python solution/data_syn/batch_infer_intern_vl.py
```

**step4 数据筛选**
经过step2后, 得到数据格式如下所示:

```
{
    "images/00012/000129775.jpg": {
        "cluster": "46",
        "distance_to_center": 122.87
    },
    "images/00092/000928445.jpg": {
        "cluster": "46",
        "distance_to_center": 149.7
    },
    ...
}
```
之后按顺序执行脚本
```shell
python solution/data_process/trans_cluster_2dict.py  # 将聚类结果转化为key为簇名,value为具体数据的字典
python solution/data_process/insert_cluster.py # 将聚类相关信息插入到原数据json中
python solution/data_process/dynamic_increase.py  # 根据聚类信息, 动态进行挑选
```

**step5 进行训练**
修改对应路径名,进行训练
```
bash toolkit/train_mgm_2b_stage_1.sh
```

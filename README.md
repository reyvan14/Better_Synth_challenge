## 赛题解读：  
  
Better Synth 是一项以数据为中心的挑战赛，考察如何合成与清洗图文数据以在多模态大模型上取得更优的图片理解能力。 本次比赛基于 Mini-Gemini 模型进行训练，只关注于预训练（模态间对齐）阶段的数据合成与清洗，指令微调阶段为固定数据集。 要求参赛者基于种子数据集进行数据合成与清洗，产出一份基于种子数据集的更高质量、更多样性的数据集，并在给定计算约束下进行训练。  
  
### 数据约束：  
  
- 种子数据集数据量（预训练阶段）为400k，预训练数据量约束为200k。 
- 因为训练参数是固定的，本次挑战的关键在于数据的质量，这是影响预训练效果的关键。  

## 整体方案： 
  
### 数据分析：  
经过前期多次试验发现image-text similarity 和 image-text matching的结合效果比较不错。存在两个比较大的问题是：
- 两者高得分的交集数据相对较少，所以需要合成大部分数据。
- 单纯由这两个算子筛选出来的数据会出一些干扰和波动，主要原因是存在部分低质量图片造成的。
![myplot092003](https://github.com/user-attachments/assets/fda90205-800d-4987-8902-51d92340ab27)
![myplot092005](https://github.com/user-attachments/assets/499aa6a1-1fc7-4841-b971-0938332fabc6)

 
### 数据处理：
1. 数据集处理流程是：清洗→合成→清洗。  
2. 经过观察发现数据集存在很多分辨率较差以及无意义的图片，所以利用 `image_nsfw_filter` 和 `image_aesthetics_filter` 这两个算子过滤掉无意义和质量较差的图片，过滤掉不会带来太多收益的图片。
3. 但是这两个算子也会“误杀”部分高质量的情况。采取原则是宁缺毋滥，因为脏数据给模型训练会带来更多负收益，所以尽量保留高质量的图片，减少低质量图片。（思考：可以考虑由大模型去进行筛选内容不适合的图片）
  
![000111904](https://github.com/user-attachments/assets/4178b8c6-2465-43e3-af5d-a3fdac29470c)

![001229550](https://github.com/user-attachments/assets/d6bd9f16-e5be-42d3-8fa0-84b280f1af77)

4. 经过过滤后，剩下的数据远少于200k，所以需要利用不够100k的数据合成更多的数据。一开始的出发点是图生文的想法来进行数据增强，并没想到文生图的方案。（思考：可以结合文生图进行数据增强）。
  - 初赛使用 Qwen-vl （复赛使用 Qwen2-VL-2B-Instruct-AWQ）通过调整 prompt 来生成数据，一张图片对应多条文本描述。目的是引入多样化的描述，帮助模型学习到更广泛的图像和文本之间的关系，使模型不仅局限于简单的图文对齐，还能理解不同风格、不同角度的描述，从而提升模型的泛化能力。  
  
### prompt 例子：  
  
```  
"请帮忙分析这张图片，可以帮助模型提高对图像内容的理解能力，包括对象识别、场景分析和整体内容，用英文回答，简洁（不要超过十个单词），切勿捏造。\n"  
"输出格式参考如下（每句话加上序号）：\n"  
"1.。。。。\n"  
"2.。。。。\n"  
"3.。。。。\n"  
```  
  
### 数据对比：  
  
| caption | image_text_matching_score | image_text_similarity |  
|------------------------------------------------------------------------------------------------|----------------------------|-----------------------|  
| signature candle with reed diffuser in white                                                    | 0.996043086                | 0.312515765           |  
| The image shows a fragrance diffuser set. | 0.9960151315                | 0.3336452842          |  
| It includes a bottle and reeds. | 0.9976474643                | 0.2579694986            |  
| The reeds are placed inside the bottle. | 0.9909159541                | 0.2503024936           |  
| The packaging is silver with a black ribbon. | 0.6999676824                | 0.2475108355           |
| The design suggests it's for home fragrance. | 0.8599568605                | 0.247142747           |

  
  
5. 利用 `image_text_matching_score` 和 `image_text_similarity` 组合过滤新生成的数据。

## 代码实现： 
### 环境准备：
**step 1. 依赖安装**

```shell
conda create -n dj python=3.10
conda activate dj
bash install.sh
```

**step 2. 资源下载**

```shell
bash download.sh
```

**step 3. 第一次清洗数据**

```shell
dj-process --config configs/config_test1.yaml
```


**step 4. 数据合成**

```shell
python solution/qwenvl2data.py
```

**step 5. 第二次清洗数据**

```shell
dj-process --config configs/config_test2.yaml
```


**step 6. 训练**

```shell
bash toolkit/train_mgm_2b_stage_1.sh
```

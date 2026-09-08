# Coughing Pig Identification


本项目提供论文：

**Coughing pig identification method based on multimodal fusion and hierarchical temporal modelling in group-housed environments**

对应的代码和处理后数据集。

本方法利用音频和视觉信息融合，实现群养环境下发生咳嗽行为猪只的个体识别。

## 项目目录结构

```
Coughing-pig-identification
│
├── Detection_cough
│   ├── audio_feature
│   │   └── 音频特征文件
│   │
│   ├── clips_videos
│   │   └── 裁剪后的猪只视频帧
│   │
│   └── csv
│       └── 标签文件
│
├── trainTalkNet.py
│   └── 模型训练代码
│
├── inference.py
│   └── 模型推理代码
│
├── top1视频评估.py
│   └── 视频级Top-1识别评价
│
├── 评估（昼夜）.py
│   └── 昼夜条件性能评价
│
├── talkNet.py
│   └── 网络结构定义
│
├── dataLoader.py
│   └── 数据加载程序
│
├── loss.py
│   └── 损失函数
│
├── requirement.txt
   └── Python环境依赖
```
# 模型训练

运行以下命令进行模型训练：

```bash
python trainTalkNet.py

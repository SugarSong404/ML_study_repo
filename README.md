# ML_study_repo
A repository for documenting and sharing my study process with machine learning techniques, including my routines and datasets

my python version: **3.8.20**

use `pip install -r requirements.txt` to install libs needed

# 内容说明

## 笔记

**Notes** 目录下存放学习笔记

- **机器学习的数学原理——回归问题**

  简单讲了线性回归与逻辑回归算法的推导与使用
  
- **机器学习的数学原理——聚类与决策树**

  介绍了两种常见的聚类算法以及决策树和它们的工作流程

## 代码

**Codes** 目录下存放例程

- **LinearRegression**

  线性回归，目录下有e1,e2两个测试例程

  - e1：连续五位数字的下一位预测
  - e2：空气质量分值预测

  `LinearRegression.py`是手搓的线性回归库

  支持多项式回归与正弦非线性回归

  训练时可选择全批量、随机、小批量等梯度下降方法

  有预测与评估功能，能够保存与加载模型

- **LogisticRegression**

  逻辑回归，目录下有e3,e4两个测试例程

  - e3：简单二维二分类数据
  - e4：经典鸢尾花数据集
  
  `LogisticRegression.py`是手搓的逻辑回归库
  
  支持多项式回归与正弦非线性特征
  
  训练时可选择全批量、随机、小批量等梯度下降方法
  
  训练时可以选择sigmoid或softmax激活函数
  
  有预测与评估功能，能够保存与加载模型
  
- **KMeans**
  
  k-means算法，目录下有e5,e6两个测试例程
  
  - e5：简单二维聚类数据集
  
  - e6：小狗图片图像分割
  
  `KMeans.py`是手搓的k-means库
  
  评估标准支持inertia与轮廓系数
  
  有预测功能，能够保存与加载模型
  
- **DBSCAN**

  DBSCAN算法，目录下有e7一个测试例程
  
  - e7：简单二维聚类数据集
  
  `DBSCAN.py`是手搓的DBSCAN库
  
  支持计算k_distance便于找出合适的领域半径
  
  train训练过后能将不同簇存在clusters中，离群点存在noise中
  
  能够保存与加载模型
  
- **DecisionTree**
  
  DecisionTree算法，目录下有e8,e9两个测试例程
  
  - e8：空气质量分值预测
  - e9：电影票是否购买预测
  
  `DecisionTree.py`是手搓的决策树库
  
  要事先输入数组来声明当前特征是连续还是离散
  
  可以配置选择分类决策树与回归决策树进行训练
  
  评估标准分别为精度与决定系数
  
  支持显示树形结构
  
  有预测功能，能够保存与加载模型

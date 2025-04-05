# 🧠 CIFAR-10 三层神经网络分类器

该项目实现了一个 **手动实现反向传播的三层神经网络**，用于对 CIFAR-10 数据集进行图像分类，支持训练、测试、参数调优与可视化，完全基于 NumPy。

---

## 📁 项目结构

. ├── main.py # 主程序入口 

├── model.py # 神经网络结构与前向/反向传播 

├── train.py # 模型训练与保存
 
├── test.py # 模型加载与测试 

├── hyperparam.py # 超参数搜索模块 

├── visualize.py # 权重与训练结果可视化 

├── utils.py # 数据加载与辅助函数 

├── best_model.pkl # 保存的最佳模型参数 

├── metrics.png # 损失与准确率图 

├── ./figs/weights_input_hidden.png # 第一层权重可视化图 

├── ./figs/weights_hidden_output.png # 第二层权重可视化图 

├── ./figs/output_distribution.png # 第一层权重可视化图 
── README.md # 使用说明文档


---

## 📦 环境依赖

- Python >= 3.7  
- NumPy  
- scikit-learn  
- matplotlib  

使用如下命令安装依赖（推荐使用虚拟环境）：

```bash
pip install numpy scikit-learn matplotlib
```
##📁 数据准备
请从 CIFAR-10 官网 下载 cifar-10-python.tar.gz，解压后放置到项目根目录，解压结果如下：

bash
复制
编辑
./cifar-10-batches-py/

├── data_batch_1

├── ...

├── test_batch

##🚀 如何训练模型
运行主程序即可开始训练：
```
python main.py
```
该命令将执行以下操作：

加载并预处理 CIFAR-10 数据

初始化神经网络

多轮训练

自动保存验证集准确率最优的模型到 best_model.pkl

绘制训练/验证损失与验证准确率曲线（保存为 metrics.png）

可视化第一层权重（保存为 weights_input_hidden.png）
可视化第二层权重（保存为 weights_hidden_output.png）
可视化第三层权重（保存为 output_distribution.png）
##🧪 如何测试模型
使用训练好的模型在测试集上进行分类测试：
```
python test.py
```

##🔍 参数查找
运行以下命令自动尝试不同的超参数组合并输出性能：
```
python hyperparam.py
```
你可以根据输出结果选择最优参数。

##📊 可视化结果
训练结束后会生成以下图像：

metrics.png：训练/验证 loss 曲线与验证准确率曲线

weights_input_hidden.png：第一层的权重可视化
weights_hidden_output.png：第二层的权重可视化
output_distribution：第三层的权重可视化

##模型结构概览
输入层：3072（32×32×3）

隐藏层：可配置大小（默认128）+ ReLU 激活

输出层：10 类别 + Softmax

损失函数：交叉熵损失 + L2 正则

优化器：SGD + 学习率调度

##📌 说明
该项目完全使用 NumPy 手工构建，不依赖 PyTorch、TensorFlow 等深度学习框架，适合课程作业或基础学习。欢迎扩展更多功能！
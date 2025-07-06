# 神经网络参数网格搜索与图表生成
本项目为反向传播神经网络（BPNN）训练过程中模型参数的调整提供了便捷的工具。在网格搜索（grid search）方法中，需要选定各个参数的取值范围，组成多维空间网格，并详尽地搜索这些给定值的最佳组合，包括隐藏层层数、每层神经元数、激活函数、优化器、学习率等。当输入项与输出项有多种组合需要实验时，训练过程将变得更为复杂。
本项目为以上情况提供了简单易用的训练程序，同时能够输出各个模型训练过程中的拟合优度（R²）数据，并快速生成可视化图表，便于参数优化调整。

## 模型参数调整与训练
在模型训练前，需要在 **train. py** 程序中设定数据读取范围以及神经网络模型的各个参数。训练程序默认设定为在一次执行中测试多种每层神经元数量，并将训练结果导出到同一个excel文件中。

```python 
file_path = 'example_data.xlsx' 	# 读取数据的表格路径  
input_column = [1, 3, 5, 7, 9] 		# 模型输入项所在的表格列数组  
output_column = [10] 			# 模型输出项所在的表格列数组  
  
units_min = 5 					# 训练中测试的每层最小神经元数量  
units_max = 25 					# 训练中测试的每层最大神经元数量(包含)  
units_interval = 5 				# 训练中测试的神经元间隔数量  
hidden_layer_num = 3 			# 模型中使用的隐藏层数量  
epoch_num = 1000 				# 模型的最大训练步数  
log_length = 5 					# 记录一次R²数据的模型训练步数间隔  
  
learning_rate = 0.005 				# 模型学习率  
loss_function = 'mse' 				# 模型使用的损失函数  
optimizer = optimizers.Adam(learning_rate=learning_rate) # 模型使用的优化器  
activation = 'relu' 				# 除输出层外，其余层使用的激活函数  
activation_output = activation 		# 模型输出层使用的激活函数  
scaler = MinMaxScaler() 			# 预处理时使用的数据归一化方法  
k_splits = 5 						# k-fold划分数量，也决定训练与测试集划分比例  
k_splits_on = True 				# 是否执行k-fold交叉验证多次训练
patience = 40 					# 验证集性能持续未改善时停止训练的连续记录次数  
  
log_name = f'{activation}_l{hidden_layer_num}' # 训练日志输出名称  
model_name = log_name 			# 保存的最优模型文件输出名称
```

## 训练结果图表生成
本项目为训练结果的可视化准备了折线图模板，能够方便地导出训练结果折线图，并修改图表的数值范围、名称、外观与颜色。在完成模型训练后，运行 **plot_line. py** 程序以自动生成对应图表。

![输入图片说明](/selu_l3_lr0.005_line.png)



 







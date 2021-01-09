import numpy as np 
import keras 
from keras import models
from keras import layers  
from keras.datasets import boston_housing#从数据库中载入数据集
#加载数据
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data() 
#由于这个数据集取值范围比较大，因此对其进行标准化
#将每个特征减去平均值，再除以标准差

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
#不能用测试集的平均值和标准差不然会影响预测结果
test_data -= mean
test_data /= std

#模型的构建
#当需要将一个模型多次实例化时，可以构建一个函数来创建模型

def building_model(): 
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))#这里没有激活函数，是一个线性层可以预测任意值。这里是一个标量回归，即预测单一连续值的回归
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#4折交叉验证
k = 4
num_val_samples = len(train_data) // k#每一折的数据数量
'''
num_epochs = 100
all_scores = []

for i in range(k): 
    #验证数据，第k个分区的数据
    print('processing fold #', i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_data[i*num_val_samples:(i+1)*num_val_samples]
    #剩下用于训练的数据
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[i*num_val_samples:(i+1)*num_val_samples]], axis=0)
    partial_train_targets = np.concatenate([train_data[:i*num_val_samples],train_data[i*num_val_samples:(i+1)*num_val_samples]], axis=0) 

    model = building_model()
    model.fit(partial_train_data, partial_train_targets,epochs =num_epochs, batch_size = 1, verbose = 0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose =0)#在验证数据上评估模型

    all_scores.append(val_mse)

    print(val_mse)
    print(val_mae)
    print(all_scores)
'''
#修改循环，使得保留每轮的验证分数记录
num_epochs = 100
all_mae_histories = []

for i in range(k): 
    #验证数据，第k个分区的数据
    print('processing fold #', i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_data[i*num_val_samples:(i+1)*num_val_samples]
    #剩下用于训练的数据
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[i*num_val_samples:(i+1)*num_val_samples]], axis=0)
    partial_train_targets = np.concatenate([train_data[:i*num_val_samples],train_data[i*num_val_samples:(i+1)*num_val_samples]], axis=0) 

    model = building_model()
    history = model.fit(partial_train_data, partial_train_targets,validation_data = (val_data,val_targets),epochs = num_epochs, batch_size=1, verbose=0)

    mae_history = history.history['val_mae']#书上的val_mean_absolute_error写法出现报错，应该是history里的键改了
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


import matplotlib.pyplot as plt  
'''这是epochs=500d的时候，删除前十个取值范围大的点，将每个数据点替换为前面数据点的指数移动平均值
def smooth_curve(points, factor =0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))

        else:  
            smoothed_points.append(point)
    return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:])
'''


plt.plot(range(1,len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#通过画图观察曲线走势，确定最佳的epochs，使得不会过拟合
#之后就可以使用epochs重新训练模型,并用测试集测试

model = building_model()
model.fit(train_data,train_targets,epochs=80,batch_size=16,verbose =0)

test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
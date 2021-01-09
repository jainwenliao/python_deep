
import numpy as np
from keras.datasets import reuters
#下载数据集及取前10000个出现频率最高的词汇
(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)
print(len(train_data))
print(len(test_data))

#数据向量化
def vectorize_sequences(sequences, dimension = 10000): 
    results = np.zeros((len(sequences),dimension))#创建一个零矩阵
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1 #如果序列中元素的位置和矩阵一样，则将矩阵响应位置的值变为1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#y_train = np.asarray(train_labels).astype('float32')
#y_test = np.asarray(test_labels).astype('float32')

#标签向量化直接用keras内置的categorical encoding,将其变为one-hot编码
#from keras.utils.np_utils import to_categorical

def to_one_hot(labels, dimension =46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

from keras import models
from keras import layers

#定义模型
model = models.Sequential()

model.add(layers.Dense(64,activation='relu',input_shape = (10000,)))
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(46,activation = 'softmax'))

#编译模型
model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#训练模型
#预留验证集
x_val = x_train[:1000]
partial_train_data = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_train_labels = one_hot_train_labels[1000:]

#训练模型
history = model.fit(partial_train_data,
                    partial_train_labels,
                    epochs = 5,
                    batch_size=32,
                    validation_data=(x_val,y_val))

impot matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()  

plt.show()

results = model.evaluate(x_test,one_hot_test_labels)
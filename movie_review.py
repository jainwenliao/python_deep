import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt


from keras.datasets import imdb#导入电影评价数据
from keras import models,layers,optimizers

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)#取前10000条评论

#解码为英文单词
word_index =imdb.get_word_index()
reverse_word_index = dict(
    [(value,key) for (key,value) in word_index.items()]
)
decode_review = "".join(
    [reverse_word_index.get(i-3,'?') for i in train_data[0]]
)

def vectorize_sequences(sequences, dimension =10000):
    results = np.zeros((len(sequences), dimension))#创建一个形状为(len(sequences),dimension)的零矩阵
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1 #将results[i]的指定索引设为1

    return results
#数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#模型定义
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape =(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#编译模型，选择优化方法，损失函数以及指标
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accurary'])

#留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

#训练模型
model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics = ['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 5,
                    batch_size = 64,
                    validation_data = (x_val, y_val))

#history_dict = history.history 
#print(history_dict.keys())

#绘制训练损失和验证损失
#loss_val = history_dict['loss']
#val_loss_val = history_dict['val_loss']

#epochs = range(1,len(loss_val)+1)

#plt.plot(epochs,loss_val,'bo',label='training loss')
#plt.plot(epochs,val_loss_val,'b',label='val loss')
#plt.title('training and validation loss')
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.legend()

#plt.show()

a= model.predict(x_test)
print(a)
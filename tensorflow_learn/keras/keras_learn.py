import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

print(tf.version.VERSION)
print(tf.keras.__version__)

# 定义序列模型，进行层的堆叠
model = tf.keras.Sequential()
# 添加一个64单元的全连接层到模型中
model.add(layers.Dense(64, activation='relu'))
# 添加另一个
model.add(layers.Dense(64, activation='relu'))
# 添加一个10个输出单元的softmax层
model.add(layers.Dense(10, activation='softmax'))

# # 创建有sigmoid激活函数的全连接层
# layers.Dense(64, activation='sigmoid')
# layers.Dense(64, activation=tf.sigmoid)
# # 有因子为0.01的l1\l2正则化线性层应用到核矩阵
# layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
# layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
# layers.Dense(64, kernel_initializer='orthogonal')
# layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))
model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

if __name__ == '__main__':
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))
    model.fit(data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

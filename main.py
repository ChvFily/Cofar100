"""
@author :chvfily
time: 2021-01-05 23:16
"""

import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
from tensorflow import keras

## 卷积层设计
conv_layers = [ ## 5 unit conv + max pooling
    ## unit 1
    layers.Conv2D(64,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.Conv2D(64,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
    ## unit 2
    layers.Conv2D(128,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.Conv2D(128,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
    ## unit 3
    layers.Conv2D(256,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.Conv2D(256,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
    ## unit 4
    layers.Conv2D(512,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),
    ## unit 5
    layers.Conv2D(512,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same")
]

fc_layers = [
    layers.Dense(256,activation=tf.nn.relu),
    layers.Dense(128,activation=tf.nn.relu),
    layers.Dense(100,activation=tf.nn.relu)
]

## 预处理数据集
def preprocess(x,y):
    x  = tf.cast(x,dtype=tf.float32) / 255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y

## 加载/下载 数据集
(x,y),(x_test,y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y,axis=1)  ## 挤压数据
y_test = tf.squeeze(y_test,axis=1)

db_train = tf.data.Dataset.from_tensor_slices((x,y))
db_train = db_train.shuffle(10000).map(preprocess).batch(64)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess)


# print(x.shape,y.shape,x_test.shape,x_test.shape)


def main():
    ## 集合网络
    conv_net = Sequential(conv_layers)
    fc_net = Sequential(fc_layers)
    ## 构建网络参数
    conv_net.build(input_shape=[None,32,32,3])
    fc_net.build(input_shape=[None,512])
    varibales = conv_net.trainable_variables + fc_net.trainable_variables ## 结合参数
    optimizer = optimizers.Adam(lr = 1e-4 ) ## 学习率

    # x = tf.random.normal([4,32,32,3]) ## 初始化参数
    # out = network(x)
    # print(out.shape)

    for epochs in range():
        """
        train datasets
        """
        for step , (x,y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                ## 输出层
                out = conv_net(x)  ## out [b,1,1,512] 
                out = tf.reshape(out,[-1,512])  ## => [c,512] 顺应参数的集合
                logits = fc_net(out) 
                # logits = tf.nn.softmax(logits)
                y_one_hot = tf.one_hot(y,depth=100)

                ## loss compution
                loss = tf.losses.categorical_crossentropy(y_one_hot,logits,from_logits=True)
                loss = tf.reduce_mean(loss)
            ## 计算梯度
            grades = tape.gradient(loss,variables)  ## 函数 与 需要优化的参数 
            optimizer.apply_gradients(zip(grades,variables))
            if step % 100 ==0 :
                print(step,"loss:",float(loss))

        """"
        test datasets
        """
        for x,y in db_test:
            out = conv_net(x)  ## out [b,1,1,512] 
            out = tf.reshape(out,[-1,512])  ## => [c,512] 顺应参数的集合
            logits = fc_net(out) 
            prob = tf.nn.softmax(logits,axis=1)
            pred = tf.argmax(prob,axis=1) ## 得到可能行最大的下标
            pred = tf.cast(pred,dtype=tf.int32) 
            ## 计算正确的数量
            correct = tf.cast(tf.equal(pred,y),dtype=int32)
            correct = tf.reduce_sum(correct)
            toatal_correct += correct
            total_sum += x.shape[0]
            acc = toatal_correct / total_sum
        print(step,"acc:",acc)



if __name__ == "__main__":
    main()
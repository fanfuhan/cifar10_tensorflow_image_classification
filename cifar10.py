
# coding: utf-8

# ### 导包

# In[ ]:


import tensorflow as tf
import os
import cifar10_input
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ### 设置算法超参数

# In[ ]:


learning_rate_init = 0.001
l2loss_ratio = 0.001
training_epochs = 5
batch_size = 100
display_step = 100
conv1_kernel_num = 64
conv2_kernel_num = 64
fc1_units_num = 256
fc2_units_num = 128
fc3_units_num = cifar10_input.NUM_CLASSES


# ### 数据集中输入图像的参数

# In[ ]:


dataset_dir = './cifar10_data/'
image_size = cifar10_input.IMAGE_SIZE
image_channel = 3
n_classes = cifar10_input.NUM_CLASSES
num_examples_per_epoch_for_train = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_examples_per_epoch_for_eval = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# In[ ]:


def get_distorted_train_batch(data_dir, batch_size):
    if not data_dir:
        raise ValueError('please supply a data_dir')
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    return images, labels


# In[ ]:


def get_undistorted_eval_batch(data_dir, eval_data, batch_size):
    if not data_dir:
        raise ValueError('please supply a data_dir')
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=batch_size)
    return images, labels


# ### 根据指定的维数返回初始化好的指定名称的权重 Variable

# In[ ]:


def WeightsVariable(shape, name_str='weights', stddev=0.1):
    # 单cpu
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

    # 多gpu
    # weights = tf.get_variable(name_str, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # return weights


# ### 根据指定的维数返回初始化好的指定名称的权重 Variable

# In[ ]:


def BiasesVariable(shape, name_str='biases', init_value=0.0):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)


# ### 2维卷积层的封装（包含激活函数）

# In[ ]:


def Conv2d(x, W, b, stride=1, padding='SAME', activation=tf.nn.relu, act_name='relu'):
    with tf.name_scope('conv2d_bias'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y, b)
    with tf.name_scope(act_name):
        y = activation(y)
    return y


# ### 2维池化层pool的封装

# In[ ]:


def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2, padding='SAME'):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding)


# ### 全连接层的封装

# In[ ]:


def FullyConnected(x, W, b, activation=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        y = activation(y)
    return y


# ### 为每一层的激活输出添加汇总节点

# In[ ]:


def AddActivationSummary(x):
    tf.summary.histogram('/activations', x)
    tf.summary.scalar('/sparsity', tf.nn.zero_fraction(x)) # 稀疏性


# ### 为所有损失节点添加标量汇总操作

# In[ ]:


def AddLossesSummary(losses):
    # 计算所有损失的滑动平均
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses)
    
    # 为所有损失及平滑处理的损失绑定标量汇总节点
    for loss in losses:
        tf.summary.scalar(loss.op.name + '(raw)', loss)
        tf.summary.scalar(loss.op.name + '(avg)', loss_averages.average(loss))
    return loss_averages_op


# ### 打印每一层输出张量的shape

# In[1]:


def print_layers_shape(t):
    print(t.op.name, ' ', t.get_shape().as_list())


# ### 前向推断过程

# In[ ]:


def Inference(images_holder):
    # 第一个卷积层
    with tf.name_scope('Conv2d_1'):
        weights = WeightsVariable(shape=[5, 5, image_channel, conv1_kernel_num], stddev=5e-2)
        biases = BiasesVariable(shape=[conv1_kernel_num])
        conv1_out = Conv2d(images_holder, weights, biases)
        AddActivationSummary(conv1_out)
        print_layers_shape(conv1_out)
        
    # 第一个池化层
    with tf.name_scope('Pool2d_1'):
        pool1_out = Pool2d(conv1_out, k=3, stride=2)
        
    # 第二个卷积层
    with tf.name_scope('Conv2d_2'):
        weights = WeightsVariable(shape=[5, 5, conv1_kernel_num, conv2_kernel_num], stddev=5e-2)
        biases = BiasesVariable(shape=[conv2_kernel_num])
        conv2_out = Conv2d(pool1_out, weights, biases)
        AddActivationSummary(conv2_out)
        
    # 第二个池化层
    with tf.name_scope('Pool2d_2'):
        pool2_out = Pool2d(conv2_out, k=3, stride=2)
    
    # 将二维特征图变为一维特征向量
    with tf.name_scope('FeatsReshape'):
        features = tf.reshape(pool2_out, [batch_size, -1])
        feats_dim = features.get_shape()[1].value  # 得到上一行 -1 所指代的值
        
    # 第一个全连接层
    with tf.name_scope('FC1_nonlinear'):
        weights = WeightsVariable(shape=[feats_dim, fc1_units_num], stddev=4e-2)
        biases = BiasesVariable(shape=[fc1_units_num], init_value=0.1)
        fc1_out = FullyConnected(features, weights, biases)
        AddActivationSummary(fc1_out)
        # 加入L2损失
        with tf.name_scope('L2_loss'):
            weight_loss = tf.multiply(tf.nn.l2_loss(weights), l2loss_ratio, name='fc1_weight_loss')
            tf.add_to_collection('losses', weight_loss)
    
    # 第二个全连接层
    with tf.name_scope('FC2_nonlinear'):
        weights = WeightsVariable(shape=[fc1_units_num, fc2_units_num], stddev=4e-2)
        biases = BiasesVariable(shape=[fc2_units_num], init_value=0.1)
        fc2_out = FullyConnected(fc1_out, weights, biases)
        AddActivationSummary(fc2_out)
        # 加入L2损失
        with tf.name_scope('L2_loss'):
            weight_loss = tf.multiply(tf.nn.l2_loss(weights), l2loss_ratio, name='fc2_weight_loss')
            tf.add_to_collection('losses', weight_loss)
        
    # 第三个全连接层
    with tf.name_scope('FC3_linear'):
        weights = WeightsVariable(shape=[fc2_units_num, fc3_units_num], stddev=1.0/fc2_units_num)
        biases = BiasesVariable(shape=[fc3_units_num])
        logits = FullyConnected(fc2_out, weights, biases, activation=tf.identity, act_name='linear')
        AddActivationSummary(logits)
        
    return logits


# ### 调用上面写的函数构造计算图，并设计会话流程

# In[ ]:


def TrainModel():
    with tf.Graph().as_default():
        # 计算图输入
        with tf.name_scope('Inputs'):
            images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel], name='images')
            labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')

        # 计算图前向推断过程
        with tf.name_scope('Inference'):
            logits = Inference(images_holder)

        # 定义损失层
        with tf.name_scope('Loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder, logits=logits)
            cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_loss')
            tf.add_to_collection('losses', cross_entropy_loss)
            # 总损失 = 交叉熵损失 + L2损失
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            average_losses = AddLossesSummary(tf.get_collection('losses') + [total_loss])

        # 定义优化训练层
        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=global_step)

        # 定义模型评估层
        with tf.name_scope('Evaluate'):
            top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k=1)

        # 定义获取训练样本批次的节点
        with tf.name_scope('GetTrainBatch'):
            images_train, labels_train = get_distorted_train_batch(data_dir=dataset_dir, batch_size=batch_size)

        # 定义获取测试样本批次的节点
        with tf.name_scope('GetTestBatch'):
            images_test, labels_test = get_undistorted_eval_batch(eval_data=True, data_dir=dataset_dir, batch_size=batch_size)

        # 收集所有汇总节点
        merged_summaries = tf.summary.merge_all()
            
        # 添加所有变量的初始化节点
        init_op = tf.global_variables_initializer()

        print("把计算图写入事件文件...")
        # graph_writer = tf.summary.FileWriter(logdir='events/', graph=tf.get_default_graph())
        # graph_writer.close()
        summary_writer = tf.summary.FileWriter(logdir='events/')
        summary_writer.add_graph(graph=tf.get_default_graph())
        summary_writer.flush()
        
        with tf.Session() as sess:
            sess.run(init_op)
            
            print('==>>>>>>>==开始在训练集上训练模型==<<<<<<<==')
            total_batches = int(num_examples_per_epoch_for_train / batch_size)
            print("per batch size: ", batch_size)
            print("train sample count per epoch:", num_examples_per_epoch_for_train)
            print("total batch count per epoch:", total_batches)
            # 启动数据读取队列
            tf.train.start_queue_runners()
            # 记录模型被训练的步数
            training_step = 0
            # 训练指定轮数，每一轮的训练样本总数为：num_examples_per_epoch_for_train
            for epoch in range(training_epochs):
                # 每一轮都要把所有的batch跑一遍
                for batch_idx in range(total_batches):
                    # 运行获取批次训练数据的计算图，取出一个批次数据
                    images_batch, labels_batch = sess.run([images_train, labels_train])
                    # 运行优化器训练节点
                    _, loss_value, avg_losses= sess.run([train_op, total_loss, average_losses], feed_dict={images_holder:images_batch, 
                                                                                labels_holder:labels_batch,
                                                                                learning_rate:learning_rate_init})
                    # 每调用一次训练节点，training_step就加1，最终 == training_epochs * total_batch
                    training_step = sess.run(global_step)
                    # 每训练display_step次，计算当前模型的损失和分类准确率
                    if training_step % display_step == 0:
                        # 运行Evaluate节点，计算当前批次的训练样本的准确率
                        predictions = sess.run([top_K_op], feed_dict={images_holder:images_batch, labels_holder:labels_batch})
                        # 计算当前批次的预测正确样本量
                        batch_accuracy = np.sum(predictions) / batch_size
                        print("train step: " + str(training_step) + ", train loss= " + "{:.6f}".format(loss_value) + 
                              ", train accuracy=" + "{:.5f}".format(batch_accuracy))
                        
                        # 运行汇总节点
                        summaries_str = sess.run(merged_summaries, feed_dict=
                                                 {images_holder: images_batch, labels_holder: labels_batch})
                        summary_writer.add_summary(summary=summaries_str, global_step=training_step)
                        summary_writer.flush()
            summary_writer.close()         
            print("训练完毕！")
                        
            print('==>>>>>>>==开始在测试集上评估模型==<<<<<<<==')
            total_batches = int(num_examples_per_epoch_for_eval / batch_size)
            total_examples = total_batches * batch_size # 当除不尽batch_size时，num_examples_per_epoch_for_evalv ！= total_examples
            print("per batch size: ", batch_size)
            print("test sample count per epoch:", total_examples)
            print("total batch count per epoch:", total_batches)
            correc_predicted = 0
            
            for test_step in range(total_batches):
                # 运行获取批次测试数据的计算图，取出一个批次数据
                images_batch, labels_batch = sess.run([images_test, labels_test])
                # 运行Evaluate节点，计算当前批次的训练样本的准确率
                predictions = sess.run([top_K_op], feed_dict={images_holder:images_batch, labels_holder:labels_batch})
                # 累计每个批次的预测正确样本量
                correc_predicted += np.sum(predictions)
            
            accuracy_score = correc_predicted / total_examples
            print("-------->accuracy on test examples: ",accuracy_score)


# In[ ]:


def main(argv=None):
    train_dir = './events/'
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    TrainModel()


# In[ ]:


if __name__ == '__main__':
    tf.app.run()


# ### 结果

"""Routine for decoding the CIFAR-10 binary file format."""

import os
import sys
import numpy as np
import tarfile #解压文件
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

# 用于描述CiFar数据集的全局变量
IMAGE_SIZE = 32  # 数据 height=width=IMAGE_SIZE
IMAGE_DEPTH = 3 # 数据通道
NUM_CLASSES_CIFAR10 = 10 # CiFar10 数据，10分类
NUM_CLASSES_CIFAR20 = 20  # CiFar100 数据，粗分类，20类，coarse label
NUM_CLASSES_CIFAR100 = 100 # CiFar100 数据，细分类，100类 , fine label

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000 # TRAIN data
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000 # EVAL data

# 从网站上下载数据集存放到 data_dir 指定的目录下
CIFAR10_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
CIFAR100_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'


def maybe_download_and_extract(data_dir, data_url=CIFAR10_DATA_URL):
    dest_directory  = data_dir  # 存放数据目录
    DATA_URL = data_url
    # 判断是否存在，不存在就创建，
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory,filename)
    # 判断数据是否已经存在，否则执行下载
    if not os.path.exists(filepath):
        def _progress(count,block_size,total_size):
            sys.stdout.write('\r>>Downloading %s %.1f%%'%(filename,
                                                          float(count*block_size)/float(total_size)*100.0))
            sys.stdout.flush()
        filepath,_ = urllib.request.urlretrieve(DATA_URL,filepath,_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully down loaded',filename,statinfo.st_size,'bytes.')
    # 将下载下来的压缩文件解压缩
    tarfile.open(filepath,'r:gz').extractall(dest_directory)


def read_cifar10(filename_queue, coarse_or_fine=None):

    # 定义类，为记录类，一个记录表示一张图像
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # input format.
    result.height = 32
    result.width = 32
    result.depth = 3

    # cifar10 binary 中的样本记录
    #  <1 x label><3072 x pixel>
    #         .......
    #  <1 x label><3072 x pixel>

    # label 标签字节数
    label_bytes = 1

    # 图像字节数
    image_bytes = result.height * result.width * result.depth
    # 每一条样本记录由 标签+图像 组成， 其字节数是固定的，3073
    record_bytes = label_bytes + image_bytes

    # 读取器，创建一个固定长度记录读取器，读取一个样本记录的所有字节 (label_bytes+image_bytes)
    # 由于 cifar10 中的记录没有 header_bytes 和 footer_bytes, 所以头字节与尾字节设置为0
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes, header_bytes=0, footer_bytes=0)
    # 调用读取器对象的 read 方法返回一条记录，
    # result.key, value样本在整体样本中的位置; value,样本中的字节码
    result.key, value = reader.read(filename_queue)

    # 将一个字节组成的 string 类别的记录转换为长度为 record_bytes, 类型为 uint8 的一个数字向量
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 取标签，第一个字节代表了标签， 我们把他从 uint8 转换为 int32
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32) # [0, label_bytes]

    # 取图像数据，剩余的所有字节都是图像数据， 把他从一维张量 [depth * height * width]
    # 转为三维张量 [ depth, height, width]
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes+image_bytes]), # [label_bytes， image_bytes]
                           [result.depth, result.height, result.width])
    # 把图像的空间位置和深度位置顺序由 [depth, height, width] 转换成 [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def read_cifar100(filename_queue, coarse_or_fine='fine'):

    # 定义类，为记录类，一个记录表示一张图像
    class CIFAR100Record(object):
        pass
    result = CIFAR100Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # input format.
    result.height = 32
    result.width = 32
    result.depth = 3

    # cifar100 binary 中的样本记录
    # Cifar100 中每个样本记录都有两个类别标签，
    # 第一个字节是粗略分类标签，第二个字节是精细分类标签：

    #  <1 x coarse label><1 x fine label><3072 x pixel>
    #         .......
    #  <1 x coarse label><1 x fine label><3072 x pixel>

    # label 标签字节数
    coarse_label_bytes = 1
    fine_label_bytes = 1

    # 图像字节数
    image_bytes = result.height * result.width * result.depth
    # 每一条样本记录由 粗标签+细标签+图像 组成， 其字节数是固定的，3074
    record_bytes = coarse_label_bytes + fine_label_bytes + image_bytes

    # 读取器，创建一个固定长度记录读取器，读取一个样本记录的所有字节 (label_bytes+image_bytes)
    # 由于 cifar10 中的记录没有 header_bytes 和 footer_bytes, 所以头字节与尾字节设置为0
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes, header_bytes=0, footer_bytes=0)
    # 调用读取器对象的 read 方法返回一条记录，
    # result.key, value样本在整体样本中的位置; value,样本中的字节码
    result.key, value = reader.read(filename_queue)

    # 将一个字节组成的 string 类别的记录转换为长度为 record_bytes, 类型为 uint8 的一个数字向量
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 取粗分类标签，第一个字节代表了粗分类标签， 我们把他从 uint8 转换为 int32
    coarse_label = tf.cast(
        tf.strided_slice(record_bytes, [0], [coarse_label_bytes]), tf.int32) # [0, label_bytes]
    # 取细分类标签，第二个字节代表了细分类标签， 我们把他从 uint8 转换为 int32
    fine_label = tf.cast(
        tf.strided_slice(record_bytes, [coarse_label_bytes], [coarse_label_bytes+fine_label_bytes]), tf.int32)

    if coarse_or_fine == 'fine':
        result.label = fine_label # 细分类标签，100类别
    else:
        result.label = coarse_label  # 粗分类标签，20类别

    # 取图像数据，剩余的所有字节都是图像数据， 把他从一维张量 [depth * height * width]
    # 转为三维张量 [ depth, height, width]
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes,
                         [coarse_label_bytes+fine_label_bytes],
                         [coarse_label_bytes+fine_label_bytes+image_bytes]),
        [result.depth, result.height, result.width])
    # 把图像的空间位置和深度位置顺序由 [depth, height, width] 转换成 [height, width, depth].32*32*3
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size,shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16  # 并发执行线程
  if shuffle:
      images, label_batch = tf.train.shuffle_batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
  else:
      images, label_batch = tf.train.batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,)

  # Display the training images in the visualizer.
  # tf.summary.image('images', images, max_outputs=9)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(cifar10or20or100, data_dir, batch_size):
    # 数据处理，扩充数据集
    """
    使用 Reader ops 构造 distorted input 用于 CIFAR 的训练
    输入参数：
    cifar10or20or100: 指定要读取的数据集是 cifar10 还是细分类 cifar100 或者粗分类 cifar100
    data_dir: 指向 CIFAR-10 或者 CIFAR-100 数据集的目录
    batch_size: 单个批次的图像数量
    :return: images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size
             labels: Labels. 1D tensor of [batch_size] size
    """
    # 判断是读取 cifar10 还是 cifar100 （cifar100又分20粗分类或者100细分类）
    if cifar10or20or100 == 10:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)for i in xrange(1, 6)]
        read_cifar = read_cifar10
        coarse_of_fine = None
    if cifar10or20or100 == 20:
        filenames = [os.path.join(data_dir, 'train.bin')]
        read_cifar = read_cifar100
        coarse_of_fine = 'coarse'
    if cifar10or20or100 == 100:
        filenames = [os.path.join(data_dir, 'train.bin')]
        read_cifar = read_cifar100
        coarse_of_fine = 'fine'
    # 检查数据文件是否存在，不存在报错
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 根据文件名列表创建一个文件名队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 从文件名队列的文件中读取样本
    read_input = read_cifar(filename_queue,coarse_or_fine=coarse_of_fine)

    # 将无符号8位图像数据转换成 float32 位
    casted_image = tf.cast(read_input.uint8image, tf.float32)

    # 要生成的目标图像的大小， 在这里与原图像的尺寸保持一致
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 为图像添加 Padding=4, 图像尺寸变为 [32+4,,32+4], 为后面的随机裁切流出位置
    padded_image = tf.image.resize_image_with_crop_or_pad(casted_image, width+4, height+4)

    # 下面的操作为原始图像添加了很多不同的 distortions, 扩增了原始训练数据集

    # 第一种，裁剪，在扩展的 [36, 36]大小的图像中随机裁剪出 [height,width], 即[32, 32] 的图像区域
    distorted_image = tf.random_crop(padded_image, [height, width, 3])

    # 第二种，水平翻转，将图像进行随机的水平翻转，（左边和右边的像素对调）
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # 下面这两个操作不满足交换律， 即 亮度调整+对比度调整 和 对比度调整+亮度调整，
    #  两个操作执行先后顺序产生的结果是不一样的， 可以采取随机的顺序来执行这两个操作
    # np.random.randn : 标准正太分布,均值：0；方差：1
    np_random_randn  = np.random.randn(1)
    if np_random_randn>0:
        # 第三种，亮度调整，原像素加上一个随机数，数的范围在 [-63,63]
        distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)
        # 第四种，对比度调整，原像素乘以一个随机数，数的范围在 [0.2, 1.8]
        distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)
    else:
        # 第四种，对比度调整，原像素乘以一个随机数，数的范围在 [0.2, 1.8]
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        # 第三种，亮度调整，原像素加上一个随机数，数的范围在 [-63,63]
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)


    # 数据集标准化操作： 减均值 + 方差归一化，
    float_image = tf.image.per_image_standardization(distorted_image)

    # 设置数据集中张量的形状
    float_image.set_shape([height, width, 3]) # 32*32*3
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

    # 通过构造样本队列产生一个批次的图像和标签
    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle =True)


def inputs(cifar10or20or100,eval_data, data_dir, batch_size):
    """
    cifar10or20or100: 指定要读取的数据集是 cifar10 还是细分类 cifar100 或者粗分类 cifar100
    eval_data: Ture or False, 指示要读取的是训练集还是测试集
    data_dir: 指向 CIFAR-10 或者 CIFAR-100 数据集的目录
    batch_size: 单个批次的图像数量
    :return: images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size
             labels: Labels. 1D tensor of [batch_size] size
    """
    print('...正在调用...cifar_input...'+'cifar'+str(cifar10or20or100))

    # 判断是读取 cifar10 还是 cifar100 （cifar100又分20粗分类或者100细分类）
    if cifar10or20or100 ==10:
        read_cifar = read_cifar10
        coarse_of_fine = None
        if not eval_data:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)for i in xrange(1, 6)] # 列表生成
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    if cifar10or20or100 ==20 or cifar10or20or100 ==100:
        read_cifar = read_cifar100
        if not eval_data:
            filenames = [os.path.join(data_dir, 'train.bin')] # 列表生成
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    if cifar10or20or100 == 100:
        coarse_of_fine = 'fine'
    if cifar10or20or100 == 20:
        coarse_of_fine = 'coarse'

    # 检查指定目录文件是否存在，不存在则报错
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 根据文件名列表创建一个文件名队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 从文件名队列的文件中读取样本
    read_input = read_cifar(filename_queue, coarse_or_fine=coarse_of_fine)
    # 将无符号 8位图像数据转换成 float32 位
    casted_image = tf.cast(read_input.uint8image, tf.float32)

    # 要生成的目标图像的大小， 在这里与原图像的尺寸保持一致
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 用于评估过程的图像数据预处理 ，此处原图像与处理后图像大小一样，都是32*32，则没有发生裁剪处理
    resized_image = tf.image.resize_image_with_crop_or_pad(casted_image,width, height)

    # 数据集标准化操作： 减去均值 + 方差归一化
    float_image = tf.image.per_image_standardization(resized_image)

    # 设置数据集中张量的形状
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

    # 通过构造样本队列产生一个批次的图像和标签
    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,shuffle=None)

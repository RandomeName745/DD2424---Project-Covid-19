import tensorflow as tf
import numpy


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1, ratio=4):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()

        self.excitation1 = tf.keras.layers.Dense(units=filter_num//ratio,
                                                 activation=tf.nn.relu)

        self.excitation2 = tf.keras.layers.Dense(units=filter_num,
                                                 activation=tf.nn.sigmoid)

        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        squeeze = self.squeeze(x)
        excitation1 = self.excitation1(squeeze)
        excitation2 = self.excitation2(excitation1)
        x = x * excitation2

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1, ratio=4):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        # SE block
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()

        self.excitation1 = tf.keras.layers.Dense(units=filter_num * 4//ratio,
                                                 activation=tf.nn.relu)

        self.excitation2 = tf.keras.layers.Dense(units=filter_num * 4,
                                                 activation=tf.nn.sigmoid)

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        squeeze = self.squeeze(x)
        excitation = self.excitation1(squeeze)
        excitation = self.excitation2(excitation)
        # print("---------------------------", excitation.shape)
        # print("---------------------------", x.shape)
        # excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        x = x * excitation

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layer(filter_num, blocks, stride=1, ratio=4):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride, ratio=ratio))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1, ratio=ratio))

    return res_block


def make_bottleneck_layer(filter_num, blocks, stride=1, ratio=4):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride, ratio=ratio))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1, ratio=ratio))

    return res_block


# def SE_layer(input_feature, layer_name, ratio=4):

#     with tf.name_scope(layer_name):
#         channel = input_feature.get_shape()[-1]

#         squeeze = tf.keras.layers.GlobalAveragePooling2D(input_feature)
#         excitation = tf.layers.dense(inputs=squeeze,
#                                      units=channel//ratio,
#                                      activation=tf.nn.relu,
#                                      name=layer_name + 'bottleneck_fc')

#         excitation = tf.layers.dense(inputs=excitation,
#                                      units=channel,
#                                      activation=tf.nn.sigmoid,

#                                      name=layer_name + 'recover_fc')

#         scale = input_feature * excitation

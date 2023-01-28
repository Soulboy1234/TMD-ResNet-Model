import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops

## ResNets + Droupout + Log + tanh
def ResNetsDroupoutLogTanhBlock(inputData, dense, activationFun='tanh', dropoutRate=0.2):
    x = inputData
    Dense1 = tf.keras.layers.Dense(dense,   activation=activationFun)(x)
    Dense2 = tf.keras.layers.Dense(dense*2, activation=activationFun)(Dense1)
    Dense3 = tf.keras.layers.Dense(dense*4, activation=activationFun)(Dense2)
    Dense3 = tf.keras.layers.Dropout(dropoutRate)(Dense3, training=True)
    Dense4 = tf.keras.layers.Dense(dense*2, activation=activationFun)(Dense3)
    Dense5 = tf.keras.layers.Dense(dense,   activation=activationFun)(Dense4)
    outputData = tf.keras.layers.add([Dense1,Dense5])
    return outputData

def TMDResNetModel(dropoutRate=0.6):
    # Reset graph
    ops.reset_default_graph()
    # Input
    inputData = keras.layers.Input(shape=(27),name='input')
    # ResNets + Dropout
    block1 = ResNetsDroupoutLogTanhBlock(inputData, 128, activationFun='tanh', dropoutRate=dropoutRate)
    block2 = ResNetsDroupoutLogTanhBlock(block1, 256, activationFun='tanh', dropoutRate=dropoutRate)
    block3 = ResNetsDroupoutLogTanhBlock(block2, 512, activationFun='tanh', dropoutRate=dropoutRate)
    block4 = ResNetsDroupoutLogTanhBlock(block3, 256, activationFun='tanh', dropoutRate=dropoutRate)
    block5 = ResNetsDroupoutLogTanhBlock(block4, 128, activationFun='tanh', dropoutRate=dropoutRate)
    block6 = ResNetsDroupoutLogTanhBlock(block5, 64, activationFun='tanh', dropoutRate=dropoutRate)
    block7 = ResNetsDroupoutLogTanhBlock(block6, 32, activationFun='tanh', dropoutRate=dropoutRate)
    # Output
    outputData = tf.keras.layers.Dense(1, activation='relu')(block7)
    # Build
    modelName = 'TMDResNetModel'
    network = keras.models.Model(inputs=inputData, outputs=outputData, name=modelName)

    return network


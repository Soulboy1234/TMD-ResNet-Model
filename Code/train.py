import time
import tensorflow as tf
from tensorflow import keras
from model import TMDResNetModel
from dataIO import loadData, saveWeight, dataGenerator
import yaml
import pickle

### load config
with open('./config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.BaseLoader)

### load data
satelliteData, trainIndexList, validationIndexList, testIndexList \
    = loadData(cfg['Data']['filePath'],cfg['Data']['fileName'],cfg['Data']['indexFileName'])

### load model
network = TMDResNetModel(dropoutRate=0.6)
modelName = network.name

### init data generator
trainGen = dataGenerator(satelliteData,trainIndexList, int(cfg['Train']['batchSize']))
valGen = dataGenerator(satelliteData,validationIndexList, int(cfg['Train']['batchSize']))

### init training
history_all = []
saveName = modelName + '_weight_init.hdf5'
weightSavePath = cfg['Train']['weightSavePath']
saveWeight(weightSavePath,saveName,network)
loop = 0
lrList = []
for lr in cfg['Train']['learningRateList']:
    lrList.append(float(lr))

### training
for lr in lrList:
    metrics = ['mae']
    network.compile(loss=cfg['Train']['lossFunction'],
                    optimizer=keras.optimizers.Adam(lr),
                    metrics=metrics)  # keras.optimizers.Adam()
    loop = loop + 1
    print(f'loop:{loop} learning rate:{lr}')
    try:
        network.load_weights(weightSavePath + modelName + '_weight_loop_{}.hdf5'.format(loop - 1))
        print('load: ' + weightSavePath + modelName + '_weight_loop_{}.hdf5'.format(loop - 1))
    except:
        print('network weight from random')
        network.load_weights(weightSavePath + modelName + '_init.hdf5')

    # save training log
    logName = modelName + '_Training_{}'.format(time.strftime('%Y-%m-%d-%H-%M')) + ' loop{}.log'.format(loop)
    csvLogger = tf.keras.callbacks.CSVLogger(weightSavePath + 'Log/' + logName, separator=' ', append=False)
    history = network.fit(trainGen,
                          steps_per_epoch=int(cfg['Train']['stepsPerEpoch']),
                          epochs=int(cfg['Train']['epochPerLearningRate']),
                          validation_data=valGen,
                          validation_steps=308,
                          callbacks=[csvLogger])
    history_all.append(history.history)

    # save check point
    saveName = modelName + '_weight_loop_{}.hdf5'.format(loop)
    saveWeight(weightSavePath, saveName, network)

# save history
historyName = modelName + '_Training_history_{}'.format(time.strftime('%Y-%m-%d-%H-%M'))
with open(weightSavePath + 'History/' + historyName, 'wb') as file_pi:
    pickle.dump(history_all, file_pi)

# save model
loop_final = 5
saveName = modelName + '_weight_loop_{}.hdf5'.format(loop_final)
network.load_weights(weightSavePath + saveName)
network.save(cfg['Train']['modelSavePath'] + 'TMDResNetModel.hdf5')

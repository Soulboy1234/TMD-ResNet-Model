from dataIO import loadData, dataGenerator
import yaml
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from model import TMDResNetModel

### load config
with open('./config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.BaseLoader)

### load data
satelliteData, trainIndexList, validationIndexList, testIndexList \
    = loadData(cfg['Data']['filePath'],cfg['Data']['fileName'],cfg['Data']['indexFileName'])

### load model - optional 1
network = load_model(cfg['Train']['modelSavePath'] + 'TMDResNetModel.hdf5')

### load model weight - optional 2
# network = TMDResNetModel(dropoutRate=0.6)
# modelName = network.name
# loop_final = 5
# saveName = modelName + '_weight_loop_{}.hdf5'.format(loop_final)
# network.load_weights(cfg['Train']['weightSavePath'] + saveName)

### generator data
testGen = dataGenerator(satelliteData,testIndexList, int(cfg['Test']['batchSize']))
inputArray, gtArray = next(testGen)

### predict
result = network.predict(inputArray)

### show
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(result,gtArray)
plt.xlabel('TMDResNetModel TMD')
plt.ylabel('CHMAP TMD')
plt.plot([0,12],[0,12],'-r', linewidth=2)
plt.title('predict vs gt')

plt.subplot(1,2,2)
plt.plot(gtArray,'.k', linewidth=2)
plt.plot(result,'.r', linewidth=2)
plt.ylabel('TMD')
plt.title('predict vs gt')

plt.show()
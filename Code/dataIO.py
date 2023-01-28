import numpy as np
import random
import h5py
import os

## get batch input
def getInputBatchLog(hdf5File, dataIndexList, batchSize):
    inputArray = np.zeros([batchSize, 27])
    gtArray = np.zeros([batchSize, 1])
    try:
        for i in range(batchSize):
            index = random.choice(dataIndexList)
            inputArray[i, :] = hdf5File['input'][index]
            gtArray[i, :] = hdf5File['gt'][index, 0] * 10 ** 12
            # gt = Nn, Nn250, Nn300, Nn400, Nn500, free_T
        errorFlag = False

    except:
        errorFlag = True

    '''
    Input:
    00. ut    - s   - sin
    01. ut    - s   - cos
    02. doy   - day - sin
    03. doy   - day - cos
    04. lon   - °   - sin
    05. lon   - °   - cos
    06. lat   - °   - sin
    07. lat   - °   - cos
    08. alt   - km 

    09. f107        - sfu (daily mean)   
    10. f107a       - sfu (81 days mean)
    f107 = log10(f107)/3

    11. SYMH_d      - nt  (daily mean)     
    12. SYMH        - nt  (minute)
    13. SYMH_3h     - nt  (last 0 -3h mean)
    14. SYMH_6h     - nt  (last 3 -6h mean)
    15. SYMH_9h     - nt  (last 6 -9h mean)
    16. SYMH_12h    - nt  (last 9 -12h mean)
    17. SYMH_12_33h - nt  (last 12-33h mean)
    18. SYMH_33_57h - nt  (last 33-57h mean)
    SYMH = SYMH/500

    19. AE_d      - nt  (daily mean)     
    20. AE        - nt  (minute)
    21. AE_3h     - nt  (last 0 -3h mean)
    22. AE_6h     - nt  (last 3 -6h mean)
    23. AE_9h     - nt  (last 6 -9h mean)
    24. AE_12h    - nt  (last 9 -12h mean)
    25. AE_12_33h - nt  (last 12-33h mean)
    26. AE_33_57h - nt  (last 33-57h mean)
    AE = log10(AE)/5

    Output:
    1.Nn     - kg/m3 *10**12
    '''

    inputArray[:, 8] = (inputArray[:, 8] - 250) / 300

    inputArray[:, 9]  = np.log10(inputArray[:, 9])/3
    inputArray[:, 10] = np.log10(inputArray[:, 10])/3

    inputArray[:, 11] = inputArray[:, 11]/500
    inputArray[:, 12] = inputArray[:, 12]/500
    inputArray[:, 13] = inputArray[:, 13]/500
    inputArray[:, 14] = inputArray[:, 14]/500
    inputArray[:, 15] = inputArray[:, 15]/500
    inputArray[:, 16] = inputArray[:, 16]/500
    inputArray[:, 17] = inputArray[:, 17]/500
    inputArray[:, 18] = inputArray[:, 18]/500

    inputArray[:, 19] = np.log10(inputArray[:, 19])/5
    inputArray[:, 20] = np.log10(inputArray[:, 20])/5
    inputArray[:, 21] = np.log10(inputArray[:, 21])/5
    inputArray[:, 22] = np.log10(inputArray[:, 22])/5
    inputArray[:, 23] = np.log10(inputArray[:, 23])/5
    inputArray[:, 24] = np.log10(inputArray[:, 24])/5
    inputArray[:, 25] = np.log10(inputArray[:, 25])/5
    inputArray[:, 26] = np.log10(inputArray[:, 26])/5

    # filter the data
    fp = ~np.isnan(gtArray[:, 0]) & (inputArray[:, 8] >= 0) & (gtArray[:, 0] > 0) \
         & (inputArray[:, 9] <= 0.9) & (inputArray[:, 10] <= 0.9) \
         & (inputArray[:, 19] <= 1) & (inputArray[:, 20] <= 1) \
         & (inputArray[:, 21] <= 1) & (inputArray[:, 22] <= 1) \
         & (inputArray[:, 23] <= 1) & (inputArray[:, 24] <= 1) \
         & (inputArray[:, 25] <= 1) & (inputArray[:, 26] <= 1) \
         & (inputArray[:, 11] <= 1) & (inputArray[:, 12] <= 1) \
         & (inputArray[:, 13] <= 1) & (inputArray[:, 14] <= 1) \
         & (inputArray[:, 15] <= 1) & (inputArray[:, 16] <= 1) \
         & (inputArray[:, 17] <= 1) & (inputArray[:, 18] <= 1) \
         & (inputArray[:, 11] >= -1) & (inputArray[:, 12] >= -1) \
         & (inputArray[:, 13] >= -1) & (inputArray[:, 14] >= -1) \
         & (inputArray[:, 15] >= -1) & (inputArray[:, 16] >= -1) \
         & (inputArray[:, 17] >= -1) & (inputArray[:, 18] >= -1)

    inputArray = inputArray[fp, :]
    gtArray = gtArray[fp, :]
    return inputArray, gtArray, errorFlag

def loadData(hdf5Folder,hdf5FileName,dataSetIndexFileName):
    # hdf5Folder = '../Data/hdf5Data/'
    # hdf5FileName = 'Mehta_CHAMP_Input.hdf5'
    # dataSetIndexFileName = 'Mehta_CHAMP_dataSetIndex.hdf5'
    satelliteData = h5py.File(hdf5Folder+hdf5FileName,'r')

    # Read into the memory (as dict), speed up training - Optional setting
    satelliteData_dict = {}
    satelliteData_dict['input'] = satelliteData['input'][:]
    satelliteData_dict['gt'] = satelliteData['gt'][:]
    satelliteData = satelliteData_dict
    # gt - TMD(in situ, 250km, 300km, 400km, 500km) + T_free

    dataSetIndexFile = h5py.File(hdf5Folder+dataSetIndexFileName,'r')
    trainIndexList = dataSetIndexFile['trainIndexList'][:]
    validationIndexList = dataSetIndexFile['validationIndexList'][:]
    testIndexList = dataSetIndexFile['testIndexList']
    return satelliteData, trainIndexList, validationIndexList, testIndexList

def saveWeight(savePath,saveName,network):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    if not os.path.exists(savePath + 'Log'):
        os.makedirs(savePath + 'Log')
    if not os.path.exists(savePath + 'History'):
        os.makedirs(savePath + 'History')
    network.save_weights(savePath + saveName)

# data generator
def dataGenerator(data, dataIndexList, batchsize=32):
    while True:
        errorFlag = True
        while errorFlag:
            inputArray, gtArray, errorFlag = getInputBatchLog(data,dataIndexList,batchsize)
        yield inputArray, gtArray
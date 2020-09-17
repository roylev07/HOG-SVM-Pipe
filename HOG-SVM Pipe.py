import cv2
import os
import pickle
import numpy as np
import random
from matplotlib import pyplot as plt
import skimage
from skimage.feature import hog
from sklearn.svm import LinearSVC, SVC
import sklearn
import statistics


def getParameters():
    """
    these parameters are the best parameters that we found in the tuning pipe (below) for our best model (rbf SVM)
    (trainTestSplit and trainValSplit were not tuned and degree parameter is relevant only for poly SVM)
    :return: dictionary of all the relevant parameters for all the models
    """
    s = 100
    m = 11
    n = 20
    bins = 9
    c = 3
    gamma = 0.1
    degree = 2
    trainTestSplit = 20
    trainValSplit = 14
    params = {'S': s, 'M': m, 'N': n, 'Bins': bins, 'C': c, 'Gamma': gamma, 'Degree': degree, 'TrainTestSplit': trainTestSplit,
              'TrainValSplit': trainValSplit}
    return params


def InitData(path):
    """
    loads the data from the desired path
    :param path: the path of the data folders
    :return: dictionary of 2 lists- Data (images) and the Labels (classes)
    """
    DandL = {'Data': [], 'Labels': []}
    dirs = os.listdir(path)
    for j in class_indices:
        folder = dirs[j - 1]  # j-1 because dirs start from index 0, and classes start from 1
        folder_path = path + folder
        images = os.listdir(folder_path)
        i = 0
        while i < len(images) and i < 40:  ## Loop on images in folder
            img = cv2.imread(path + '/' + folder + "/" + images[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            DandL['Data'].append(img)
            DandL['Labels'].append(j)
            i = i + 1
    return DandL


def TrainTestSplit(data, labels, params):
    """
    splits the data into train data and test data with their respective labels
    :param data: all the data (images) that has been loaded
    :param labels: all the labels of the data (classes)
    :param params: dictionary of all the parameters including the trainTestSplit parameter
    :return: dictionary of 4 lists- 2 lists for the train (data and labels) and 2 for the test
    """
    splitData = {'TrainData': [], 'TrainLabels': [], 'TestData': [], 'TestLabels': []}
    for index in class_indices:
        tempData = []
        for i in range(0, len(data)):
            if index == labels[i]:
                tempData.append(data[i])
        # after we finish adding all imgs from the current class to temp
        for i in range(0, len(tempData)):
            if i < params['TrainTestSplit']:
                splitData['TrainData'].append(tempData[i])
                splitData['TrainLabels'].append(index)
            else:
                splitData['TestData'].append(tempData[i])
                splitData['TestLabels'].append(index)
    return splitData


def Prepare(data, params):
    """
    makes the representation of the images by HOG. prepares the data for being fit to act as input for the models.
    the preparation includes resize of the images.
    :param data: data to make representation for
    :param params: dictionary of all the parameters including m, n and s parameters
    :return: list of the representations of all the images in data param. each item in this list is a HOG representation
    vector of one image.
    """
    rep = []
    bins = params['Bins']
    m = params['M']
    n = params['N']
    s = params['S']
    for img in data:
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        imRep = skimage.feature.hog(img, orientations=bins, pixels_per_cell=(m, n), cells_per_block=(2, 2),
                                    feature_vector=True)
        rep.append(imRep)
    return rep


def Train(dataRep, labels, params, type):
    """
    trains the data on the desired model (by the required type) regarding all the relevant parameters
    :param dataRep: list of all the representation vectors of the images
    :param labels: list of all the labels of the images
    :param params: dictionary of all the parameters including the relevant for each model
    :param type: represents the required model type (Linear, Rbf or Poly)
    :return: linear SVM model or list of m models which returned from 'm_class_SVM_train' function
    """
    c = params['C']
    if type == 'Linear':
        model = LinearSVC(dual=False, C=c, multi_class='ovr')
        model.fit(dataRep, labels)
    else:
        model = m_class_SVM_train(dataRep, labels, params, type)
    return model


def m_class_SVM_train(dataRep, labels, params, type):
    """
    trains the data for Rbf and Poly models and makes the implementation of m classes SVM.
    the ith model tags the ith class as +1 and the other classes as -1 and makes a rbf/poly svm model.
    :param dataRep: list of all the representation vectors of the images
    :param labels: list of all the labels of the images
    :param params: dictionary of all the parameters including the relevant for each model
    :param type: represents the required model type (Linear, Rbf or Poly)
    return: list of m SVM models (rbf or poly)
    """
    modelsList = []
    for j in class_indices:
        tempLabels = []
        for i in range(0, len(labels)):
            if labels[i] == j:
                tempLabels.append(1)
            else:
                tempLabels.append(-1)
        c = params['C']
        gamma = params['Gamma']
        degree = params['Degree']
        if type == 'rbf':
            tempModel = SVC(C=c, kernel='rbf', gamma=gamma)
        else:
            tempModel = SVC(C=c, kernel='poly', degree=degree)
        tempModel.fit(dataRep, tempLabels)
        modelsList.append(tempModel)
    return modelsList


def Test(model, testDataRep, type):
    """
    tests the model by the test data
    :param model: the required model to test
    :param testDataRep: list of all the representation vectors of the images
    :param type: represents the required model type (Linear, Rbf or Poly)
    :return: dictionary of results of the linear/rbf/poly models. the dictionary consists of a score matrix and a
    predictions list. rbf/poly results returned by 'm_class_SVM_predict'function.
    """
    if type == "Linear":
        results = {'Scores' : model.decision_function(testDataRep), 'Predictions' : model.predict(testDataRep)}
    else:
        results = m_class_SVM_predict(model, testDataRep)
    return results


def m_class_SVM_predict(models, dataRep):
    """
    gets data representation the makes calculates its results (score matrix and predictions)
    :param models: list of m rbf/poly SVM models
    :param dataRep: list of representation vectors of the required data to get the results on
    :return: dictionary of results. the dictionary consists of a score matrix and a
    predictions list.
    """
    Predictions = []
    ScoresMatrix = np.zeros((len(dataRep), len(models)))
    for j in range(0, len(models)):
        scores = models[j].decision_function(dataRep)
        for i in range(0, len(scores)):
            ScoresMatrix[i][j] = scores[i]
    for i in range(0, len(dataRep)):
        index = np.argmax(ScoresMatrix[i])
        Predictions.append(class_indices[index])
    results = {'Scores': ScoresMatrix, 'Predictions': Predictions}
    return results


def Evaluate(results, testLabels):
    """
    evaluates the model results by the test labels and makes summary.
    :param results: dictionary of results. the dictionary consists of a score matrix and a predictions list.
    :param testLabels: list of representation vectors of the required data to evaluate
    :return: dictionary that contains the error rate, confusion matrix and 2 images for each class with the largest
    error (if exist).
    """
    predictions = results['Predictions']
    accuracyRate = sklearn.metrics.accuracy_score(testLabels, predictions)
    errorRate = 1-accuracyRate
    confusionMatrix = sklearn.metrics.confusion_matrix(testLabels, predictions)
    largestErrorIndices = []
    scoreMatrix= results['Scores']
    for j in range(0, len(class_indices)):
        tempErrorsAndIndices= {'Errors': [], 'Indices': []}
        for i in range(0, len(testLabels)):
            if (testLabels[i] == class_indices[j]) and (predictions[i] != class_indices[j]):
                tempErrorsAndIndices['Errors'].append(scoreMatrix[i][j] - max(scoreMatrix[i]))  # negative number
                tempErrorsAndIndices['Indices'].append(i)
        topClassjErrorIndices = []
        if len(tempErrorsAndIndices['Indices']) > 2:
            largestErrorLocalIndex = tempErrorsAndIndices['Errors'].index(min(tempErrorsAndIndices['Errors']))
            largestErrorGlobalIndex = tempErrorsAndIndices['Indices'][largestErrorLocalIndex]
            tempErrorsAndIndices['Errors'].pop(largestErrorLocalIndex)
            tempErrorsAndIndices['Indices'].pop(largestErrorLocalIndex)
            secondErrorLocalIndex = tempErrorsAndIndices['Errors'].index(min(tempErrorsAndIndices['Errors']))
            secondErrorGlobalIndex = tempErrorsAndIndices['Indices'][secondErrorLocalIndex]

            topClassjErrorIndices.append(largestErrorGlobalIndex)
            topClassjErrorIndices.append(secondErrorGlobalIndex)
        else:
            for k in range(0, len(tempErrorsAndIndices['Indices'])):
                topClassjErrorIndices.append(tempErrorsAndIndices['Indices'][k])
        largestErrorIndices.append(topClassjErrorIndices)
    return {'ErrorRate': errorRate, 'ConfusionMatrix': confusionMatrix, 'largestErrorIndices': largestErrorIndices}


def ReportResults(summary, testData):
    """
    prints the summary of the model evaluation
    :param summary: dictionary of the evaluation which contains the error rate, the confusion matrix and list of m
    lists (pre each class) which each of them contains the 2 images (or less) with the largest error.
    :param testData: list of the images of the test data.
    """
    print("Test Error: " + str(summary['ErrorRate']))
    print("Confusion Matrix:")
    print(summary['ConfusionMatrix'])
    for j in range(0, len(class_indices)):
        classJIndices = summary['largestErrorIndices'][j]
        if len(classJIndices) > 0:
            for i in range(0, len(classJIndices)):
                index = classJIndices[i]
                plt.imshow(testData[index], cmap='gray')
                plt.title("Class " + str(class_indices[j]) + "- Image #" + str(i+1))
                plt.show()


#----------Main--------------------------

data_path = "C:/Users/wr/Downloads/101_ObjectCategories/"
class_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Params = getParameters()
DandL = InitData(data_path)
SplitData = TrainTestSplit(DandL['Data'], DandL['Labels'], Params)
TrainDataRep = Prepare(SplitData['TrainData'], Params)
RbfSvmModel = Train(TrainDataRep, SplitData['TrainLabels'], Params, "rbf")
TestDataRep = Prepare(SplitData['TestData'], Params)
RbfSvmResults = Test(RbfSvmModel, TestDataRep, "rbf")
SummaryByRbfSvm = Evaluate(RbfSvmResults, SplitData['TestLabels'])
ReportResults(SummaryByRbfSvm, SplitData['TestData'])


'''''
### Linear Model
LinearSvmModel = Train(TrainDataRep, SplitData['TrainLabels'], Params, "Linear")
LinearSvmResults = Test(LinearSvmModel,TestDataRep, "Linear")
SummaryByLinearSvm = Evaluate(LinearSvmResults, SplitData['TestLabels'])
ReportResults(SummaryByLinearSvm, SplitData['TestData'])

### Poly Model
PolySvmModel = Train(TrainDataRep, SplitData['TrainLabels'], Params, "poly")
PolySvmResults = Test(PolySvmModel,TestDataRep, "poly")
SummaryByPolySvm = Evaluate(PolySvmResults, SplitData['TestLabels'])
ReportResults(SummaryByPolySvm, SplitData['TestData'])
'''''

'''''
### Test Error Graph By parameters set for each model 
x = ['Default Paramaters', 'First Tuning', 'Second Tuning']
yLinear = [0.40414507772020725, 0.38196891191709844, 0.37823834196891193]
yRbf = [0.533678756476684, 0.36787564766839376, 0.3523316062176166]
yPoly = [0.39896373056994816, 0.3523316062176166, 0.3523316062176166]
plt.plot(x, yLinear)
plt.plot(x, yRbf)
plt.plot(x, yPoly)
plt.xlabel('Tuning Parameters')
plt.ylabel('Test Error')
plt.savefig('TuningResults.png')
plt.close()
'''''

'''''
########## Tuning parameters for all the models- was written in different script ################

def getDefaultParameters():
    s = 200
    m = 16
    n = 16
    bins = 9
    c = 1
    gamma = 0.1
    degree = 3
    trainTestSplit = 20
    trainValSplit = 14
    params = {'S': s, 'M': m, 'N': n, 'Bins': bins, 'C': c, 'Gamma': gamma, 'Degree': degree, 'TrainTestSplit': trainTestSplit, 'TrainValSplit': trainValSplit}
    return params

def InitTuningData(path):
    DandL = {'Data': [], 'Labels': []}
    dirs = os.listdir(path)
    for j in class_indices:
        folder = dirs[j - 1]  # j-1 because dirs start from index 0, and classes start from 1
        folder_path = path + folder
        images = os.listdir(folder_path)
        i = 0
        while i < 20:  ## Loop on images in folder
            img = cv2.imread(path + '/' + folder + "/" + images[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            DandL['Data'].append(img)
            DandL['Labels'].append(j)
            i = i + 1
    return DandL


def TrainValSplit(data, labels, params):
    splitData = {'TrainData': [], 'TrainLabels': [], 'ValData': [], 'ValLabels': []}
    for index in class_indices:
        tempData = []
        for i in range(0, len(data)):
            if index == labels[i]:
                tempData.append(data[i])
        # after we finish adding all imgs from the current class to temp
        for i in range(0, len(tempData)):
            if i < params['TrainValSplit']:
                splitData['TrainData'].append(tempData[i])
                splitData['TrainLabels'].append(index)
            else:
                splitData['ValData'].append(tempData[i])
                splitData['ValLabels'].append(index)
    return splitData


def Prepare(data, params):
    rep = []
    bins = params['Bins']
    m = params['M']
    n = params['N']
    s = params['S']
    for img in data:
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        imRep = skimage.feature.hog(img, orientations=bins, pixels_per_cell=(m, n), cells_per_block=(2, 2),
                                    feature_vector=True)
        rep.append(imRep)
    return rep


def Train(dataRep, labels, params, type):
    model = []
    c = params['C']
    if type == 'Linear':
        model = LinearSVC(dual=False, C=c, multi_class='ovr')
        model.fit(dataRep, labels)
    else:
        model = m_class_SVM_train(dataRep, labels, params, type)
    return model


def m_class_SVM_train(dataRep, labels, params, type):
    modelsList = []
    for j in class_indices:
        tempLabels = []
        for i in range(0, len(labels)):
            if labels[i] == j:
                tempLabels.append(1)
            else:
                tempLabels.append(-1)
        c = params['C']
        gamma = params['Gamma']
        degree = params['Degree']
        if type == 'rbf':
            tempModel = SVC(C=c, kernel='rbf', gamma=gamma)
        else:
            tempModel = SVC(C=c, kernel='poly', degree=degree)
        tempModel.fit(dataRep, tempLabels)
        modelsList.append(tempModel)
    return modelsList


def m_class_SVM_predict(models, dataRep):
    Predictions = []
    ScoresMatrix = np.zeros((len(dataRep), len(models)))
    for j in range(0, len(models)):
        scores = models[j].decision_function(dataRep)
        for i in range(0, len(scores)):
            ScoresMatrix[i][j] = scores[i]
    for i in range(0, len(dataRep)):
        index = np.argmax(ScoresMatrix[i])
        Predictions.append(class_indices[index])
    results = {'Scores': ScoresMatrix, 'Predictions': Predictions}
    return results


def Validation(model, valDataRep, type):
    if type == "Linear":
        results = {'Scores' : model.decision_function(valDataRep), 'Predictions' : model.predict(valDataRep)}
    else:
        results = m_class_SVM_predict(model, valDataRep)
    return results


def findValError(splitData, currentParams, type):
    TrainDataRep = Prepare(splitData['TrainData'], currentParams)
    ValDataRep = Prepare(splitData['ValData'], currentParams)
    if type == "Linear":
        LinearSvmModel = Train(TrainDataRep, splitData['TrainLabels'], currentParams, "Linear")
        LinearSvmValResults = Validation(LinearSvmModel, ValDataRep, "Linear")
        ValidationError = TuningEvaluate(LinearSvmValResults, splitData['ValLabels'])
    else:
        if type == "rbf":
            RbfSvmModel = Train(TrainDataRep, splitData['TrainLabels'], currentParams, "rbf")
            RbfSvmValResults = Validation(RbfSvmModel, ValDataRep, "rbf")
            ValidationError = TuningEvaluate(RbfSvmValResults, splitData['ValLabels'])
        else:
            PolySvmModel = Train(TrainDataRep, splitData['TrainLabels'], currentParams, "poly")
            PolySvmValResults = Validation(PolySvmModel, ValDataRep, "poly")
            ValidationError = TuningEvaluate(PolySvmValResults, splitData['ValLabels'])
    return ValidationError


def TuningEvaluate(results, valLabels):
    predictions = results['Predictions']
    validationAccuracyRate = sklearn.metrics.accuracy_score(valLabels, predictions)
    validationErrorRate = 1 - validationAccuracyRate
    return validationErrorRate


def EqualsDicts(dict1, dict2):
    flag = True
    if dict1['S'] != dict2['S']:
        flag = False
    if dict1['M'] != dict2['M']:
        flag = False
    if dict1['N'] != dict2['N']:
        flag = False
    if dict1['Bins'] != dict2['Bins']:
        flag = False
    if dict1['C'] != dict2['C']:
        flag = False
    if dict1['Gamma'] != dict2['Gamma']:
        flag = False
    return flag


def printPlots(sValuesAndErrors, mValuesAndErrors, nValuesAndErrors, binsValuesAndErrors, cValuesAndErrors, gammaValuesAndErrors, degreeValuesAndErrors, type):
    plt.plot(sValuesAndErrors['Values'], sValuesAndErrors['Errors'])
    plt.xlabel('S values')
    plt.ylabel('Validation Error')
    plt.savefig(str(type) + ' S.png')
    plt.close()

    plt.plot(mValuesAndErrors['Values'], mValuesAndErrors['Errors'])
    plt.xlabel('M values')
    plt.ylabel('Validation Error')
    plt.savefig(str(type) + ' M.png')
    plt.close()

    plt.plot(nValuesAndErrors['Values'], nValuesAndErrors['Errors'])
    plt.xlabel('N values')
    plt.ylabel('Validation Error')
    plt.savefig(str(type) + ' N.png')
    plt.close()

    plt.plot(binsValuesAndErrors['Values'], binsValuesAndErrors['Errors'])
    plt.xlabel('Bins values')
    plt.ylabel('Validation Error')
    plt.savefig(str(type) + ' Bins.png')
    plt.close()

    plt.semilogx(cValuesAndErrors['Values'], cValuesAndErrors['Errors'])
    plt.xlabel('C values')
    plt.ylabel('Validation Error')
    plt.savefig(str(type) + ' C.png')
    plt.close()

    if type == "rbf":
        plt.plot(gammaValuesAndErrors['Values'], gammaValuesAndErrors['Errors'])
        plt.xlabel('Gamma values')
        plt.ylabel('Validation Error')
        plt.savefig(str(type) + ' Gamma.png')
        plt.close()

    if type == "poly":
        plt.plot(degreeValuesAndErrors['Values'], degreeValuesAndErrors['Errors'])
        plt.xlabel('Degree values')
        plt.ylabel('Validation Error')
        plt.savefig(str(type) + ' Degree.png')
        plt.close()


def Tuning(splitData, params, type):
    currentParams = copy.deepcopy(params)
    print("Initial:")
    print("Current Params = " + str(currentParams))
    for iterarion in range(0, 5):
        previousParams = copy.deepcopy(currentParams)
        print("Start Iteration:")
        print("Previous Params = " + str(previousParams))
        print("Current Params = " + str(currentParams))
        sValuesAndErrors = {'Values': range(100, 201, 10), 'Errors': []}
        for s in sValuesAndErrors['Values']:
            currentParams['S'] = s
            ValError = findValError(splitData, currentParams, type)
            sValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = sValuesAndErrors['Errors'].index(min(sValuesAndErrors['Errors']))
        sMinError = sValuesAndErrors['Values'][minErrorIndex]
        currentParams['S'] = sMinError
        print("best s for type " + str(type) + " is " + str(currentParams['S']))

        mValuesAndErrors = {'Values': range(2, 16, 1), 'Errors': []}
        for m in mValuesAndErrors['Values']:
            currentParams['M'] = m
            ValError = findValError(splitData, currentParams, type)
            mValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = mValuesAndErrors['Errors'].index(min(mValuesAndErrors['Errors']))
        mMinError = mValuesAndErrors['Values'][minErrorIndex]
        currentParams['M'] = mMinError
        print("best M for type " + str(type) + " is " + str(currentParams['M']))

        nValuesAndErrors = {'Values': range(8, 30, 1), 'Errors': []}
        for n in nValuesAndErrors['Values']:
            currentParams['N'] = n
            ValError = findValError(splitData, currentParams, type)
            nValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = nValuesAndErrors['Errors'].index(min(nValuesAndErrors['Errors']))
        nMinError = nValuesAndErrors['Values'][minErrorIndex]
        currentParams['N'] = nMinError
        print("best N for type " + str(type) + " is " + str(currentParams['N']))

        binsValuesAndErrors = {'Values': range(4, 21, 1), 'Errors': []}
        for bin in binsValuesAndErrors['Values']:
            currentParams['Bins'] = bin
            ValError = findValError(splitData, currentParams, type)
            binsValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = binsValuesAndErrors['Errors'].index(min(binsValuesAndErrors['Errors']))
        binMinError = binsValuesAndErrors['Values'][minErrorIndex]
        currentParams['Bins'] = binMinError
        print("best Bins for type " + str(type) + " is " + str(currentParams['Bins']))
        print("Validation Error " + str(type) + " is " + str(ValError))

        # cList = np.logspace(-3, 3, 7)
        cList = [0.1, 0.25, 0.5, 1, 2, 3, 4]
        cValuesAndErrors = {'Values': cList, 'Errors': []}
        for c in cValuesAndErrors['Values']:
            currentParams['C'] = c
            ValError = findValError(splitData, currentParams, type)
            cValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = cValuesAndErrors['Errors'].index(min(cValuesAndErrors['Errors']))
        cMinError = cValuesAndErrors['Values'][minErrorIndex]
        currentParams['C'] = cMinError
        print("best c for type " + str(type) + " is " + str(currentParams['C']))
        print("Validation Error " + str(type) + " is " + str(ValError))

        if type == "rbf":
            # gammaList = np.logspace(-4, 0, 5)
            gammaList = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]
            gammaValuesAndErrors = {'Values': gammaList, 'Errors': []}
            for gamma in gammaValuesAndErrors['Values']:
                currentParams['Gamma'] = gamma
                ValError = findValError(splitData, currentParams, type)
                gammaValuesAndErrors['Errors'].append(ValError)
            minErrorIndex = gammaValuesAndErrors['Errors'].index(min(gammaValuesAndErrors['Errors']))
            gammaMinError = gammaValuesAndErrors['Values'][minErrorIndex]
            currentParams['Gamma'] = gammaMinError
            print("best Gamma for type " + str(type) + " is " + str(currentParams['Gamma']))

        if type == "poly":
            degreeValuesAndErrors = {'Values': range(1, 12, 1), 'Errors': []}
            for degree in degreeValuesAndErrors['Values']:
                currentParams['Degree'] = degree
                ValError = findValError(splitData, currentParams, type)
                degreeValuesAndErrors['Errors'].append(ValError)
            minErrorIndex = degreeValuesAndErrors['Errors'].index(min(degreeValuesAndErrors['Errors']))
            degreeMinError = degreeValuesAndErrors['Values'][minErrorIndex]
            currentParams['Degree'] = degreeMinError
            print("best Degree for type " + str(type) + " is " + str(currentParams['Degree']))

        print("End Iteration:")
        print("Previous Params = " + str(previousParams))
        print("Current Params = " + str(currentParams))
        if EqualsDicts(currentParams, previousParams):
            break
    if type == "Linear":
        printPlots(sValuesAndErrors, mValuesAndErrors, nValuesAndErrors, binsValuesAndErrors, cValuesAndErrors,
                   [], [], type)
    else:
        if type == "rbf":
            printPlots(sValuesAndErrors, mValuesAndErrors, nValuesAndErrors, binsValuesAndErrors, cValuesAndErrors,
                       gammaValuesAndErrors, [], type)
        else:
            printPlots(sValuesAndErrors, mValuesAndErrors, nValuesAndErrors, binsValuesAndErrors, cValuesAndErrors,
                       [], degreeValuesAndErrors, type)
    return currentParams


#------------------------------------TuningMain--------------------------

# we started with default parameters and got new parameters after the first tuning.
# after that, we made another tuning with higher resolution in the best range of each parameter and found our best parameters
# LinearParams, RbfParams and PolyParams reflect the parameters after our first tuning for each model
# BestParamsLinear, BestParamsRbf and BestParamsPoly reflect the best parameters for each model (after the second tuning)

data_path = "C:/Users/wr/Downloads/101_ObjectCategories/"
class_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#Params = getDefaultParameters() 
LinearParams = {'S': 100, 'M': 11, 'N': 11, 'Bins': 17, 'C': 1, 'Gamma': 0.1, 'Degree': 2, 'TrainTestSplit': 20, 'TrainValSplit': 14}
RbfParams = {'S': 100, 'M': 11, 'N': 20, 'Bins': 9, 'C': 3, 'Gamma': 0.1, 'Degree': 2, 'TrainTestSplit': 20, 'TrainValSplit': 14}
PolyParams = {'S': 150, 'M': 5, 'N': 14, 'Bins': 14, 'C': 1, 'Gamma': 0.1, 'Degree': 2, 'TrainTestSplit': 20, 'TrainValSplit': 14}
DandL = InitTuningData(data_path)
SplitData = TrainValSplit(DandL['Data'], DandL['Labels'], LinearParams)
#BestParamsLinear = Tuning(SplitData, LinearParams, "Linear")
BestParamsRbf = Tuning(SplitData, RbfParams, "rbf")
#BestParamsPoly = Tuning(SplitData, PolyParams, "poly")

'''''


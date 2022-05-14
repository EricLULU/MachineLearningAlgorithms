from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

def file2matrix(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 生产numberOfLines 行，3列的矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        # 移除头尾的空格和换行符
        listFromLine = line.split('\t')
        returnMat[index:] = listFromLine[0:3]
        # 返回倒数第一个
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, dataSet, label, k):
    dataSetSize = dataSet.shape[0]
    print("dataSetSize:", dataSetSize)
    # 复制相同的行数
    diffMat = tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # 按照第二个维度进行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances ** 0.5
    # 按照从小达到排序，按照索引的顺序返回
    sortDistances = distance.argsort()

    classCount = {}

    for i in range(k):
        voteIlabel = label[sortDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print("sortedClassCount: ", sortedClassCount)
    return sortedClassCount[0][0]


# 归一化数据，使得数组处于0 - 1或者1 - -1之间
def autoNormal(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest(file):
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix(file)
    norMat, ranges, minVals = autoNormal(datingDataMat)
    m = norMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errCount = 0
    for i in range(numTestVecs):
        classifilterResult = classify0(norMat[i, :], norMat[numTestVecs:m, :],
                                       datingLabels[numTestVecs:m], 3)
        print("the classfitier came back with: %d, the real answer is: %d",
              classifilterResult, datingLabels[i])
        if (classifilterResult != datingLabels[i]):
            errCount += 1.0
    print("the total error rate is: %f", errCount / float(numTestVecs))


def img2Vector(fileName):
    returnVnect = zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            # 转为行向量
            returnVnect[0, 32 * i + j] = int(lineStr[j])
    return returnVnect


def handWritingClassTest():
    hwLabels = []
    train_path = r'C:\Users\25893\Documents\GitHub\MachineLearningAlgorithms\machinelearninginaction-master\Ch02\digits\trainingDigits'
    trainingFileList = os.listdir(train_path)

    m = len(trainingFileList)
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)    #标签向量添加
        trainMat[i, :] = img2Vector('{}\\{}'.format(train_path, fileNameStr))

    test_file_path = r'C:\Users\25893\Documents\GitHub\MachineLearningAlgorithms\machinelearninginaction-master\Ch02\digits\testDigits'
    testFileList = os.listdir(test_file_path)
    error_count = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('{}\\{}'.format(test_file_path, fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainMat, hwLabels, 3)
        print("the classfier came back with: {} the real answer is: {}".format(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            error_count += 1.0
        print("the total number of errors is: {}".format(error_count))
        print("the total error rate is: {}".format(error_count/float(mTest)))





if __name__ == '__main__':
    # group,labels = createDataSet()
    # sortedClassCount = classify0([0,0], group,labels, 3)
    # print("sortedClassCount: ", sortedClassCount)
    file = r'C:\Users\25893\Documents\GitHub\MachineLearningAlgorithms\machinelearninginaction-master\Ch02' \
           r'\datingTestSet2.txt'
    # datingDataMat, datingLabels = file2matrix(file)
    # plt.subplot(111)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # # 分别选取第2列和第三列，数据散点图
    # plt.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels),
    #             15.0*array(datingLabels))
    # plt.show()
    # 约会预测
    # datingClassTest(file)

    handWritingClassTest()
    print("run over")
    sys.exit()

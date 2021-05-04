import numpy as np


class Perceptron(object):
    def __init__(self, featuresCount, oneVsRes, iterations=20):
        self.iterations = iterations
        self.weights = np.zeros(featuresCount + 1)
        self.oneVsRes = oneVsRes
        self.predictedClass = []
        self.confidence = []
        self.testLabels = []

    def train(self, features, labels, learningRate):
        for episode in range(self.iterations):
            for feature, label in zip(features, labels):
                # Compute the activation score
                a = np.dot(feature, self.weights[1:]) + self.weights[0]

                # Update the bias term and the weights
                self.weights[0] = self.weights[0] + learningRate * label * self._sigmoid(-label * a)
                self.weights[1:] = np.add(self.weights[1:],
                                          learningRate * label * self._sigmoid(-label * a) * feature)
            print(self.weights)

    # Implement sigmoid function
    @staticmethod
    def _sigmoid(input_):
        # Bound the argument to prevent overflow
        input_ = np.clip(input_, -100, 100)
        return 1 / (1 + np.exp(-input_))

    # predicts class and compares with actual
    def test(self, features, labels):

        for feature, label in zip(features, labels):
            predictedClass, confidence = self._activation(feature)
            self.predictedClass.append(predictedClass)
            self.confidence.append(confidence)

        self.testLabels = labels
        if not oneVsRest:
            results(self.predictedClass, self.testLabels, 0.0)

    def _activation(self, feature):
        weightedSum = np.dot(feature, self.weights[1:]) + self.weights[0]
        if weightedSum > 0:
            predictedClass = 1
            confidence = self._sigmoid(weightedSum)
        else:
            predictedClass = -1
            confidence = 1 - self._sigmoid(weightedSum)
        return predictedClass, confidence


def results(predictedClass, testLabels, label):
    tp, tn, fp, fn = 0, 0, 0, 0
    for predicted, label in zip(predictedClass, testLabels):
        if predicted == label:  # true prediction
            if label > 0:
                tp += 1
            else:
                tn += 1
        else:  # false prediction
            if label > 0:
                fp += 1
            else:
                fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        # if 0/0
        recall = 1.0
    fScore = 2 * precision * recall / (precision + recall)

    print("Accuracy: %.2f" % accuracy)
    print("Precision: %.2f" % precision)
    print("Recall: %.2f" % recall)
    print("F-score: %.2f" % fScore)


def dataPrep(fileName, classA, classB, oneVsRest_):
    data = []
    for line in open(fileName, 'r'):
        ln = np.array(line.strip().replace('class-', '').split(',')).astype(float)

        if ln[-1] == classA:
            ln[-1] = 1
            data.append(ln)
        elif oneVsRest_:  # make the rest of classes equal to 1
            ln[-1] = -1
            data.append(ln)
        elif ln[-1] == classB:  # make only the second class to compare equal to 1
            ln[-1] = -1
            data.append(ln)  # if its class c then it is not added to data

    data = np.asmatrix(data, dtype='float64')

    # reshuffle the data
    permutation = np.random.permutation(len(data))
    data = data[permutation]

    return data[:, :-1], data[:, -1]


def learn(oneVsRest_, classA, classB, learningRate):
    trainingInputs, trainingLabels = dataPrep('train.data', classA, classB, oneVsRest_)
    perceptron = Perceptron(featuresCount=4, oneVsRes=oneVsRest_)
    if oneVsRest_:
        print('Training class %d vs REST with coefficient %f' % (classA, learningRate))
    else:
        print('Training class %d vs class %d...' % (classA, classB))
    print('----BIAS---- ---------------------WEIGHTS---------------------')

    perceptron.train(trainingInputs, trainingLabels, learningRate)

    print('Testing...')
    testInputs, testLabels = dataPrep('test.data', classA,
                                      classB, oneVsRest_)

    perceptron.test(testInputs, testLabels)
    print('--------------------------------------------------------------------------------')
    return perceptron


class1 = 1.0
class2 = 2.0
class3 = 3.0
# one-vs-one approach
oneVsRest = False
learningRt = 0.01
# class 1 vs 2
learn(oneVsRest, class1, class2, learningRt)
# class 2 vs 3
learn(oneVsRest, class2, class3, learningRt)
# class 1 vs 3
learn(oneVsRest, class1, class3, learningRt)

# one-vs-rest approach
oneVsRest = True
classB_ = 0.0
lrnRate = [0.01, 0.1, 1.0, 10.0, 100.0]

for lRate in lrnRate:
    # class1-vs-rest, class2-vs-rest and class3-vs-rest
    perceptronList = [learn(oneVsRest, 1.0, classB_, lRate),
                      learn(oneVsRest, 2.0, classB_, lRate),
                      learn(oneVsRest, 3.0, classB_, lRate)]
    predictedAndActual = []

    # for each test data append the highest confidence prediction
    for c1, c2, c3, p1, p2, p3, t1, t2, t3 in zip(perceptronList[0].confidence, perceptronList[1].confidence,
                                      perceptronList[2].confidence, perceptronList[0].predictedClass,
                                      perceptronList[1].predictedClass, perceptronList[2].predictedClass,
                                      perceptronList[0].testLabels, perceptronList[1].testLabels,
                                      perceptronList[2].testLabels):
        if c1 >= c2 and c1 >= c3:
            # use model Class 1 vs REST for output
            predictedAndActual.append([p1, t1, '1'])
        elif c2 >= c1 and c2 >= c3:
            # use model Class 2 vs REST for output
            predictedAndActual.append([p2, t2, '2'])
        else:
            # use model Class 3 vs REST for output
            predictedAndActual.append([p3, t3, '3'])

    print('One-vs-rest approach results with regularisation coefficient %f :' % lRate)
    predictedAndActual = np.asmatrix(predictedAndActual, dtype='float64')
    results(predictedAndActual[:, 0], predictedAndActual[:, 1], predictedAndActual[:, 2])

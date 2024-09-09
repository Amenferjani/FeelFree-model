import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import copy 
import time
from models import TextClassifier
from tools import Preprocessing, parameter_parser, DatasetMapper

class Run:
    def __init__(self,args) :
        self.__initData__(args)
        self.args = args
        self.batch_size = args.batch_size
        self.model = TextClassifier(args)
    
    def __initData__(self,args):
        self.preprocessing = Preprocessing(args)
        self.preprocessing.loadData()
        self.preprocessing.prepareTokens()

        self.y_train = self.preprocessing.y_train
        self.y_test = self.preprocessing.y_test

        self.x_train = self.preprocessing.text2sequence(
            self.preprocessing.x_train)
        self.x_test = self.preprocessing.text2sequence(
            self.preprocessing.x_test)
        self.y_train += 1
        self.y_test += 1

    def train(self):
        trainingSet = DatasetMapper(self.x_train, self.y_train)
        testSet = DatasetMapper(self.x_test, self.y_test)

        self.trainingLoader = DataLoader(trainingSet,batch_size=self.batch_size)
        self.testLoader = DataLoader(testSet)

        optimizer = optim.RMSprop(
            self.model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        # patience = 2
        # best_loss = float('inf')
        # epochs_without_improvement = 0
        # best_model_weights = copy.deepcopy(self.model.state_dict())
        start = time.time()
        for epoch in range(args.epochs):
            print("********** START **********")
            predictions = []
            self.model.train()
            for x_batch, y_batch in self.trainingLoader:

                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.LongTensor)
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                predictions += list(y_pred.squeeze().detach().numpy())

            testPredictions = self.evaluation()
            trainActuary = self.calculateAccuracy(self.y_train, predictions)
            testAccuracy = self.calculateAccuracy(self.y_test, testPredictions)
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (
                epoch+1, loss.item(), trainActuary, testAccuracy))
        current = time.time()
        total = current - start
        print("********** END **********")
        print(f'\n train took : {total/60} mins')

    def evaluation(self):
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.testLoader:
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.LongTensor)
                y_pred = self.model(x)
                predictions += list(y_pred.detach().numpy())

        return predictions

    def calculateAccuracy(grandTruth, predictions):
        truePositives = 0
        trueNegatives = 0
        for true, pred in zip(grandTruth, predictions):
            if (pred > 0.5) and (true == 1):
                truePositives += 1
            elif (pred < 0.5) and (true == 0):
                trueNegatives += 1
            else:
                pass
        return (truePositives+trueNegatives) / len(grandTruth)

if __name__ == "__main__":
    args = parameter_parser()
    # run = Run(args)
    # run.train()
    print(torch.cuda.is_available())
    print(torch.__version__)

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Model:

    def __init__(self, NumberOfHiddenLayer, NeuronsPerLayers, Epochs, LearnigRate, ActivationFunction, Bias, HasBias):
        self.NumberOfHiddenLayer = int(NumberOfHiddenLayer)
        self.NeuronsPerLayers = NeuronsPerLayers
        self.Epochs = int(Epochs)
        self.LearnigRate = float(LearnigRate)
        self.ActivationFunction = ActivationFunction
        self.HasBias = bool(int(HasBias))
        self.ChoosenActivationFunction = (self.Sigmoid if ActivationFunction == 'Sigmoid' else self.Tanh)
        self.Get_Differentiation = (self.SigmoidDiff if ActivationFunction == 'Sigmoid' else self.TanhDiff)
        self.InputLayerSize = 4
        self.OutputLayerSize = 3
        if self.HasBias:
            self.Bias = float(Bias)
            self.InputLayerSize += 1

        self.FeaturesName = ['SETOSA', 'VERSICOLR', 'VIRGINICA']

    def GetNeuronsSizePerLayer(self):
        return [int(i) for i in self.NeuronsPerLayers.split(',')]

    def PreProcessing(self):
        Data = pd.read_csv("IrisData.txt", sep=",")
        Data = Data.sample(frac=1).reset_index(drop=True)

        self.x_train, self.y_train, self.x_test, self.y_test = self.Train_Test_Split(Data.iloc[0:,:4],Data.iloc[0:,4])

        self.y_train =  self.LabelEncoder(self.y_train)
        self.y_test   = self.LabelEncoder(self.y_test)

        if self.HasBias:
            self.x_test.insert(0,'X0',1)
            self.x_train.insert(0,'X0',1)


    def CreateModel(self):
        self.NeuronsSize = self.GetNeuronsSizePerLayer()

        self.LayersWeight = []

        LastLayerSize = self.InputLayerSize

        for i in range(self.NumberOfHiddenLayer):
            Layer = np.random.uniform(size=(self.NeuronsSize[i],LastLayerSize),low=0.1,high=0.5)
            self.LayersWeight.append(Layer)
            LastLayerSize = self.NeuronsSize[i]

        self.LayersWeight.append(np.random.uniform(size=(self.OutputLayerSize,LastLayerSize),low=0.1,high=0.5))

        if self.HasBias:
            for i in self.LayersWeight:
                i[:-1] = self.Bias

    def Get_Class_Label(self, Class):
        if Class == 'Iris-setosa':
            return [1, 0, 0]
        if Class == 'Iris-versicolor':
            return [0, 1, 0]
        return [0, 0, 1]


    def LabelEncoder(self, Y):
        LabeledData = []
        for Example in Y:
            LabeledData.append(self.Get_Class_Label(Example))

        return LabeledData

    def GetNeuronOutput(self,W,X):
       Net = self.GetNet(W,X)
       Z = self.ChoosenActivationFunction(Net)
       return Z

    def Forward(self,row):
        X = np.array(row).reshape(self.InputLayerSize, 1)
        Zs = []

        for LayerWeights in self.LayersWeight:
            X = self.GetNeuronOutput(LayerWeights, X)
            Zs.append(X)

        return Zs

    def Backward(self,index,Zs):

        Grident = []

        DesiredOutput = self.y_train[index]
        DesiredOutput=np.array(DesiredOutput).reshape((3,1))
        # Output Layer
        LastLayerIndex = len(Zs) - 1
        Error = DesiredOutput - np.array(Zs[LastLayerIndex]).reshape((self.OutputLayerSize,1))
        LastGrident = Error * np.array(self.Get_Differentiation(Zs[LastLayerIndex])).reshape((self.OutputLayerSize,1))

        Grident.append(LastGrident)

        for i in range(LastLayerIndex - 1,  -1 , -1):
            LastGrident = np.dot(self.LayersWeight[i+1].T,LastGrident)
            LastGrident = np.multiply(LastGrident , self.Get_Differentiation(Zs[i]))
            Grident.append(LastGrident)

        Grident.reverse()

        return Grident

    def UpdateWeights(self,Zs):
         for Index,Gredient in enumerate(self.Gridents):
             self.LayersWeight[Index] = self.LayersWeight[Index] + self.LearnigRate * np.dot(Gredient , Zs[Index].T)

    def GetNet(self, W, X):
        return np.dot(W,X)

    def Sigmoid(self,Net):
        return 1 / (1 + np.exp(-Net))

    def Tanh(self,Net):
        Net = np.exp(-Net)
        return (1 - Net) / (1 + Net)


    def SigmoidDiff(self,Net):
        y = Net
        return  np.multiply(y , (1 - y))

    def TanhDiff(self,Net):
        y = Net
        return np.multiply((1 - y),(1 + y))

    def Train(self):
        for i in range(self.Epochs):
            for indx, row in self.x_train.iterrows():
                Zs = self.Forward(row)
                self.Gridents = self.Backward(indx,Zs)
                X = np.array(row).reshape((self.InputLayerSize,1))
                Zs.insert(0,X)
                self.UpdateWeights(Zs)

        self.CaclulateAccuracy(self.x_train,self.y_train)


    def EncodeProb(self,Probabilties):
        Label = []
        mx = max(Probabilties)
        for Probabilty in Probabilties:
            if Probabilty == mx:
                Label.append(1)
            else:
                Label.append(0)
        return  Label

    def GetClassNumber(self,y):
        if y == [1,0,0]:
            return 0
        if y == [0,1,0]:
            return 1
        return 2

    def CaclulateAccuracy(self,X,Y):
        Ans = 0
        OutPutClasses = []
        CoffMat = np.zeros((self.OutputLayerSize,self.OutputLayerSize))

        for indx, row in X.iterrows():
            Output = self.Forward(row)
            OutputLayer = Output[len(Output) - 1]
            OutputClass = self.EncodeProb(OutputLayer)
            Ans += (OutputClass == Y[indx])
            CoffMat[self.GetClassNumber(Y[indx])][self.GetClassNumber(OutputClass)] += 1
            OutPutClasses.append(OutputClass)
            print(f"Prdicted Class {OutputClass}  Class Name: {self.FeaturesName[self.GetClassNumber(OutputClass)]}")


        print("Accuracy: ",(Ans / len(Y)) * 100)

        cm_df = pd.DataFrame(CoffMat,
                             index=self.FeaturesName,
                             columns=self.FeaturesName)

        plt.figure("Backward Propagation", figsize=(5, 4))
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()


    def Test(self):
        self.Train()
        self.CaclulateAccuracy(self.x_test,self.y_test)

    def Train_Test_Split(self, X, Y):

        X_try = X.copy()

        X_train1 = X_try.iloc[:30, ]
        X_test1 = X.iloc[30:50, ]

        X_train2 = X.iloc[50:80, ]
        X_test2 = X.iloc[80:100, ]

        X_train3 = X.iloc[100:130, ]
        X_test3 = X.iloc[130:150, ]

        Y_train1 = Y.iloc[:30, ]
        Y_test1 = Y.iloc[30:50, ]

        Y_train2 = Y.iloc[50:80, ]
        Y_test2 = Y.iloc[80:100, ]

        Y_train3 = Y.iloc[100:130, ]
        Y_test3 = Y.iloc[130:150, ]

        X_train = X_train1.append(X_train2).append(X_train3)
        Y_train = Y_train1.append(Y_train2).append(Y_train3)


        X_test = X_test1.append(X_test2).append(X_test3)
        Y_test = Y_test1.append(Y_test2).append(Y_test3)

        X_train.reset_index(inplace=True, drop=True)
        Y_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        Y_test.reset_index(inplace=True, drop=True)

        return X_train, Y_train, X_test, Y_test

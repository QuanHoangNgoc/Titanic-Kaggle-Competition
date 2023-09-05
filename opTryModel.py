import GlobalVar
import Library
import opProcessFeature
import TryModel
from GlobalVar import *
from Library import *
from opProcessFeature import *
from TryModel import *


def call(D, sizeModel, norm2, norm1, lr, batchSize, epochs) -> None:
    init(
        _saveIndex=str(NP.random.randint(2**31 - 1)),
        _dim=D,
        _size=sizeModel,
        _norm2=norm2,
        _norm1=norm1,
        _lr=lr,
        _bs=batchSize,
        _eps=epochs,
    )

    model = Sequential()
    model = build(model=model)
    model = compile(model=model)
    model.summary()

    history = tryTrain(model=model, X=X, Y=Y, Xcv=Xcv, Ycv=Ycv)
    evaluateTest = model.evaluate(x=Xtest, y=Ytest)

    plotHistory(model=model, history=history, Xtest=Xtest, Ytest=Ytest)
    confusionMatrix(model=model, X=X, Y=Y)

    saveHistory(
        history=history,
        param={
            "sizeModel": sizeModel,
            "L2": norm2,
            "L1": norm1,
            "lr": lr,
            "batchSize": batchSize,
            "epochs": epochs,
        },
        evaluateTest=evaluateTest,
    )
    saveModel(model=model)
    model.summary()


M = X.shape[0]
D = X.shape[1]
sizeModel = [5, 2]
norm2List = [1e-2, 1e-4]
norm1List = [1e-2, 1e-4]
lrList = [1e-3, 1]
bsList = [32, 8]
epsList = [100, 10, 50]


def loop(lis1, lis2, lis3, lis4, lis5):
    for i1 in lis1:
        for i2 in lis2:
            for i3 in lis3:
                for i4 in lis4:
                    for i5 in lis5:
                        call(
                            D=D,
                            sizeModel=sizeModel,
                            norm2=i1,
                            norm1=i2,
                            lr=i3,
                            batchSize=i4,
                            epochs=i5,
                        )
                        print(i1, i2, i3, i4, i5)


loop(norm2List, norm1List, lrList, bsList, epsList)

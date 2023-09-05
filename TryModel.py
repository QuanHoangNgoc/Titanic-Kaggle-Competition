import GlobalVar
import Library
from GlobalVar import *
from Library import *

saveIndex = str(NP.random.randint(2**31 - 1))
D = 11
sizeModel = [5, 2]
norm2 = 1e-2
norm1 = 1e-2
lr = 1e-3
batchSize = 32
epochs = 64


def init(
    _saveIndex: str,
    _dim: int,
    _size: list,
    _norm2: float,
    _norm1: float,
    _lr: float,
    _bs: int,
    _eps: int,
) -> None:
    global saveIndex, sizeModel, D, norm2, norm1, lr, batchSize, epochs

    saveIndex = _saveIndex
    D = _dim
    sizeModel = _size
    norm2 = _norm2
    norm1 = _norm1
    lr = _lr
    batchSize = _bs
    epochs = _eps

    newpath = RUN_FOLDER + "\\" + saveIndex
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def build(model: Sequential) -> Sequential:
    def addLayer(size: int, act: str, l2: float, l1: float, name: str) -> None:
        model.add(
            Dense(
                units=size,
                activation=act,
                kernel_regularizer=regularizers.L1L2(l2=l2, l1=l1),
                name=name,
            )
        )

    model.add(Input(shape=(D,)))
    for layer in range(len(sizeModel)):
        addLayer(
            size=sizeModel[layer], act="relu", l2=norm2, l1=norm1, name=str(layer + 1)
        )
    addLayer(size=1, act="linear", l2=norm2, l1=norm1, name="out")  # ! version
    return model


def compile(model: Sequential) -> Sequential:
    model.compile(
        loss=BinaryCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=lr),
        # 0.001
        metrics=BinaryAccuracy(threshold=thresHold, name="acc"),
    )
    return model


def tryTrain(model: Sequential, X: NP, Y: NP, Xcv: NP, Ycv: NP) -> History:
    return model.fit(
        x=X, y=Y, validation_data=(Xcv, Ycv), epochs=epochs, batch_size=batchSize
    )


def plotHistory(model: Sequential, history: History, Xtest: NP, Ytest: NP) -> None:
    def plotLine(label: str, const: float) -> None:
        line = [const for i in range(len(his["loss"]))]
        plt.plot(line, ls=":", label=label)

    plt.close()
    his = history.history
    evaluateTest = model.evaluate(x=Xtest, y=Ytest)

    for i in range(len(model.metrics_names)):
        plt.subplot(1, len(model.metrics_names), i + 1)
        met = model.metrics_names[i]

        plt.plot(his[met], label=met)
        plt.plot(his["val_" + met], label="val_" + met)
        plotLine("test_" + met, evaluateTest[i])

        if i == 1:
            plotLine("upper_accuracy", upperAccuracy)
            plotLine("lower_accuracy", lowerAccuracy)
        plt.legend()

    plt.savefig(
        RUN_FOLDER + "\\" + saveIndex + "\\plot.png",
        bbox_inches="tight",
    )  #!
    plt.close()


def predict(model: Sequential, X: NP) -> NP:
    yProbability = keras.activations.sigmoid(model.predict(X))
    yHat = NP.where(yProbability >= thresHold, 1, 0)
    return yHat


def confusionMatrix(model: Sequential, X: NP, Y: NP) -> None:
    plt.close()
    predicted = predict(model=model, X=X)
    actual = Y

    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=[False, True]
    )
    cm_display.plot()
    plt.savefig(RUN_FOLDER + "\\" + saveIndex + "\\matrix.png")  #!
    plt.close()


def saveModel(model: Sequential) -> None:
    model.save(RUN_FOLDER + "\\" + saveIndex + "\\model.h5")  #!
    model.save_weights(RUN_FOLDER + "\\" + saveIndex + "\\weights.h5")  #!
    print(f"{saveIndex=}")
    file = open(RUN_FOLDER + "\\saveIndex.txt", "a")  #!
    file.write(saveIndex + "\n")
    file.close()


def saveHistory(history: History, param: dict, evaluateTest: tuple) -> None:
    dic = history.history  # dic
    dic = dic | param
    for x in dic:
        dic[x] = [dic[x]]
    dic["test_loss"] = [evaluateTest[0]]
    dic["test_acc"] = [evaluateTest[1]]
    jsonS = json.dumps(dic)
    file = open(RUN_FOLDER + "\\" + saveIndex + "\\his.json", "w")  #!
    file.write(jsonS)
    file.close()

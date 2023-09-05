import GlobalVar
import Library
from GlobalVar import *
from Library import *
from opProcessFeature import *
from TryModel import build, compile, confusionMatrix, init, plotHistory, predict

final = str(173831686)
model = Sequential()
model = keras.saving.load_model(RUN_FOLDER + "\\" + final + "\\model.h5")


Yctpre = DF(PD.read_csv(RUN_FOLDER + "\\Data\\contest.csv"))
Yctpre = Yctpre.loc[:, ["Survived"]].to_numpy()
# print(Yctpre)
Xtmp = NP.concatenate([X, Xcv, Xct], axis=0)
Ytmp = NP.concatenate([Y, Ycv, Yctpre], axis=0)
Xtmp = NP.concatenate([Xtmp, X, Xcv], axis=0)
Ytmp = NP.concatenate([Ytmp, Y, Ycv], axis=0)


init(
    _saveIndex="99999999999999",
    _dim=Xtmp.shape[1],
    _size=[8, 2],
    _norm2=0.0001,
    _norm1=0.01,
    _lr=0.001,
    _bs=32,
    _eps=100,
)
model = Sequential()
model = build(model=model)
model.compile(
    loss=BinaryCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=1e-1),  # 0.001
    metrics=BinaryAccuracy(threshold=thresHold, name="acc"),
)
history = model.fit(x=Xtmp, y=Ytmp, batch_size=256, epochs=300, validation_split=0.2)
print(Xtmp.shape)
print(Ytmp.shape)
model.summary()
plotHistory(model=model, history=history, Xtest=Xtmp, Ytest=Ytmp)
confusionMatrix(model=model, X=Xtmp, Y=Ytmp)
print("evaluate: (loss, acc)= ", model.evaluate(x=Xtest, y=Ytest))


print(Xct)
print(Xct.shape)
Yct = predict(model=model, X=Xct)
# Yct = model.predict(x=Xct)
# print(Yct)
# print(IDct)
Yct = Yct.reshape((-1,))
IDct = IDct.to_numpy().reshape((-1,))
# print(Yct)
# print(IDct)
df = DF(dict({"PassengerId": IDct, "Survived": Yct}))
# print(df)


# print(predict(model, example))
# print(targetExample)

t = NP.sum(NP.where(Yct == 1, 1, 0))
f = NP.sum(NP.where(Yct == 0, 1, 0))
print("true: ", t)
print("false: ", f)
print("false %:", f / (t + f))


if os.path.exists("contest.csv"):
    os.remove("contest.csv")
df.to_csv("contest.csv", index=False)

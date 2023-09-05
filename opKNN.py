from GlobalVar import *
from Library import *
from opProcessFeature import *

print(example)
print(Xall)


Yctpre = DF(PD.read_csv(RUN_FOLDER + "\\Data\\contest.csv"))
Yctpre = Yctpre.loc[:, ["Survived"]].to_numpy()

Xtmp = NP.concatenate([X, Xcv, Xct], axis=0)
Ytmp = NP.concatenate([Y, Ycv, Yctpre], axis=0)
Xtmp = NP.concatenate([Xtmp, X, Xcv], axis=0)
Ytmp = NP.concatenate([Ytmp, Y, Ycv], axis=0)

knnModel = KNeighborsClassifier(n_neighbors=15, weights="distance")
svc = SVC(random_state=2023)
rdforest = RandomForestClassifier(max_depth=5, n_estimators=500)


kfold = StratifiedKFold(5, shuffle=True, random_state=2023)
crossScore = cross_val_score(estimator=knnModel, X=Xtmp, y=Ytmp, cv=kfold)
print(crossScore)
print(crossScore.mean() * 100, crossScore.std() * 100)
crossScore = cross_val_score(estimator=svc, X=Xtmp, y=Ytmp, cv=kfold)
print(crossScore)
print(crossScore.mean() * 100, crossScore.std() * 100)
crossScore = cross_val_score(estimator=rdforest, X=Xtmp, y=Ytmp, cv=kfold)
print(crossScore)
print(crossScore.mean() * 100, crossScore.std() * 100)


knnModel.fit(X=Xtmp, y=Ytmp)
svc.fit(X=Xtmp, y=Ytmp)
rdforest.fit(X=Xtmp, y=Ytmp)

print("final score: ")
print(knnModel.score(X=Xtest, y=Ytest))
print(svc.score(X=Xtest, y=Ytest))
print(rdforest.score(X=Xtest, y=Ytest))


Yct = rdforest.predict(X=Xct)
Yct = Yct.reshape((-1,))
IDct = IDct.to_numpy().reshape((-1,))
print(Yct)
df = DF(dict({"PassengerId": IDct, "Survived": Yct}))
t = NP.sum(NP.where(Yct == 1, 1, 0))
f = NP.sum(NP.where(Yct == 0, 1, 0))
print("true: ", t)
print("false: ", f)
print("false %:", f / (t + f))


FILE = True
if FILE and os.path.exists("contest.csv"):
    os.remove("contest.csv")
df.to_csv("contest.csv", index=False)

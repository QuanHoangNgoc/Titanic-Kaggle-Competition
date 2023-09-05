import GlobalVar
import Library
import ProcessData
from GlobalVar import *
from Library import *
from ProcessData import *

X, Y, Xct, IDct = createData(
    trainPath="\\Data\\train.csv", testPath="\\Data\\test.csv", targetIndex=1, idIndex=0
)

X, Xcv, Xtest, Y, Ycv, Ytest = splitData(X=X, Y=Y)
Xall = PD.concat([X, Xcv, Xtest, Xct], ignore_index=True)  # !

Xall, X, Xcv, Xtest, Xct = combineColumns(
    arg=[Xall, X, Xcv, Xtest, Xct], col="Combine", col1="SibSp", col2="Parch"
)
Xall, X, Xcv, Xtest, Xct = dropColumns(arg=[Xall, X, Xcv, Xtest, Xct], unCol=unCol)

Xall, X, Xcv, Xtest, Xct = encodeLabel(arg=[Xall, X, Xcv, Xtest, Xct], boolCol=boolCol)
Xall, X, Xcv, Xtest, Xct = encodeOnehot(
    arg=[Xall, X, Xcv, Xtest, Xct], categoryCol=categoryCol
)

# print(DF(Xall).isnull().sum())
# print(DF(Xall))
Xall = NP.concatenate([X, Xcv, Xtest, Xct], axis=0)  # !
Xall, X, Xcv, Xtest, Xct = replaceMissingByKNN(arg=[Xall, X, Xcv, Xtest, Xct])
# print(DF(Xall))

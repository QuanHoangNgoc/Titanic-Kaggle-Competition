import GlobalVar
import Library
import opProcessData
import ProcessFeature
from GlobalVar import *
from Library import *
from opProcessData import *
from ProcessFeature import *

Xall = NP.concatenate([X, Xcv, Xtest, Xct], axis=0)  # !
Xall, X, Xcv, Xtest, Xct = scaling(arg=[Xall, X, Xcv, Xtest, Xct])

# Xall, X, Xcv, Xtest, Xct = scalingMinMax(arg=[Xall, X, Xcv, Xtest, Xct])

# cast matrix
Y = Y.to_numpy()
Ycv = Ycv.to_numpy()
Ytest = Ytest.to_numpy()
example = NP.expand_dims(X[0], axis=0)
targetExample = NP.expand_dims(Y[0], axis=0)

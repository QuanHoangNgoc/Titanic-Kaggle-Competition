import GlobalVar
import Library
from GlobalVar import *
from Library import *


def scaling(arg: list):
    scl = StandardScaler()
    scl.fit(arg[0])
    for i in range(len(arg)):
        arg[i] = scl.transform(arg[i])
    return arg


def scalingMinMax(arg: list) -> list:
    scl = MinMaxScaler()
    scl.fit(arg[0])
    for i in range(len(arg)):
        arg[i] = scl.transform(arg[i])
    return arg

import GlobalVar
import Library
from GlobalVar import *
from Library import *


def createData(trainPath: str, testPath: str, targetIndex: int, idIndex: int) -> any:
    df = DF(PD.read_csv(RUN_FOLDER + trainPath))
    dfct = DF(PD.read_csv(RUN_FOLDER + testPath))
    X = df.drop(columns=[df.columns[targetIndex]])
    Y = df.iloc[:, [targetIndex]]
    Xct = dfct
    IDct = dfct.iloc[:, [idIndex]]
    return X, Y, Xct, IDct


def dupdicateData(X: DF, Y: DF, seed: int = GlobalVar.seed) -> any:
    NP.random.seed = seed
    id1 = NP.where(Y.iloc[:, 0] == 1)[0]

    idsub = NP.random.permutation(id1)
    # log.warning(idsub)
    size = abs(len(X) - 2 * len(id1))
    # log.warning(f"{size=}")
    idsub = idsub[:size]
    # log.warning(f"{idsub=}")

    Xdup = X.iloc[idsub.tolist(), :]
    Ydup = Y.iloc[idsub.tolist(), :]
    X = PD.concat([X, Xdup])
    Y = PD.concat([Y, Ydup])
    return X, Y


def splitData(
    X: DF, Y: DF, percent: list = GlobalVar.percent, seed: int = GlobalVar.seed
) -> any:
    argX, argY = [], []
    xtmp, ytmp = X, Y
    for p in percent:
        x, xtmp, y, ytmp = train_test_split(
            xtmp, ytmp, test_size=1 - p, random_state=seed
        )
        argX += [x]
        argY += [y]
    argX += [xtmp]
    argY += [ytmp]
    return argX + argY


def combineColumns(arg: list, col: str, col1: str, col2: str) -> any:
    for i in range(len(arg)):
        arg[i][col] = arg[i][col1] + arg[i][col2]
    return arg


def dropColumns(arg: list, unCol: list) -> any:
    for i in range(len(arg)):
        arg[i] = arg[i].drop(columns=unCol)
    return arg


def encodeLabel(arg: list, boolCol: list) -> any:
    le = LabelEncoder()

    def encodeTransform(df: DF, col: list) -> any:
        df.loc[:, col] = le.transform(df.loc[:, col])
        return df

    for col in boolCol:
        le.fit(arg[0].loc[:, col])
        for i in range(len(arg)):
            arg[i] = encodeTransform(df=arg[i], col=col)
    return arg


def plotDistribution(df: DF, col: str, Y: DF) -> None:
    plt.close()
    plt.hist(df.loc[:, col], color="c")
    plt.hist(df.loc[Y.iloc[:, 0] == 1, col])
    plt.show()


def replaceNull(arg: list, quanCol: list) -> any:
    impute = SimpleImputer(missing_values=NP.NaN, strategy="mean")
    impute.fit(X=arg[0].loc[:, quanCol])
    for i in range(len(arg)):
        arg[i].loc[:, quanCol] = impute.transform(arg[i].loc[:, quanCol])
    return arg


def replaceMissingByKNN(arg: list) -> any:
    impute = KNNImputer(n_neighbors=5)
    impute.fit(X=arg[0])
    for i in range(len(arg)):
        arg[i] = impute.transform(arg[i])
    return arg


def encodeOnehot(arg: list, categoryCol: list) -> any:
    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(), categoryCol)],
        remainder="passthrough",
    )
    ct.fit(X=arg[0])
    for i in range(len(arg)):
        arg[i] = ct.transform(X=arg[i])
    return arg


def encodeOnehotSuper(arg: list, category: list, percent: float) -> any:
    pass

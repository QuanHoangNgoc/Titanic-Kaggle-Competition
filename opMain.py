import GlobalVar
import Library

# import opTryModel
from GlobalVar import *
from Library import *

# from opTryModel import *

indexList = [s for s in open(RUN_FOLDER + "\\saveIndex.txt", "r").readline().split()]
df = DF()
for index in indexList:
    file = open(RUN_FOLDER + "\\" + index + "\\his.json")
    row = json.load(file)  # to dict
    row = DF(row)  # to DF
    # print(row) # !
    df = PD.concat([df, row], ignore_index=True)  # add
length = 10


def repeat(number: float, length: int) -> list:
    return [number] * length


def plot(df: DF, name: str) -> None:
    subdf = df.loc[:, [name]]
    lis, lis2 = [], []
    for i in range(len(subdf)):
        v = subdf.iloc[i][0]
        if isinstance(v, float):
            lis += repeat(v, length=length)  # !!!
        else:
            lis += v[-length:]  # !!!
            lis2 += repeat(NP.mean(v[-length:]), length=length)  # !!!
    plt.plot(lis, label=name)
    if len(lis2) > 0:
        plt.plot(lis2, label="mean_" + name)


def draw(df: DF) -> None:
    plt.subplot(2, 2, 3)
    # print(df.loc[:, "regul"].apply(lambda x: x[0])) # series
    plt.plot(df.loc[:, "regul"].apply(lambda x: x[0]), label="regul")
    plt.legend()
    plt.grid(axis="both")

    plt.subplot(2, 2, 4)
    plt.plot(df.loc[:, "lrAdapt"].apply(lambda x: x), label="lrAdapt")
    plt.legend()
    plt.grid(axis="both")

    # plt.subplot(2, 2, 5)
    # plt.plot(df.loc[:, "batchSize"].apply(lambda x: x), label="batchSize")
    # plt.legend()
    # plt.grid(axis="both")

    # plt.subplot(2, 2, 6)
    # plt.plot(df.loc[:, "epochs"].apply(lambda x: x), label="epochs")
    # plt.legend()
    # plt.grid(axis="both")


plt.close()
plt.subplot(2, 2, 1)
plot(df=df, name="loss")
plot(df=df, name="val_loss")
plot(df=df, name="test_loss")
plt.legend()
plt.grid(axis="both")

plt.subplot(2, 2, 2)
plot(df=df, name="acc")
plot(df=df, name="val_acc")
plot(df=df, name="test_acc")
# plt.legend()
plt.grid(axis="both")

draw(df=df)
plt.show()
plt.close()

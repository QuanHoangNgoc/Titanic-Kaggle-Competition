from Library import *

RUN_FOLDER = os.path.dirname(os.path.abspath(__file__))
log.warning(RUN_FOLDER)
LINELINE = ""
for cnt in range(100):
    LINELINE += "-"
log.warning(LINELINE)


percent = [0.8, 0.5]
seed = NP.random.randint(101)
unCol = ["Name", "Ticket", "Cabin", "SibSp", "Parch"]
boolCol = ["Sex"] + ["Pclass", "Embarked"]
categoryCol = ["Pclass", "Embarked"]
# quanCol = ["Age", "SibSp", "Parch", "Fare"]


thresHold = 0.5
upperAccuracy = 0.92
lowerAccuracy = 0.82

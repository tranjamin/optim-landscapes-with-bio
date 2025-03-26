import sys
sys.path.insert(1, './Modules')

from Modules import DataManager
from Modules import Regression
from Modules import GenericModel
import pandas as pd
import time
from sklearn.metrics import accuracy_score


df = pd.read_csv("../datasets/heart.csv")
data = DataManager.Data(df)

data.make_categorical("HeartDisease")
data.normalise_numeric()

train_data, test_data = data.train_test_split(split_ratio=0.8)

train_x, train_y = train_data.input_output_split("HeartDisease")
test_x, test_y = test_data.input_output_split("HeartDisease")

for i in range(1, 2**11):
    stime = time.time()
    bitmask = [bool(int(b)) for b in bin(i)[2:].zfill(len(train_x.columns))]
    masked_train_x = DataManager.Data(train_x.get_data().loc[:, bitmask])
    masked_test_x = DataManager.Data(test_x.get_data().loc[:, bitmask])

    masked_train_x = masked_train_x.encode_categoricals()
    masked_test_x = masked_test_x.encode_categoricals()

    model = Regression.BinaryRegression()
    model.set_err_function(accuracy_score)
    model.add_training_data(masked_train_x, train_y)
    model.add_testing_data(masked_test_x, test_y)
    model.fit()

    score = model.evaluate_testing_performance()

    print(f"Model {i}: Misclass: {1 - score} | Eval time: {time.time() - stime} seconds")


### Models
## SVM
## NN
## Lin Regression
## RF
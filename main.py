from getdata import get_data
from processor import get_processed
from backend import Model
import pandas as pd

train_data = get_data("train.csv")
X, y = get_processed(train_data)

model = Model(X, y, 0.0015, 2501)

model.fit()

test_data = get_data("test.csv")
X_test = get_processed(test_data, train=False)

preds = model.predict(X_test)

result = pd.DataFrame(data = [test_data["Id"], preds], columns=["Id", "SalePrice"])
result.to_csv("result.csv")

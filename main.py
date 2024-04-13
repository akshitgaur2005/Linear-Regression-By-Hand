from getdata import get_data
from processor import get_processed
from backend import Model
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

train_data = pd.read_csv("./data-folder/train.csv")
train_data.drop(columns = ["Alley", "PoolQC", "MiscFeature", "Fireplaces", "MasVnrType", "FireplaceQu", "Fence"], inplace=True)
X_train = train_data.drop(columns = ["Id", "SalePrice"])
y_train = train_data["SalePrice"].to_numpy()

print(X_train.shape)
enc.fit(X_train)
X_train = enc.transform(X_train)
print(X_train.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_train)

model = Model(X_scaled, y_train, 0.002, 2500)

model.fit()

test_data = pd.read_csv("./data-folder/test.csv")
ids = test_data["Id"]
test_data.drop(columns = ["Alley", "PoolQC", "MiscFeature", "Fireplaces", "MasVnrType", "FireplaceQu", "Fence"], inplace=True)
X_test = test_data.drop(columns = ["Id"])

X_test = enc.transform(X_test)
X_test_scaled = scaler.transform(X_test)

preds = model.predict(X_test_scaled)

result = pd.DataFrame({
    "Id": ids,
    "SalePrice": preds.reshape(-1,)
    })

result.to_csv("result.csv", index = False)

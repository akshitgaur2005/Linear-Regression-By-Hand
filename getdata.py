import pandas as pd

data = pd.read_csv("./data-folder/train.csv")

data.drop(columns = ["Alley", "PoolQC", "MiscFeature", "Fireplaces"], inplace=True)

data = pd.get_dummies(data, dtype=int, dummy_na=True)
data.drop(columns= ["Neighborhood_Blueste"], inplace=True)
data.fillna({"LotFrontage": data["LotFrontage"].mode()[0], "MasVnrArea": data["MasVnrArea"].mode()[0], "GarageYrBlt": data["GarageYrBlt"].mode()[0]}, inplace=True)

def get_data():
    return data

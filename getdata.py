import pandas as pd

def get_data(filename):
    data = pd.read_csv(f"./data-folder/{filename}")
    
    data.drop(columns = ["Alley", "PoolQC", "MiscFeature", "Fireplaces", "MasVnrType", "FireplaceQu", "Fence"], inplace=True)

    data = pd.get_dummies(data, dtype=float, dummy_na=True)
    data.drop(columns= ["Neighborhood_Blueste"], inplace=True)
    data.fillna({"LotFrontage": data["LotFrontage"].mode()[0], "MasVnrArea": data["MasVnrArea"].mode()[0], "GarageYrBlt": data["GarageYrBlt"].mode()[0]}, inplace=True)
    
    columns = pd.Series(data.columns)
    columns.to_csv(filename)
    #print(columns)

    return data

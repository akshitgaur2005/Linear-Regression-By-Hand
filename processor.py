from sklearn import preprocessing

def get_processed(data):
    X = data.drop(columns = ["SalePrice"])
    y = data["SalePrice"]

    scaler = preprocessing.StandardScaler()

    X_scaled = scaler.fit_transform(X)
    y_numpy = y.to_numpy().reshape(-1, 1)

    return X_scaled, y_numpy

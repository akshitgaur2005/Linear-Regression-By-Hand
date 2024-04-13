from sklearn import preprocessing

def get_processed(data, train=True):
    if train:
        X = data.drop(columns = ["Id", "SalePrice"])
        y = data["SalePrice"]
        y_numpy = y.to_numpy().reshape(-1, 1)
    else:
        X = data.drop(columns = ["Id"])

    scaler = preprocessing.StandardScaler()

    X_scaled = scaler.fit_transform(X)
    
    if train:
        return X_scaled, y_numpy
    else:
        return X_scaled

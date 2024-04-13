from getdata import get_data
from processor import get_processed

def main():
    train_data = get_data("train.csv")
    X, y = get_processed(train_data)
    print(f"X: {X}\n y: {y}")


main()

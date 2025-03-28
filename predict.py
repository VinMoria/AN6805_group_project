import pickle
from MyTool import MyTool
import os
from pathlib import Path


def predict_all_models(i, model):
    X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()
    new_test_case = X_test.iloc[i:i+1].copy()  # 取测试集中的第一行作为新测试用例
    print("\nCase:\n")

    for idx, (col_name, value) in enumerate(new_test_case.items(), start=1):
        print(f"{idx}. {col_name}: {value.iloc[0]}")

    model_file = f"models/{model}_model.pkl"
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    # 预测新测试用例的分类概率
    probabilities = model.predict_proba(new_test_case)
    print(f"P(Placed): {probabilities[0][1]*100:.2f}%")

    while True:
        user_input = input("Change (?) to (?):")
        if user_input == "Q":
            break
        index, val = user_input.split()
        index = int(index) - 1
        new_test_case.iloc[0, index] = val
        for idx, (col_name, value) in enumerate(new_test_case.items(), start=1):
            print(f"{idx}. {col_name}: {value.iloc[0]}")
            probabilities = model.predict_proba(new_test_case)
        print(f"P(Placed): {probabilities[0][1]*100:.2f}%")









if __name__ == "__main__":
    predict_all_models(45,"KNN")


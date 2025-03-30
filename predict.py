import pickle
from MyTool import MyTool
import os
from pathlib import Path
import pandas as pd


def predict_all_models(i, model):
    X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()
    new_test_case = X_test.iloc[i : i + 1].copy()  # 取测试集中的第一行作为新测试用例
    print("\nCase:")

    for idx, (col_name, value) in enumerate(new_test_case.items(), start=1):
        print(f"{idx}. {col_name}: {value.iloc[0]}")

    model_file = f"models/{model}_model.pkl"
    with open(model_file, "rb") as file:
        model = pickle.load(file)

    # 预测新测试用例的分类概率
    probabilities = model.predict_proba(new_test_case)
    print(f"\nP(Placed): {probabilities[0][1]*100:.2f}%")

    while True:
        user_input = input(
            "----------------------\n Press 'Q' to Quit\nChange (?) to (?):"
        )
        if user_input == "Q":
            break
        index, val = user_input.split()
        index = int(index) - 1

        # 获取原始数据类型并转换 val
        original_value = new_test_case.iloc[0, index]
        if pd.api.types.is_integer_dtype(original_value):
            val = int(val)
        elif pd.api.types.is_float_dtype(original_value):
            val = float(val)
        elif pd.api.types.is_bool_dtype(original_value):
            val = bool(val)
        else:
            val = str(val)  # 默认转为字符串

        new_test_case.iloc[0, index] = val
        for idx, (col_name, value) in enumerate(new_test_case.items(), start=1):
            print(f"{idx}. {col_name}: {value.iloc[0]}")
            probabilities = model.predict_proba(new_test_case)
        print(f"\nP(Placed): {probabilities[0][1]*100:.2f}%")


if __name__ == "__main__":
    predict_all_models(1, "XGB")

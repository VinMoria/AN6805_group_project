import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


class MyTool:
    def getdata():
        # 读取数据
        df = pd.read_csv("placementdata.csv")

        # 目标变量编码（Placed -> 1, NotPlaced -> 0）
        label_encoder = LabelEncoder()
        df["PlacementStatus"] = label_encoder.fit_transform(df["PlacementStatus"])

        # 分离特征和目标变量
        X = df.drop(["PlacementStatus", "StudentID"], axis=1)
        y = df["PlacementStatus"]

        # 检查并处理分类特征
        categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

        # 使用ColumnTransformer对分类列进行独热编码
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(), categorical_cols),
                ("num", "passthrough", numeric_cols),
            ]
        )

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, preprocessor

    def save(model, name):
        # 保存模型到 models 文件夹
        os.makedirs("models", exist_ok=True)  # 确保文件夹存在
        model_path = os.path.join("models", f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")

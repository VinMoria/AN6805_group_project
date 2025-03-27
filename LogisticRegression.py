import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

# 读取数据
df = pd.read_csv("placementdata.csv")

# 目标变量编码（Placed -> 1, NotPlaced -> 0）
label_encoder = LabelEncoder()
df['PlacementStatus'] = label_encoder.fit_transform(df['PlacementStatus'])

# 分离特征和目标变量
X = df.drop('PlacementStatus', axis=1)
y = df['PlacementStatus']

# 检查并处理分类特征
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# 使用ColumnTransformer对分类列进行独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 预处理数据（独热编码分类变量）
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 创建并训练逻辑回归模型
model = LogisticRegression(max_iter=1000)  # 增加max_iter避免收敛警告
model.fit(X_train_processed, y_train)

# 预测并评估模型
y_pred = model.predict(X_test_processed)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
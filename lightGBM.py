import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

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

# 定义LightGBM模型
lgb_model = lgb.LGBMClassifier(random_state=42, objective='binary')

# 定义超参数网格
param_grid = {
    'n_estimators': [50, 100, 150],          # 树的数量
    'learning_rate': [0.01, 0.1, 0.2],       # 学习率
    'max_depth': [3, 5, 7],                  # 树的最大深度
    'num_leaves': [15, 31, 63],              # 叶子节点数
    'min_child_samples': [10, 20, 30]        # 叶子节点最小样本数
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    scoring='accuracy',  # 评估指标
    cv=5,                # 5折交叉验证
    n_jobs=-1,           # 使用所有CPU核心
    verbose=1            # 输出详细日志
)

# 执行网格搜索
grid_search.fit(X_train_processed, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)

# 使用最佳模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_processed)

# 评估模型
print("\n测试集准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
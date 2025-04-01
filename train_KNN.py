from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from MyTool import MyTool
import seaborn as sns
import matplotlib.pyplot as plt

# 划分训练集和测试集
X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()

# 创建KNN模型管道，包含预处理和标准化步骤
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),  # 独热编码后不需要中心化
        ("knn", KNeighborsClassifier()),
    ]
)

# 定义GridSearchCV的参数网格
param_grid = {
    "knn__n_neighbors": [78, 79, 80, 81, 82],  # 不同的K值
    "knn__metric": ["euclidean", "manhattan"],
    "knn__weights": ["uniform", "distance"],  # 是否加权
    "scaler": [StandardScaler(with_mean=False), "passthrough"],  # 是否归一化
}

# 使用GridSearchCV进行超参数调优
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", verbose=1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 评估模型
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

MyTool.save(best_model, "KNN_model")  # 保存模型

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Best Parameters: {'knn__metric': 'euclidean', 'knn__n_neighbors': 79, 'knn__weights': 'uniform', 'scaler': StandardScaler(with_mean=False)}

# Accuracy: 0.795

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.82      0.83      0.83      1172
#            1       0.75      0.75      0.75       828

#     accuracy                           0.80      2000
#    macro avg       0.79      0.79      0.79      2000
# weighted avg       0.79      0.80      0.79      2000

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler  # 新增导入
import matplotlib.pyplot as plt
from MyTool import MyTool
import seaborn as sns
from sklearn.pipeline import Pipeline
import numpy as np

X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()

# 修改后的Pipeline（新增StandardScaler）
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),  # 标准化层
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# 预测并评估模型
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 获取特征名称
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names = np.array(X_train.columns)

# 获取标准化后的系数（此时可直接比较）
model = pipeline.named_steps['model']
coefficients = model.coef_[0]

# 特征重要性排序（无需再处理量纲）
feature_importance = sorted(
    zip(feature_names, coefficients),
    key=lambda x: abs(x[1]),
    reverse=True
)

print("\nTop 10 most important features (标准化系数):")
for feature, coef in feature_importance[:10]:
    print(f"{feature}: {coef:.4f}")

MyTool.save(pipeline, "logistic_regression_model")

# Accuracy: 0.7945

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.83      0.81      0.82      1172
#            1       0.74      0.77      0.76       828

#     accuracy                           0.79      2000
#    macro avg       0.79      0.79      0.79      2000
# weighted avg       0.80      0.79      0.79      2000
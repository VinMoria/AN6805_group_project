from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from MyTool import MyTool

# 获取数据和预处理器
X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()

# 创建Pipeline，使用神经网络模型
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MLPClassifier(random_state=42, max_iter=300))
])

# 训练模型
pipeline.fit(X_train, y_train)

# 预测并评估
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 保存模型
MyTool.save(pipeline, "NeuralNetwork_model")

# Accuracy: 0.791

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.81      0.83      0.82      1172
#            1       0.76      0.73      0.74       828

#     accuracy                           0.79      2000
#    macro avg       0.79      0.78      0.78      2000
# weighted avg       0.79      0.79      0.79      2000
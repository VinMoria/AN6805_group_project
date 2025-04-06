from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
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


# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 计算精度、召回率（可针对每个类别输出）
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
accuracy = accuracy_score(y_test, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0','Class 1'],
            yticklabels=['Class 0','Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('nnmatrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印关键指标
print("\n================ Metrics ================")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision for Class 0: {:.2f}".format(precision[0]))
print("Precision for Class 1: {:.2f}".format(precision[1]))
print("Recall for Class 0: {:.2f}".format(recall[0]))
print("Recall for Class 1: {:.2f}".format(recall[1]))
print("Random State:", 42)
print("Max Iter:", 300)


# Accuracy: 0.791

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.81      0.83      0.82      1172
#            1       0.76      0.73      0.74       828

#     accuracy                           0.79      2000
#    macro avg       0.79      0.78      0.78      2000
# weighted avg       0.79      0.79      0.79      2000
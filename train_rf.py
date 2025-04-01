from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from MyTool import MyTool
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()

# 创建Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])



pipeline.fit(X_train, y_train)

# 预测并评估模型
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

MyTool.save(pipeline, "RandomForest_model")

# Accuracy: 0.781

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.83      0.82      1172
#            1       0.75      0.71      0.73       828

#     accuracy                           0.78      2000
#    macro avg       0.78      0.77      0.77      2000
# weighted avg       0.78      0.78      0.78      2000
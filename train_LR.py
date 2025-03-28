from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from MyTool import MyTool
from sklearn.pipeline import Pipeline


X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()

# 创建Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000) )
])


pipeline.fit(X_train, y_train)

# 预测并评估模型
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

MyTool.save(pipeline, "logistic_regression_model")
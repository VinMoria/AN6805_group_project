from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from MyTool import MyTool
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 划分训练集和测试集
X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()

# 创建Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(random_state=42))
])

# 定义参数网格
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)

# 使用最佳模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 评估模型
print("\n测试集准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))

# 获取特征重要性
xgb_model = best_model.named_steps['model']
feature_importances = xgb_model.feature_importances_

# 获取特征名称
try:
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
except AttributeError:
    # 如果preprocessor没有get_feature_names_out方法，使用默认的列名
    feature_names = np.array(X_train.columns)

# 按重要性排序（从高到低）
sorted_idx = np.argsort(feature_importances)
sorted_importances = feature_importances[sorted_idx]
sorted_feature_names = feature_names[sorted_idx]

# 绘制排序后的特征重要性
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
plt.yticks(range(len(sorted_importances)), sorted_feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('XGBoost Feature Importance (Sorted)')

# 在每个柱状条旁边显示具体数值
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01,  # x 坐标（数值右侧）
             bar.get_y() + bar.get_height() / 2,  # y 坐标（柱状条中心）
             f'{width:.4f}',  # 显示4位小数
             ha='left', va='center')

plt.tight_layout()
plt.show()


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

# 保存模型
MyTool.save(best_model, "XGB_model")

# 最佳参数组合: {'model__colsample_bytree': 0.8, 'model__learning_rate': 0.1, 'model__max_depth': 3, 'model__n_estimators': 50, 'model__subsample': 0.8}

# 测试集准确率: 0.7965

# 分类报告:
#                precision    recall  f1-score   support

#            0       0.82      0.83      0.83      1172
#            1       0.76      0.74      0.75       828

#     accuracy                           0.80      2000
#    macro avg       0.79      0.79      0.79      2000
# weighted avg       0.80      0.80      0.80      2000
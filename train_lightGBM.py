from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
from MyTool import MyTool
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()

# 创建Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lgb.LGBMClassifier(random_state=42, objective='binary'))
])

# 定义超参数网格
param_grid = {
    'model__n_estimators': [50, 100, 150],          # 树的数量
    'model__learning_rate': [0.01, 0.1, 0.2],       # 学习率
    'model__max_depth': [3, 5, 7],                  # 树的最大深度
    'model__num_leaves': [15, 31, 63],              # 叶子节点数
    'model__min_child_samples': [10, 20, 30]        # 叶子节点最小样本数
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',  # 评估指标
    cv=5,                # 5折交叉验证
    n_jobs=-1,           # 使用所有CPU核心
    verbose=1            # 输出详细日志
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

MyTool.save(best_model, "lightGBM_model")


# 最佳参数组合: {'model__learning_rate': 0.1, 'model__max_depth': 3, 'model__min_child_samples': 30, 'model__n_estimators': 100, 'model__num_leaves': 15}

# 测试集准确率: 0.793

# 分类报告:
#                precision    recall  f1-score   support

#            0       0.82      0.83      0.82      1172
#            1       0.76      0.74      0.75       828

#     accuracy                           0.79      2000
#    macro avg       0.79      0.79      0.79      2000
# weighted avg       0.79      0.79      0.79      2000



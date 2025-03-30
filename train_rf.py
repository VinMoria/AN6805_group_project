from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from MyTool import MyTool
import pandas as pd

# 获取数据和预处理器
X_train, X_test, y_train, y_test, preprocessor = MyTool.getdata()

# 创建Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# 定义参数网格
param_grid = {
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

# 使用GridSearchCV进行调参
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Training Accuracy:", grid.best_score_)

# 用最优模型进行预测评估
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 提取并展示特征重要性排名
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
importances = best_model.named_steps['model'].feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
print("\nTop 10 Feature Importances:")
print(feat_imp_df.head(10))

# 保存模型
MyTool.save(best_model, "RandomForest_model_Optimized")

#Best Parameters: {'model__max_depth': 10, 'model__min_samples_split': 2, 'model__n_estimators': 150}
#Best Training Accuracy: 0.7986266814172609
#Test Accuracy: 0.787

#Classification Report:
#                precision    recall  f1-score   support

#            0       0.81      0.83      0.82      1172
#            1       0.75      0.73      0.74       828

#     accuracy                           0.79      2000
#    macro avg       0.78      0.78      0.78      2000
# weighted avg       0.79      0.79      0.79      2000


# Top 10 Feature Importances:
#                                Feature  Importance
# 11                      num__HSC_Marks    0.191351
# 8               num__AptitudeTestScore    0.184326
# 10                      num__SSC_Marks    0.099736
# 6                        num__Projects    0.093452
# 4                            num__CGPA    0.090799
# 1   cat__ExtracurricularActivities_Yes    0.079739
# 0    cat__ExtracurricularActivities_No    0.072304
# 9                num__SoftSkillsRating    0.070181
# 7        num__Workshops/Certifications    0.061179
# 5                     num__Internships    0.021618

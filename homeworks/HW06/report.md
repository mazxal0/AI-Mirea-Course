# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-04.csv`
- Размер: (25000, 62)
- Целевая переменная: `target` (0: 23770 и 1: 1230)
- Признаки: все числовые

## 2. Protocol

- Разбиение: train/test (0.2, `random_state=42`)
- Подбор: CV на train (5 fold, grid_params like max_depth, min_samples_leaf, max_features and etc)
- Метрики: accuracy, F1, ROC-AUC (ROC-AUC так как работаем с вероятностями, accuracy помогает сравнивать модели с практически равными значениями ROC-AUC, F1 нужен что бы посмотреть на соотношение recall и precision)

## 3. Models

Опишите, какие модели сравнивали и какие гиперпараметры подбирали.

Минимум:

- DummyClassifier (baseline)
- LogisticRegression (baseline из S05)
- DecisionTreeClassifier (контроль сложности: `max_depth` + `min_samples_leaf` + `ccp_alpha`)
- RandomForestClassifier (контроль сложности: `max_depth` + `min_samples_leaf` + `max_features`)
- HistGradientBoosting (контроль сложности: `max_depth` + `learning_rate` + `max_leaf_nodes`)
- StackingClassifier(estimators=[`LogisticRegression`, `RandomForest`, `HistGradientBoostingClassifier`] и `final_estimator=LogisticRegression`)

## 4. Results

| Model                         | Accuracy | Precision | Recall | F1-score | ROC AUC |
|-------------------------------|----------|-----------|--------|----------|---------|
| RandomForest                  | 0.9776   | 0.8641    | 0.6463 | 0.7395   | 0.8205  |
| StackingClassifier            | 0.9786   | 0.8927    | 0.6423 | 0.7470   | 0.8191  |
| HistGradientBoostingClassifier| 0.9760   | 0.9500    | 0.5407 | 0.6891   | 0.7696  |
| DecisionTreeClassifier        | 0.9648   | 0.7966    | 0.3821 | 0.5165   | 0.6885  |
| LogisticRegression            | 0.9632   | 0.9079    | 0.2805 | 0.4286   | 0.6395  |
| DummyClassifier               | 0.9508   | 1.0000    | 0.0000 | 0.0000   | 0.5000  |
- Победитель RandomForest (по AUC-ROC) так как имеет наилучшее значение, чем остальные модели + RandomForest имеет меньше настраиваемых параметров, чем Stacking и поэтому RandomForest дает наилучшее показатели при сильном дисбалансе

## 5. Analysis

- Устойчивость: модель RandomForest очень устойчива к изменению RANDOM_STATE
- Ошибки: confusion matrix расположена по адресу `'./artifacts/figures/RandomForest_metrics/confusion_matrix.png'`, самое лучшее f1-score у StackingClassifier, но ROC-AUC у RandomForest лучше  
- Самые главные признаки это f54, f53 и f13, которые делают самый наибольший вклад

## 6. Conclusion

В данном домашнем задании я смог выучить и изучить множество новых моделей, которые по разному работают:
- Decision Tree легко переобучается
- RandomForest помогает уйти от переобучения за счёт усреднения множества простых деревьев
- Ансамбли могут помочь лучше, чем другие модели если в данных есть сильная нелинейная зависимость 
Я смог реализовать полный цикл ML-протокола, который вывел лучшую модель по нужной метрике в моей задаче(ROC-AUC)
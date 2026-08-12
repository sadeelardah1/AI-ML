import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,)
from sklearn.metrics import f1_score


# 2. Load Dataset
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
dataset_path = BASE_DIR / "train_and_test2.csv"
titanic = pd.read_csv(dataset_path)


# اسم الـ target في الملف الأصلي هو 2urvived
# نغيره إلى اسم أوضح وأسهل للقراءة

titanic = titanic.rename(columns={"2urvived": "Survived"})
print("Dataset shape:", titanic.shape)
print("\nFirst 5 rows:")
print(titanic.head())


# 3. Feature Engineering


# Feature Engineering معناها:
# إنشاء أو تحويل Features بحيث نعطي الموديل معلومات أوضح وأفيد.
# نعمل نسخة حتى ما نغير الـ DataFrame الأصلي مباشرة
titanic_fe = titanic.copy()



# 3.1 Creating a New Feature: family_size
# sibsp = عدد الإخوة / الأزواج الموجودين مع الشخص
# Parch = عدد الآباء / الأبناء الموجودين مع الشخص
# نضيف 1 حتى نحسب الشخص نفسه.
# مثال:
# sibsp = 2
# Parch = 1
# family_size = 2 + 1 + 1 = 4
titanic_fe["family_size"] = (
    titanic_fe["sibsp"]
    + titanic_fe["Parch"]
    + 1
)


# 3.2 Creating a New Feature: fare_per_person
# Fare ممكن يكون سعر التذكرة لمجموعة أو عائلة كاملة.
# لذلك نقسم السعر على family_size
# حتى نحصل على تكلفة تقريبية لكل شخص.
titanic_fe["fare_per_person"] = (titanic_fe["Fare"] / titanic_fe["family_size"])



# 3.3 Creating a Binary Feature: is_alone
# إذا كان family_size = 1
# فهذا يعني أن الشخص يسافر لوحده.
# True يتحول إلى 1
# False يتحول إلى 0
titanic_fe["is_alone"] = (titanic_fe["family_size"] == 1).astype(int)


print("\nEngineered Features:")
print(
    titanic_fe[
        [
            "sibsp",
            "Parch",
            "family_size",
            "Fare",
            "fare_per_person",
            "is_alone",
        ]
    ].head()
)

# 4. Other Feature Engineering Techniques We Learned
# 4.1 Binning
# Binning معناها:
# تحويل feature رقمية مستمرة إلى مجموعات.
# مثال:
# Age = 5  -> Child
# Age = 30 -> Adult
# Age = 70 -> Senior

titanic_fe["age_group"] = pd.cut(
    titanic_fe["Age"],
    bins=[0, 12, 18, 60, np.inf],
    labels=[
        "Child",
        "Teen",
        "Adult",
        "Senior",
    ],
)

print("\nAge Binning Example:")

print(
    titanic_fe[
        [
            "Age",
            "age_group",
        ]
    ].head()
)


# 4.2 One-Hot Encoding
# معظم موديلات ML تحتاج أرقام.
# إذا عندنا Category مثل:
# Child
# Adult
# Senior
# ما بصير نعطيهم ببساطة:
# Child = 1
# Adult = 2
# Senior = 3
# لأنه هيك بنعطي ترتيب رياضي وهمي.
# One-Hot Encoding تنشئ column منفصل لكل category.

age_group_encoded = pd.get_dummies(
    titanic_fe["age_group"],
    prefix="age_group",
    dtype=int,
)
print("\nOne-Hot Encoding Example:")
print(age_group_encoded.head())


# ملاحظة:
# في الموديل الرئيسي لليوم ما استخدمنا age_group.
# هذا الجزء فقط لتوضيح التقنية اللي درسناها.


# 4.3 Datetime Extraction
# Dataset الـ Titanic الحالية لا تحتوي Date مناسبة للتطبيق.
# لذلك نعمل مثال صغير فقط لفهم الفكرة.

date_demo = pd.DataFrame(
    {
        "order_date": [
            "2026-08-10",
            "2026-08-11",
            "2026-08-12",
        ]
    }
)


# نحول العمود أولاً إلى datetime
date_demo["order_date"] = pd.to_datetime(date_demo["order_date"])

# بعدها نستخرج معلومات مفيدة من التاريخ
date_demo["year"] = (date_demo["order_date"].dt.year)
date_demo["month"] = (date_demo["order_date"].dt.month)
date_demo["day_of_week"] = (date_demo["order_date"].dt.dayofweek)
print("\nDatetime Extraction Example:")
print(date_demo)

# 4.4 Scaling
# Scaling يجعل الـ numerical features على scales متقاربة.
# مثال:
# Age  تقريباً 0 -> 80
# Fare ممكن يكون 0 -> 500+
# بعض الموديلات تتأثر بهذا الاختلاف كثيراً.
# Random Forest عادةً لا يحتاج Scaling،
# لذلك هذا مثال تعليمي فقط.

scaling_demo = titanic_fe[
    [
        "Age",
        "Fare",
    ]
].copy()


# StandardScaler:
# يجعل البيانات تقريباً حول mean = 0 و std = 1
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(scaling_demo)
standard_scaled_df = pd.DataFrame(
    standard_scaled,
    columns=[
        "Age_scaled",
        "Fare_scaled",
    ],
)
print("\nStandardScaler Example:")
print(standard_scaled_df.head())


# MinMaxScaler:
# يحول القيم غالباً إلى range بين 0 و 1

minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(scaling_demo)
minmax_scaled_df = pd.DataFrame(
    minmax_scaled,
    columns=[
        "Age_minmax",
        "Fare_minmax",
    ],
)
print("\nMinMaxScaler Example:")
print(minmax_scaled_df.head())

# 5. Select Features for Our Main Model
# هاي هي الـ original features اللي استخدمناها سابقاً
baseline_features = [
    "Age",
    "Fare",
    "Sex",
    "sibsp",
    "Parch",
    "Pclass",
]

# هاي الـ features بعد إضافة Feature Engineering

engineered_features = [
    "Age",
    "Fare",
    "Sex",
    "sibsp",
    "Parch",
    "Pclass",
    "family_size",
    "fare_per_person",
    "is_alone",
]


X = titanic_fe[engineered_features]
y = titanic_fe["Survived"]
print("\nModel Features:")
print(X.columns.tolist())


# 6. Train / Validation / Test Split
# نفس Discipline اللي أخذناها في Day 1.
# الهدف:
# 60% Train
# 20% Validation
# 20% Test
# مهم جداً:
# Test Set لا نستخدمه أثناء tuning.
# أول Split:
# 80% temp
# 20% test

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# ثاني Split:
# نأخذ 25% من الـ 80%
# 0.25 * 0.80 = 0.20
# وبالتالي يصبح:
# 60% Train
# 20% Validation
# 20% Test

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
)


print("\nSplit Sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))



# 7. Parameters vs Hyperparameters
# Hyperparameters:
# نحن نحددها قبل التدريب.
# أمثلة:
# max_depth
# n_estimators
# random_state
# بينما الـ Parameters يتعلمها الموديل أثناء .fit().
# في Random Forest مثلاً:
# أماكن الـ splits داخل الأشجار يتم تعلمها من البيانات.


# 8. Create Untuned Baseline Model
# قبل ما نعمل tuning لازم يكون عندنا Baseline.
# Baseline = موديل بدون Hyperparameter Tuning.
# فائدته:
# نعرف هل الـ tuning فعلاً حسن الأداء أم لا.
X_baseline = titanic[baseline_features]
y_baseline = titanic["Survived"]

# نعمل نفس الـ split تماماً
X_base_temp, X_base_test, y_base_temp, y_base_test = (
    train_test_split(
        X_baseline,
        y_baseline,
        test_size=0.20,
        random_state=42,
    )
)

X_base_train, X_base_val, y_base_train, y_base_val = (
    train_test_split(
        X_base_temp,
        y_base_temp,
        test_size=0.25,
        random_state=42,
    )
)


baseline_model = RandomForestClassifier(
    random_state=42
)



# 9. Evaluate Baseline with 5-Fold Cross-Validation
# بدل الاعتماد على validation split واحدة،
# نستخدم Cross-Validation مثل Day 2.
# scoring="f1"
# لأن الـ F1-score هو metric اللي استخدمناها خلال الأسبوع.
baseline_scores = cross_val_score(
    baseline_model,
    X_base_train,
    y_base_train,
    cv=5,
    scoring="f1",
)


baseline_mean_f1 = baseline_scores.mean()
baseline_std_f1 = baseline_scores.std()
print("\nBaseline CV Scores:")
print(
    np.round(
        baseline_scores,
        3,
    )
)
print(f"Baseline Mean F1: "  f"{baseline_mean_f1:.3f}")
print(f"Baseline Std: " f"{baseline_std_f1:.3f}")


# 10. Define Hyperparameter Grid
# الآن نحدد القيم اللي بدنا GridSearchCV يجربها.
# عندنا:
# n_estimators:
# 100
# 200

# max_depth:
# 5
# 10
# None
# عدد combinations:
# 2 × 3 = 6
# ومع cv=5:
# 6 × 5 = 30 model fits
param_grid = {
    "n_estimators": [
        100,
        200,
    ],

    "max_depth": [
        5,
        10,
        None,
    ],
}
print("\nHyperparameter Grid:")
print(param_grid)



# 11. GridSearchCV
# GridSearchCV:
# 1. يأخذ الموديل
# 2. يأخذ الـ Hyperparameter Grid
# 3. يجرب جميع الـ combinations
# 4. يعمل Cross-Validation لكل combination
# 5. يقارن النتائج باستخدام F1-score
# 6. يختار أفضل combination


grid = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
)
# مهم:
# نعمل tuning على Training Data فقط.
# Validation/Test لا تدخل في عملية GridSearchCV.

grid.fit(
    X_train,
    y_train,
)


# 12. Read GridSearchCV Results
# أفضل Hyperparameters

print("\nBest Parameters:")
print(grid.best_params_)


# أفضل Mean Cross-Validation Score
print(f"\nBest Cross-Validated F1: "f"{grid.best_score_:.3f}"
)


# best_estimator_
# هو أفضل موديل بعد أن اختارت GridSearchCV
# أفضل Hyperparameters وعملت له refit.
best_model = grid.best_estimator_
print("\nBest Model:")
print(best_model)



# 13. Compare Baseline vs Tuned Model

tuned_mean_f1 = grid.best_score_
improvement = (tuned_mean_f1 - baseline_mean_f1
)
print("\nModel Comparison")
print(f"Untuned Baseline F1: " f"{baseline_mean_f1:.3f}")
print(f"Tuned Model F1: " f"{tuned_mean_f1:.3f}")
print(f"Improvement: " f"{improvement:+.3f}")


# في نتائج Day 4 الحالية تقريباً:
# Baseline = 0.515
# Tuned    = 0.517
# Improvement = +0.002
# يعني التحسن موجود لكنه صغير جداً.
# وهذا طبيعي.
# مش شرط Hyperparameter Tuning يعمل قفزة كبيرة دائماً.



# 14. Feature Importance
# Random Forest يعطينا:
# feature_importances_
# وهي تساعدنا نعرف أي Features استخدمها الموديل
# بشكل أكبر أثناء بناء القرارات.
feature_importance = pd.Series(
    best_model.feature_importances_,
    index=engineered_features,
)

# نرتبها من الأعلى إلى الأقل
feature_importance = (
    feature_importance
    .sort_values(
        ascending=False
    )
)
print("\nFeature Importance:")
print(feature_importance)


# حسب نتائج Day 4:
# Age كان أعلى Feature إجمالاً.
# ومن الـ engineered features الجديدة:
# fare_per_person
# كانت الأعلى تقريباً:
# 0.210


# 15. Feature Importance Plot
plt.figure(figsize=(8, 5))

feature_importance.plot(kind="barh")
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Tuned Random Forest")
plt.tight_layout()
plt.show()

# 16. Optional Validation Check
# GridSearchCV اختارت الموديل اعتماداً على Cross-Validation.
# نقدر أيضاً نشوف أداء أفضل موديل على Validation Set
# كفحص إضافي
# لكن لا نستخدم Validation Score لاختيار إعدادات جديدة
# بشكل متكرر بعد ما خلص الـ tuning.
val_predictions = best_model.predict(
    X_val
)
validation_f1 = f1_score(
    y_val,
    val_predictions,
    zero_division=0,
)
print(f"\nValidation F1: " f"{validation_f1:.3f}")


# 17. IMPORTANT: Keep Test Set Untouched
# لا نستخدم X_test / y_test الآن.
# السبب:
# Test Set لازم تظل مستقلة تماماً عن:
# Feature selection
# Model selection
# Hyperparameter tuning
# حتى نستخدمها في Final Evaluation لاحقاً.
# هذا نفس المبدأ اللي أخذناه من Day 1.
print("\nTest set is still untouched.")

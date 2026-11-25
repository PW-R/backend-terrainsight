import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

# ใช้ evaluate_kmeans.py เวอร์ชันล่าสุดที่มีสองฟังก์ชันนี้
from evaluate_kmeans import load_features, split_indices


RANDOM_STATE = 42
LABELS = [0, 1]


def evaluate(name, actual, predicted):
    print(f"\n{'=' * 75}\n{name}\n{'=' * 75}")

    print(f"Accuracy:          {accuracy_score(actual, predicted):.4f}")
    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_score(actual, predicted):.4f}"
    )
    print(
        f"Macro F1:          "
        f"{f1_score(actual, predicted, labels=LABELS, average='macro', zero_division=0):.4f}"
    )

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        actual,
        predicted,
        labels=LABELS,
        target_names=["Cluster 0", "Cluster 1"],
        digits=4,
        zero_division=0,
    ))

    matrix = pd.DataFrame(
        confusion_matrix(actual, predicted, labels=LABELS),
        index=["KMeans 0", "KMeans 1"],
        columns=["Predicted 0", "Predicted 1"],
    )

    print("CONFUSION MATRIX — rows=K-Means, columns=classifier")
    print(matrix.to_string())


def show_distribution(name, labels):
    counts = np.bincount(labels, minlength=2)
    print(
        f"{name}: Cluster 0={counts[0]} ({counts[0] / len(labels):.1%}), "
        f"Cluster 1={counts[1]} ({counts[1] / len(labels):.1%})"
    )


def main():
    cases, features = load_features()
    train_indices, test_indices = split_indices(cases)

    # เลือก features จาก Train เท่านั้น
    columns = features.columns[
        features.iloc[train_indices].nunique() > 1
    ].tolist()

    if not columns:
        raise ValueError("No nonconstant training features")

    x_train = features.iloc[train_indices][columns].reset_index(drop=True)
    x_test = features.iloc[test_indices][columns].reset_index(drop=True)

    if not (
        np.isfinite(x_train.to_numpy()).all()
        and np.isfinite(x_test.to_numpy()).all()
    ):
        raise ValueError("Features contain invalid numbers")

    if len(np.unique(x_train.to_numpy(), axis=0)) < 2:
        raise ValueError("Not enough distinct training cases")

    # K-Means ต้องใช้ข้อมูลปรับสเกล
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)

    kmeans = KMeans(
        n_clusters=2,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    y_train = kmeans.fit_predict(train_scaled)
    y_test = kmeans.predict(test_scaled)

    if len(np.unique(y_train)) != 2:
        raise ValueError("K-Means did not produce two training clusters")

    print(f"\n{'=' * 75}")
    print("TERRA INSIGHT — RANDOM FOREST CLASSIFICATION")
    print(f"{'=' * 75}")
    print(f"Total cases: {len(cases)}")
    print(f"Train: {len(x_train)}, Test: {len(x_test)}")
    print(f"Features: {columns}")
    print("Target: cluster assigned by training-fitted K-Means")
    print("Population is district-wide; nearby counts use a 1 km radius.")

    show_distribution("Train", y_train)
    show_distribution("Test ", y_test)

    # Baseline สำหรับตรวจว่า RF ดีกว่าทายกลุ่มส่วนใหญ่เสมอหรือไม่
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)

    evaluate(
        "BASELINE — TEST",
        y_test,
        baseline.predict(x_test),
    )

    # RF ใช้ features หน่วยเดิมได้ ไม่จำเป็นต้องปรับสเกล
    forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    forest.fit(x_train, y_train)

    train_pred = forest.predict(x_train)
    test_pred = forest.predict(x_test)

    evaluate("RANDOM FOREST — TRAIN", y_train, train_pred)
    evaluate("RANDOM FOREST — TEST", y_test, test_pred)

    print("\nTRAIN–TEST MACRO F1 GAP")
    train_f1 = f1_score(
        y_train, train_pred,
        labels=LABELS, average="macro", zero_division=0,
    )
    test_f1 = f1_score(
        y_test, test_pred,
        labels=LABELS, average="macro", zero_division=0,
    )
    print(f"Train minus Test: {train_f1 - test_f1:+.4f}")

    # เป็นความสำคัญจากการแบ่งต้นไม้ ไม่ใช่หลักฐานเชิงสาเหตุ
    importance = pd.DataFrame({
        "feature": columns,
        "tree_importance": forest.feature_importances_,
    }).sort_values("tree_importance", ascending=False)

    print("\nFEATURE IMPORTANCE — TREE SPLITS, NOT CAUSAL EFFECTS")
    print(importance.round(4).to_string(index=False))

    results = cases.iloc[test_indices][
        ["district", "lat", "lon"]
    ].reset_index(drop=True)

    results["kmeans_cluster"] = y_test
    results["rf_cluster"] = test_pred

    mismatches = results[
        results["kmeans_cluster"] != results["rf_cluster"]
    ]

    print(f"\nTEST DISAGREEMENTS: {len(mismatches)}/{len(results)}")
    if not mismatches.empty:
        print(mismatches.head(20).to_string(index=False))

    print("\nFIRST 10 TEST PREDICTIONS")
    print(results.head(10).to_string(index=False))

    print("\nDone. No files or models were saved.")
    print("Scores measure agreement with K-Means, not real-world suitability.")
    print("Nearby Train/Test points may have overlapping 1 km areas.")


if __name__ == "__main__":
    main()
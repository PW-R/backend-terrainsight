import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from evaluate_kmeans import load_features, split_indices


K = 10
RANDOM_STATE = 42

FEATURE_NAMES = {
    "population": "district population",
    "temples": "nearby temples",
    "malls": "nearby malls",
    "health": "nearby health centers",
    "risk": "nearby recorded risk points",
    "schools": "nearby schools",
    "universities": "nearby universities",
}


def describe_clusters(x_train, train_scaled, y_train, y_test):
    averages = x_train.assign(cluster=y_train).groupby("cluster").mean()

    standardized = pd.DataFrame(train_scaled, columns=x_train.columns)
    differences = standardized.assign(cluster=y_train).groupby("cluster").mean()

    names = {}

    for cluster, row in differences.iterrows():
        # Compare each group's average with the overall training average.
        standout_features = row.abs().nlargest(2).index
        descriptions = []

        for feature in standout_features:
            difference = row[feature]

            if abs(difference) < 0.25:
                continue

            if feature == "population":
                direction = "higher" if difference > 0 else "lower"
            else:
                direction = "more" if difference > 0 else "fewer"

            descriptions.append(f"{direction} {FEATURE_NAMES[feature]}")

        name = " and ".join(descriptions).capitalize() or "Mixed profile"
        names[cluster] = name

    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()

    averages.insert(0, "test_locations", test_counts.reindex(averages.index, fill_value=0))
    averages.insert(0, "train_locations", train_counts.reindex(averages.index))
    averages.insert(0, "suggested_name", pd.Series(names))

    return averages


def main():
    cases, features = load_features()
    train_indices, test_indices = split_indices(cases)

    # Select features using all training locations now that K is fixed.
    train_features = features.iloc[train_indices]
    columns = train_features.columns[train_features.nunique() > 1].tolist()

    if not columns:
        raise ValueError("No nonconstant training features")

    x_train = train_features[columns].reset_index(drop=True)
    x_test = features.iloc[test_indices][columns].reset_index(drop=True)

    if not all(np.isfinite(frame.to_numpy()).all() for frame in (x_train, x_test)):
        raise ValueError("Features contain invalid numbers")

    if len(x_train.drop_duplicates()) < K:
        raise ValueError(f"Need at least {K} distinct training feature rows")

    # K-Means uses scaled features. Random Forest uses the original values.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)

    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    y_train = kmeans.fit_predict(train_scaled)

    if len(np.unique(y_train)) != K:
        raise ValueError(f"K-Means formed fewer than {K} groups")

    y_test = kmeans.predict(scaler.transform(x_test))

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)

    forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(x_train, y_train)

    predicted = forest.predict(x_test)
    profiles = describe_clusters(x_train, train_scaled, y_train, y_test)
    matches = int(np.count_nonzero(y_test == predicted))

    print(f"\nTERRA INSIGHT — K-MEANS + RANDOM FOREST (K={K})")
    print(f"Train: {len(x_train)}, Test: {len(x_test)}")

    print("\nSUGGESTED GROUP NAMES AND AVERAGE FEATURES")
    print(profiles.round(2).to_string())
    print("\nPopulation covers the whole district. Nearby counts use a 1 km radius.")

    print("\nRANDOM FOREST TEST RESULTS")
    print(f"Majority-group baseline accuracy: {accuracy_score(y_test, baseline.predict(x_test)):.2%}")
    print(f"Random Forest matched K-Means: {matches}/{len(y_test)} ({matches / len(y_test):.2%})")
    print()
    print(classification_report(
        y_test,
        predicted,
        labels=list(range(K)),
        target_names=[f"Group {cluster}" for cluster in range(K)],
        digits=4,
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
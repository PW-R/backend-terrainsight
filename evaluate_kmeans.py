from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler


DATA = Path(__file__).resolve().parent / "data"

# ชื่อย่อ, ไฟล์, latitude, longitude, เขต, รหัส/ชื่อ, ธงพิกัด
SOURCES = [
    ("temples", "bangkok_temples_clean.csv",
     "latitude", "longitude", "district", "temple_name", None),

    ("malls", "cleaned ที่ตั้งห้างสรรพสินค้า.csv",
     "lat", "lng", "dname", "id_depart", None),

    ("health", "health_centers_clean.csv",
     "Latitude", "longitude", "DISTRICT", "BHCName", "coord_suspect"),

    ("risk", "risk_points_clean.csv",
     "ALatitude", "ALongitude", "AreaDist", "AName", None),

    ("schools", "schools_clean.csv",
     "ละติจูด", "ลองจิจูด", "เขต_ชื่อล้วน",
     "รหัสสถานศึกษา", "พิกัดผิดปกติ"),

    ("universities", "universities_clean.csv",
     "lat", "lng", "DISTRICT", "id", None),
]


def read(filename):
    for encoding in ["utf-8-sig", "cp874", "tis-620"]:
        try:
            return pd.read_csv(DATA / filename, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode: {filename}")


def district(value):
    if pd.isna(value):
        return ""

    value = re.sub(r"\s+", "", str(value))
    value = re.sub(r"^(เขต|อำเภอ)", "", value)
    return {"สัมพันธ์วงศ์": "สัมพันธวงศ์"}.get(value, value)


def number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def load_features():
    df = read("bangkok_population_clean.csv")

    population = pd.DataFrame({
        "district": df["เขต"].map(district),
        "population": number(df["รวม"]),
    }).set_index("district")

    if population.index.duplicated().any():
        raise ValueError("Duplicate population districts")

    if (
        "" in population.index
        or not np.isfinite(population.to_numpy()).all()
        or (population["population"] < 0).any()
    ):
        raise ValueError("Invalid population data")

    sources, locations = {}, []

    for name, filename, lat, lon, area, identity, flag in SOURCES:
        df = read(filename)
        original_count = len(df)

        if name == "risk":
            years = number(df["AFiscalYeal"])

            if years.isna().any():
                raise ValueError("Invalid risk reporting year")

            latest = int(years.max())
            df = df.loc[years == latest].copy()
            print(f"Risk reporting year: {latest}")

        flagged_count = 0

        if flag:
            values = df[flag].astype(str).str.strip().str.lower()

            if not values.isin(["true", "false", "1", "0"]).all():
                raise ValueError(f"Invalid flags: {filename}")

            flagged = values.isin(["true", "1"])
            flagged_count = int(flagged.sum())
            df = df.loc[~flagged].copy()

        points = pd.DataFrame({
            "lat": number(df[lat]),
            "lon": number(df[lon]),
            "district": df[area].map(district),
            "identity": df[identity],
        })

        valid = (
            points["lat"].between(13, 14)
            & points["lon"].between(100, 101)
        )

        invalid_count = int((~valid).sum())
        points = points.loc[valid].drop_duplicates(
            ["identity", "lat", "lon"]
        )

        if points.empty:
            raise ValueError(f"No usable coordinates: {filename}")

        unknown = ~points["district"].isin(population.index)

        if unknown.any():
            print(points.loc[unknown].to_string(index=False))
            raise ValueError(f"Unknown districts: {filename}")

        print(
            f"{name}: original={original_count}, used={len(points)}, "
            f"flagged_removed={flagged_count}, "
            f"invalid_coordinates_removed={invalid_count}"
        )

        sources[name] = points[["lat", "lon"]].to_numpy()
        locations.append(points[["district", "lat", "lon"]])

    cases = (
        pd.concat(locations, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    conflicts = cases.groupby(["lat", "lon"])["district"].nunique()

    if (conflicts > 1).any():
        raise ValueError("A coordinate has conflicting district labels")

    features = population.loc[cases["district"]].reset_index(drop=True)
    coordinates = np.radians(cases[["lat", "lon"]].to_numpy())

    for name, points in sources.items():
        tree = BallTree(np.radians(points), metric="haversine")

        features[name] = tree.query_radius(
            coordinates,
            r=1 / 6371.0088,
            count_only=True,
        )

    return cases, features


def split_indices(cases):
    train, test = [], []

    for area, group in cases.groupby("district", sort=True):
        indices = group.index.to_numpy().copy()
        rng = np.random.RandomState(42 + sum(map(ord, area)))
        rng.shuffle(indices)

        count = (
            min(len(indices) - 1, max(1, int(np.ceil(len(indices) * .2))))
            if len(indices) > 1 else 0
        )

        test.extend(indices[:count])
        train.extend(indices[count:])

    if not train or not test:
        raise ValueError("Insufficient data for Train/Test")

    return train, test


def evaluate(name, raw, scaled, labels, model):
    cluster_count = model.n_clusters
    counts = np.bincount(labels, minlength=cluster_count)
    unique_labels = len(np.unique(labels))

    silhouette = (
        silhouette_score(scaled, labels)
        if 1 < unique_labels < len(labels)
        else None
    )

    # Average squared distance to the assigned training centroid.
    squared_distances = (
        (scaled - model.cluster_centers_[labels]) ** 2
    ).sum(axis=1)

    print(f"\n{'=' * 75}\n{name}\n{'=' * 75}")
    print(f"Cases: {len(raw)}")
    print(
        f"Silhouette: {silhouette:.4f}"
        if silhouette is not None
        else "Silhouette: unavailable (not enough represented clusters)"
    )
    print(f"Mean squared centroid distance: {squared_distances.mean():.4f}")

    profile = raw.assign(cluster=labels).groupby("cluster").mean()
    profile = profile.reindex(range(cluster_count))
    profile.insert(0, "size", counts)
    profile.insert(1, "pct", counts / len(raw) * 100)

    print("\nMEAN FEATURES PER CLUSTER")
    print(profile.round(2).to_string())

    return silhouette


def main():
    cases, features = load_features()
    train, test = split_indices(cases)

    # Choose features from Train only.
    columns = features.columns[
        features.iloc[train].nunique() > 1
    ].tolist()

    if not columns:
        raise ValueError("No nonconstant training features")

    x_train = features.iloc[train][columns].reset_index(drop=True)
    x_test = features.iloc[test][columns].reset_index(drop=True)

    if not (
        np.isfinite(x_train.to_numpy()).all()
        and np.isfinite(x_test.to_numpy()).all()
    ):
        raise ValueError("Features contain invalid numbers")

    if len(np.unique(x_train.to_numpy(), axis=0)) < 2:
        raise ValueError("Not enough distinct training cases")

    # Fit on Train only, then transform Test.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)

    # Same baseline settings as the previous seed-42 experiment.
    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10,
    )

    train_labels = model.fit_predict(train_scaled)
    test_labels = model.predict(test_scaled)

    print(
        f"\nTotal={len(cases)}, Train={len(train)}, Test={len(test)}"
    )
    print(f"Features: {columns}")
    print("Population is district-wide; nearby counts use a 1 km radius.")

    train_score = evaluate(
        "TRAIN — K=2",
        x_train,
        train_scaled,
        train_labels,
        model,
    )

    test_score = evaluate(
        "TEST — K=2",
        x_test,
        test_scaled,
        test_labels,
        model,
    )

    if train_score is not None and test_score is not None:
        print(
            f"\nTrain minus Test Silhouette: "
            f"{train_score - test_score:+.4f}"
        )

    test_cases = cases.iloc[test].reset_index(drop=True)
    test_cases["cluster"] = test_labels

    print("\nTOP 5 TEST DISTRICTS PER CLUSTER")

    for cluster in range(2):
        top = (
            test_cases.loc[
                test_cases["cluster"] == cluster, "district"
            ]
            .value_counts()
            .head(5)
        )
        print(f"Cluster {cluster}: {top.to_dict()}")

    print("\nDone. No files or models were saved.")
    print("Test labels come from the fitted training K-Means.")
    print("Silhouette measures separation, not prediction accuracy.")
    print("Nearby Train/Test points may have overlapping 1 km areas.")


if __name__ == "__main__":
    main()
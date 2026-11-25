from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler


DATA = Path(__file__).resolve().parent / "data"
RANDOM_STATE = 42
MIN_K = 2
MAX_K = 50

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
        rng = np.random.RandomState(RANDOM_STATE + sum(map(ord, area)))
        rng.shuffle(indices)

        count = (
            min(len(indices) - 1, max(1, int(np.ceil(len(indices) * 0.2))))
            if len(indices) > 1 else 0
        )

        test.extend(indices[:count])
        train.extend(indices[count:])

    if not train or not test:
        raise ValueError("Insufficient data for Train/Test")

    return train, test


def get_silhouette(scaled, labels):
    represented_clusters = len(np.unique(labels))

    if 1 < represented_clusters < len(labels):
        return silhouette_score(scaled, labels)

    return None


def evaluate(name, raw, scaled, labels, model):
    cluster_count = model.n_clusters
    counts = np.bincount(labels, minlength=cluster_count)
    silhouette = get_silhouette(scaled, labels)

    # Average squared distance to the assigned training centroid.
    squared_distances = (
        (scaled - model.cluster_centers_[labels]) ** 2
    ).sum(axis=1)

    print(f"\n{'=' * 75}\n{name}\n{'=' * 75}")
    print(f"Cases: {len(raw)}")
    print(f"Clusters represented: {np.count_nonzero(counts)}/{cluster_count}")
    print(
        f"Silhouette: {silhouette:.4f}"
        if silhouette is not None
        else "Silhouette: unavailable for this label distribution"
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

    # Split the original training set again to get a validation set.
    # split_indices uses row index labels, so keep the original case indices.
    fit_indices, validation_indices = split_indices(cases.iloc[train])

    columns = features.columns[
        features.iloc[fit_indices].nunique() > 1
    ].tolist()

    if not columns:
        raise ValueError("No nonconstant features in the fitting set")

    x_fit = features.iloc[fit_indices][columns]
    x_validation = features.iloc[validation_indices][columns]
    x_train = features.iloc[train][columns].reset_index(drop=True)
    x_test = features.iloc[test][columns].reset_index(drop=True)

    if not all(
        np.isfinite(frame.to_numpy()).all()
        for frame in (x_fit, x_validation, x_test)
    ):
        raise ValueError("Features contain invalid numbers")

    search_scaler = StandardScaler()
    fit_scaled = search_scaler.fit_transform(x_fit)
    validation_scaled = search_scaler.transform(x_validation)

    distinct_fit_rows = len(np.unique(fit_scaled, axis=0))
    max_supported_k = min(MAX_K, distinct_fit_rows)

    if max_supported_k < MIN_K:
        raise ValueError("Not enough distinct fitting cases for K-Means")

    print(f"\nTotal locations: {len(cases)}")
    print(
        f"Fit: {len(x_fit)}, Validation: {len(x_validation)}, "
        f"Final Test: {len(x_test)}"
    )
    print(f"Features: {columns}")
    print("Population is district-wide; nearby counts use a 1 km radius.")
    print(f"\nCOMPARING K={MIN_K} TO K={max_supported_k}")

    if max_supported_k < MAX_K:
        print(f"K={max_supported_k + 1} to {MAX_K} require more distinct fitting cases.")

    results = []

    for k in range(MIN_K, max_supported_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        fit_labels = model.fit_predict(fit_scaled)
        fit_counts = np.bincount(fit_labels, minlength=k)

        if (fit_counts == 0).any():
            print(f"K={k:2d} | skipped: fewer than {k} clusters formed")
            continue

        validation_labels = model.predict(validation_scaled)
        validation_coverage = len(np.unique(validation_labels))
        score = get_silhouette(validation_scaled, validation_labels)
        coverage = f"validation clusters={validation_coverage}/{k}"
        smallest_cluster = int(fit_counts.min())

        if score is None:
            print(f"K={k:2d} | silhouette unavailable | {coverage}")
            continue

        results.append({
            "K": k,
            "validation_silhouette": score,
            "validation_clusters": validation_coverage,
            "smallest_fit_cluster": smallest_cluster,
        })

        print(
            f"K={k:2d} | silhouette={score:.4f} | "
            f"{coverage} | smallest fit cluster={smallest_cluster}"
        )

    if not results:
        raise ValueError("No candidate K produced a valid validation silhouette")

    ranking = pd.DataFrame(results).sort_values(
        ["validation_silhouette", "K"],
        ascending=[False, True],
    )

    print("\nTOP 10 K VALUES BY VALIDATION SILHOUETTE")
    print(ranking.head(10).round(4).to_string(index=False))

    best_k = int(ranking.iloc[0]["K"])
    print(f"\nBest K by validation silhouette: {best_k}")

    # Refit the winning K on all original training rows.
    final_scaler = StandardScaler()
    train_scaled = final_scaler.fit_transform(x_train)
    test_scaled = final_scaler.transform(x_test)

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    train_labels = final_model.fit_predict(train_scaled)
    test_labels = final_model.predict(test_scaled)

    train_score = evaluate(
        f"FINAL TRAIN — K={best_k}",
        x_train,
        train_scaled,
        train_labels,
        final_model,
    )

    test_score = evaluate(
        f"FINAL TEST — K={best_k}",
        x_test,
        test_scaled,
        test_labels,
        final_model,
    )

    if train_score is not None and test_score is not None:
        print(f"\nTrain minus Test Silhouette: {train_score - test_score:+.4f}")

    test_cases = cases.iloc[test].reset_index(drop=True)
    test_cases["cluster"] = test_labels

    print("\nTOP 5 TEST DISTRICTS PER CLUSTER")

    for cluster in range(best_k):
        top = (
            test_cases.loc[test_cases["cluster"] == cluster, "district"]
            .value_counts()
            .head(5)
        )
        print(f"Cluster {cluster}: {top.to_dict()}")

    print("\nNearby Train/Validation/Test points may have overlapping 1 km areas.")


if __name__ == "__main__":
    main()
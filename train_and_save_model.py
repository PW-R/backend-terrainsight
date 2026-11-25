from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_PATH = SCRIPT_DIR / "terra_insight_model.joblib"

K = 10
RANDOM_STATE = 42
NEARBY_RADIUS_KM = 1.0
EARTH_RADIUS_KM = 6371.0088

# Feature, CSV, latitude, longitude, district, ID, coordinate flag
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
     "ละติจูด", "ลองจิจูด", "เขต_ชื่อล้วน", "รหัสสถานศึกษา", "พิกัดผิดปกติ"),
    ("universities", "universities_clean.csv",
     "lat", "lng", "DISTRICT", "id", None),
]

GROUP_NAMES = {
    0: "School and Temple Areas",
    1: "Temple Areas with Many Recorded Risk Points",
    2: "Populous Districts with Nearby Health Centers",
    3: "Mall and University Areas",
    4: "Populous Districts with Few Nearby Schools",
    5: "Health Center Areas with Few Recorded Risk Points",
    6: "Areas with Few Nearby Schools and Temples",
    7: "Temple-Heavy Areas with Several Schools",
    8: "University and School Areas",
    9: "Mall Areas with Few Nearby Schools",
}

# Training counts and average temples from the run we used to name the groups.
# These provide a quick check that the group IDs still describe the same areas.
EXPECTED_TRAIN_COUNTS = [137, 44, 78, 31, 250, 84, 238, 65, 94, 118]
EXPECTED_AVERAGE_TEMPLES = [
    6.45, 18.66, 0.74, 1.29, 0.79, 3.20, 1.21, 12.35, 8.94, 1.40,
]


def read_csv(filename):
    path = DATA_DIR / filename

    for encoding in ("utf-8-sig", "cp874", "tis-620"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Cannot decode CSV: {path}")


def normalize_district(value):
    if pd.isna(value):
        return ""

    value = re.sub(r"\s+", "", str(value))
    value = re.sub(r"^(เขต|อำเภอ)", "", value)
    return {"สัมพันธ์วงศ์": "สัมพันธวงศ์"}.get(value, value)


def to_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def load_features():
    data = read_csv("bangkok_population_clean.csv")
    population = pd.DataFrame({
        "district": data["เขต"].map(normalize_district),
        "population": to_number(data["รวม"]),
    }).set_index("district")

    if (
        population.index.has_duplicates
        or "" in population.index
        or not np.isfinite(population.to_numpy()).all()
        or (population["population"] < 0).any()
    ):
        raise ValueError("Invalid or duplicate district population data")

    source_points = {}
    locations = []

    for name, filename, lat, lon, area, identity, flag in SOURCES:
        data = read_csv(filename)

        if name == "risk":
            years = to_number(data["AFiscalYeal"])
            if years.empty or not np.isfinite(years.to_numpy()).all():
                raise ValueError("Invalid risk reporting year")

            latest_year = years.max()
            data = data.loc[years == latest_year]
            print(f"Risk reporting year: {int(latest_year)}")

        if flag:
            flags = data[flag].astype(str).str.strip().str.lower()
            if not flags.isin(["true", "false", "1", "0"]).all():
                raise ValueError(f"Invalid coordinate flags in {filename}")

            data = data.loc[~flags.isin(["true", "1"])]

        points = pd.DataFrame({
            "lat": to_number(data[lat]),
            "lon": to_number(data[lon]),
            "district": data[area].map(normalize_district),
            "identity": data[identity],
        })

        valid = points["lat"].between(13, 14) & points["lon"].between(100, 101)
        points = points.loc[valid].drop_duplicates(["identity", "lat", "lon"])

        if points.empty:
            raise ValueError(f"No usable coordinates in {filename}")

        if not points["district"].isin(population.index).all():
            raise ValueError(f"Unknown district in {filename}")

        coordinates = points[["lat", "lon"]]
        source_points[name] = coordinates.to_numpy()
        locations.append(points[["district", "lat", "lon"]])
        print(f"{name}: {len(points)} locations used")

    cases = pd.concat(locations, ignore_index=True).drop_duplicates().reset_index(drop=True)

    if (cases.groupby(["lat", "lon"])["district"].nunique() > 1).any():
        raise ValueError("A coordinate has conflicting district labels")

    features = population.loc[cases["district"]].reset_index(drop=True)
    coordinates = np.radians(cases[["lat", "lon"]].to_numpy())

    for name, points in source_points.items():
        tree = BallTree(np.radians(points), metric="haversine")
        features[name] = tree.query_radius(
            coordinates,
            r=NEARBY_RADIUS_KM / EARTH_RADIUS_KM,
            count_only=True,
        )

    return cases, features


def split_indices(cases):
    train = []
    test = []

    # Reproduce the same district-by-district split used in the earlier run.
    for district, group in cases.groupby("district", sort=True):
        indices = group.index.to_numpy().copy()
        rng = np.random.RandomState(RANDOM_STATE + sum(map(ord, district)))
        rng.shuffle(indices)

        test_count = 0
        if len(indices) > 1:
            test_count = min(len(indices) - 1, int(np.ceil(len(indices) * 0.2)))

        test.extend(indices[:test_count])
        train.extend(indices[test_count:])

    if not train or not test:
        raise ValueError("Insufficient locations for a Train/Test split")

    return train, test


def main():
    cases, features = load_features()
    train_indices, test_indices = split_indices(cases)

    train_features = features.iloc[train_indices]
    feature_columns = train_features.columns[train_features.nunique() > 1].tolist()
    if not feature_columns:
        raise ValueError("No nonconstant training features")

    x_train = train_features[feature_columns].reset_index(drop=True)
    x_test = features.iloc[test_indices][feature_columns].reset_index(drop=True)

    if not all(np.isfinite(frame.to_numpy()).all() for frame in (x_train, x_test)):
        raise ValueError("Features contain invalid numbers")

    if len(x_train.drop_duplicates()) < K:
        raise ValueError(f"Need at least {K} distinct training feature rows")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)

    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    y_train = kmeans.fit_predict(train_scaled)

    if len(np.unique(y_train)) != K:
        raise ValueError(f"K-Means formed fewer than {K} groups")

    profiles = x_train.assign(group_id=y_train).groupby("group_id").mean()
    train_counts = np.bincount(y_train, minlength=K)
    profiles.insert(0, "train_locations", train_counts)

    print("\nTRAINING GROUP PROFILES")
    print(profiles.round(2).to_string())

    names_match = (
        np.array_equal(train_counts, EXPECTED_TRAIN_COUNTS)
        and "temples" in profiles
        and np.allclose(
            profiles["temples"].round(2),
            EXPECTED_AVERAGE_TEMPLES,
            rtol=0,
            atol=1e-8,
        )
    )

    if not names_match:
        raise RuntimeError(
            "Group profiles differ from the run used to choose the names. "
            "Review the printed profiles and update the group names and "
            "expected values before saving."
        )

    y_test = kmeans.predict(scaler.transform(x_test))

    forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(x_train, y_train)

    predictions = forest.predict(x_test)
    matches = int(np.count_nonzero(y_test == predictions))
    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=np.unique(y_test),
        average="macro",
        zero_division=0,
    )

    print(f"\nTrain locations: {len(x_train)}")
    print(f"Test locations: {len(x_test)}")
    print(f"Random Forest matched K-Means: {matches}/{len(y_test)} ({matches / len(y_test):.2%})")
    print(f"Test Macro F1: {macro_f1:.4f}")

    bundle = {
        "model": forest,
        "feature_columns": feature_columns,
        "group_names": GROUP_NAMES,
        "nearby_radius_km": NEARBY_RADIUS_KM,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, MODEL_PATH)

    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
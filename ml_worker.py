
import json
import sys
import math
import time
import traceback

from pathlib import Path
from contextlib import redirect_stdout
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd

from train_and_save_model import load_features
# บัวเพิ่มคับพี่ๆ
from district_profiles import build_district_profiles

# ==========================================
# UTF-8 Encoding Configuration
# ==========================================

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )


# ==========================================
# 1. Configuration
# ==========================================

MODEL_PATH = sys.argv[1]

MAX_RECOMMENDATIONS = 5

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

CACHE_PATH = (
    Path(__file__).resolve().parent
    / "geocode_cache.json"
)

GEOCODING_TIMEOUT = 7
GEOCODING_INTERVAL = 1.1

last_geocode_request = 0.0


FACTOR_ORDER = [
    "safety",
    "hospital",
    "school",
    "shopping",
    "park",
    "transport",
]


FACTOR_LABELS = {
    "safety": "ความปลอดภัย",
    "hospital": "โรงพยาบาล",
    "school": "โรงเรียน",
    "shopping": "ห้างสรรพสินค้า",
    "park": "สวนสาธารณะ",
    "transport": "การเดินทาง",
}


SUPPORTED_FACTORS = {
    "safety": {
        "keywords": ["risk", "ความเสี่ยง"],
        "direction": "asc",
    },

    "hospital": {
        "keywords": [
            "health",
            "hospital",
            "สุขภาพ",
            "โรงพยาบาล",
        ],
        "direction": "desc",
    },

    "school": {
        "keywords": ["school", "โรงเรียน"],
        "direction": "desc",
    },

    "shopping": {
        "keywords": ["mall", "shopping", "ห้าง"],
        "direction": "desc",
    },
}


# ==========================================
# 2. Load Geocoding Cache
# ==========================================

try:
    with open(
        CACHE_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        geocode_cache = json.load(file)

    if not isinstance(geocode_cache, dict):
        geocode_cache = {}

except (
    FileNotFoundError,
    json.JSONDecodeError,
    OSError,
):
    geocode_cache = {}


def save_geocode_cache():
    try:
        with open(
            CACHE_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                geocode_cache,
                file,
                ensure_ascii=False,
                indent=2,
            )

    except OSError as error:
        print(
            f"Could not save geocoding cache: {error}",
            file=sys.stderr,
        )


# ==========================================
# 3. Geocode Work Location
# ==========================================


def geocode_location(location):
    """
    แปลงชื่อสถานที่ทำงานเป็นพิกัด
    ใช้ OpenStreetMap Nominatim
    รองรับชื่อสถานที่ภาษาไทย เช่น สยาม
    """

    global last_geocode_request

    query = str(location).strip()

    if not query:
        raise ValueError("กรุณาระบุสถานที่ทำงาน")

    cache_key = query.casefold()

    # ใช้ Cache ก่อน
    if cache_key in geocode_cache:
        cached = geocode_cache[cache_key]

        print(
            f"Geocoding cache hit: {query}",
            file=sys.stderr,
            flush=True,
        )

        return {
            "lat": float(cached["lat"]),
            "lon": float(cached["lon"]),
        }

    # กำหนดคำค้นสำรองสำหรับสถานที่ที่พบบ่อย
    normalized_query = query.casefold()

    if normalized_query in ["สยาม", "siam"]:
        search_queries = [
            "Siam Square, Pathum Wan, Bangkok, Thailand"
        ]
    else:
        search_queries = [
            f"{query}, Bangkok, Thailand"
        ]

    results = None
    last_error = None

    for search_query in search_queries:

        # จำกัดความถี่การเรียก Nominatim
        elapsed = time.monotonic() - last_geocode_request

        if elapsed < GEOCODING_INTERVAL:
            time.sleep(GEOCODING_INTERVAL - elapsed)

        params = {
            "q": search_query,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "th",
            "accept-language": "en",
        }

        url = f"{NOMINATIM_URL}?{urlencode(params)}"

        request = Request(
            url,
            headers={
                "User-Agent": "TerraInsight/1.0 (academic project)",
                "Accept-Language": "en",
            },
        )

        try:
            last_geocode_request = time.monotonic()

            with urlopen(
                request,
                timeout=GEOCODING_TIMEOUT,
            ) as response:

                data = json.loads(
                    response.read().decode("utf-8")
                )

            if data:
                results = data
                break

        except HTTPError as error:
            last_error = error
            headers = error.headers or {}
            try:
                with error:
                    body = error.read(1000).decode("utf-8", errors="replace")
            except (OSError, ValueError):
                body = "(unavailable)"
            print(
                f"Geocoding HTTP {error.code} | "
                f"Server: {headers.get('Server', '-')} | "
                f"Via: {headers.get('Via', '-')} | "
                f"Retry-After: {headers.get('Retry-After', '-')} | "
                f"Body: {body!r}",
                file=sys.stderr,
                flush=True,
            )

        except Exception as error:
            last_error = error

            print(
                f"Geocoding error: {error}",
                file=sys.stderr,
                flush=True,
            )

    if not results:
        if last_error:
            raise ValueError(
                "ไม่สามารถค้นหาพิกัดสถานที่ได้ "
                f"รายละเอียด: {last_error}"
            )

        raise ValueError(
            f"ไม่พบสถานที่ '{query}' กรุณาตรวจสอบชื่อสถานที่"
        )

    latitude = float(results[0]["lat"])
    longitude = float(results[0]["lon"])

    if not (
        -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        raise ValueError("พิกัดสถานที่ไม่ถูกต้อง")

    coordinates = {
        "lat": latitude,
        "lon": longitude,
    }

    # บันทึก Cache โดยใช้คำที่ผู้ใช้กรอก
    geocode_cache[cache_key] = coordinates
    save_geocode_cache()

    print(
        f"Geocoded location: {query}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Matched location: {results[0].get('display_name', '')}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"Coordinates: {latitude}, {longitude}",
        file=sys.stderr,
        flush=True,
    )

    return coordinates



# ==========================================
# 4. Calculate Distance using Haversine
# ==========================================

def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    คำนวณระยะทางเส้นตรงระหว่างพิกัด
    ผลลัพธ์เป็นกิโลเมตร
    """

    earth_radius = 6371.0

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))

    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return earth_radius * c


# ==========================================
# 5. Load Existing Model
# ==========================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]

feature_columns = bundle["feature_columns"]

group_names = bundle["group_names"]

if not hasattr(model, "predict"):
    raise ValueError(
        "Loaded model does not support predict()"
    )


# ==========================================
# 6. Load Existing Features
# ==========================================

with redirect_stdout(sys.stderr):
    cases, features = load_features()


# ==========================================
# 7. Validate Data
# ==========================================

required_case_columns = [
    "district",
    "lat",
    "lon",
]

missing_case_columns = [
    col
    for col in required_case_columns
    if col not in cases.columns
]

if missing_case_columns:
    raise ValueError(
        f"Missing case columns: {missing_case_columns}"
    )

if len(cases) != len(features):
    raise ValueError(
        "Cases and features have different row counts"
    )

cases = cases.copy()

features = features.copy()

if cases["district"].isna().any():
    raise ValueError(
        "District names contain missing values"
    )

cases["district"] = (
    cases["district"]
    .astype(str)
    .str.strip()
)

if (cases["district"] == "").any():
    raise ValueError(
        "District names contain empty values"
    )

# ตรวจสอบพิกัด
cases["lat"] = pd.to_numeric(
    cases["lat"],
    errors="coerce",
)

cases["lon"] = pd.to_numeric(
    cases["lon"],
    errors="coerce",
)

# ตรวจสอบ Model Features
missing_features = [
    col
    for col in feature_columns
    if col not in features.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

# ใช้ Feature เดิมและลำดับเดิม
features = features[feature_columns].copy()

features = features.apply(
    pd.to_numeric,
    errors="coerce",
)

if features.isna().any().any():
    raise ValueError(
        "Model features contain missing or non-numeric values"
    )

if not np.isfinite(
    features.to_numpy(dtype=float)
).all():
    raise ValueError(
        "Model features contain infinite values"
    )


# ==========================================
# 8. Predict Clusters using Existing Model
# ==========================================

# ใช้ Model เดิม ไม่ได้ Train ใหม่
predictions = model.predict(features)

if len(predictions) != len(cases):
    raise ValueError(
        "Prediction count does not match cases count"
    )

prediction_data = cases[["district"]].copy()

prediction_data["cluster_id"] = [
    int(value)
    for value in predictions
]


# ==========================================
# 9. Assign One Cluster to Each District
# ==========================================

def get_main_cluster(values):

    counts = values.value_counts()

    max_count = counts.max()

    tied_clusters = counts[
        counts == max_count
    ].index.tolist()

    return int(min(tied_clusters))


district_clusters = (
    prediction_data
    .groupby("district")["cluster_id"]
    .agg(get_main_cluster)
)


# ==========================================
# 10. Calculate District Feature Averages
# ==========================================

district_features = (
    features.assign(
        district=cases["district"].values
    )
    .groupby("district")[feature_columns]
    .mean()
)


# ==========================================
# 11. Calculate District Representative Coordinates
# ==========================================

valid_coordinates = cases.dropna(
    subset=["lat", "lon"]
).copy()

valid_coordinates = valid_coordinates[
    valid_coordinates["lat"].between(-90, 90)
    & valid_coordinates["lon"].between(-180, 180)
]

district_coordinates = (
    valid_coordinates
    .groupby("district")[["lat", "lon"]]
    .mean()
)

if district_coordinates.empty:
    raise ValueError(
        "ไม่พบข้อมูลพิกัดของเขต"
    )


# ==========================================
# 12. Find Feature Values
# ==========================================

def find_feature_value(row, keywords):

    for column in row.index:

        column_name = str(column).lower()

        if any(
            keyword in column_name
            for keyword in keywords
        ):

            value = row[column]

            if pd.notna(value):
                return float(value)

    return None


def get_district_details(district):

    row = district_features.loc[district]

    return {
        "population": find_feature_value(
            row,
            ["population", "ประชากร"],
        ),

        "risk": find_feature_value(
            row,
            ["risk", "ความเสี่ยง"],
        ),

        "health": find_feature_value(
            row,
            [
                "health",
                "hospital",
                "สุขภาพ",
                "โรงพยาบาล",
            ],
        ),

        "schools": find_feature_value(
            row,
            ["school", "โรงเรียน"],
        ),

        "malls": find_feature_value(
            row,
            ["mall", "shopping", "ห้าง"],
        ),

        "temples": find_feature_value(
            row,
            ["temple", "วัด"],
        ),

        "universities": find_feature_value(
            row,
            ["universities", "university", "มหาวิทยาลัย"],
        ),
    }


# ==========================================
# 13. Create District Information
# ==========================================

district_data = []

for district, cluster_id in district_clusters.items():

    cluster_id = int(cluster_id)

    cluster_name = group_names.get(
        cluster_id,
        f"Cluster {cluster_id}",
    )

    district_data.append({
        "name": str(district),

        "district": str(district),

        "cluster_id": cluster_id,

        "cluster": cluster_name,

        "reasons": [
            (
                "Random Forest จัดพื้นที่นี้ "
                f"อยู่ในกลุ่ม {cluster_name}"
            )
        ],

        "details": get_district_details(district),
    })


# ==========================================
# 14. Validate User Preferences
# ==========================================

def validate_preferences(preferences):

    if not isinstance(preferences, dict):
        raise ValueError(
            "Missing user preferences"
        )

    work_location = preferences.get(
        "workLocation"
    )

    if (
        not isinstance(work_location, str)
        or not work_location.strip()
    ):
        raise ValueError(
            "กรุณาระบุสถานที่ทำงาน"
        )

    try:
        distance = float(
            preferences.get("distance")
        )

    except (TypeError, ValueError):
        raise ValueError(
            "ระยะทางไม่ถูกต้อง"
        )

    if (
        not np.isfinite(distance)
        or not 1 <= distance <= 20
    ):
        raise ValueError(
            "ระยะทางต้องอยู่ระหว่าง 1-20 กิโลเมตร"
        )

    travel = preferences.get("travel")

    if (
        not isinstance(travel, list)
        or len(travel) == 0
    ):
        raise ValueError(
            "กรุณาเลือกวิธีการเดินทาง"
        )

    factors = preferences.get("factors")

    if not isinstance(factors, dict):
        raise ValueError(
            "Invalid user factors"
        )

    clean_factors = {}

    for factor in FACTOR_ORDER:

        try:
            value = float(
                factors.get(factor, 3)
            )

        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid factor value: {factor}"
            )

        if (
            not np.isfinite(value)
            or not 1 <= value <= 5
        ):
            raise ValueError(
                f"Factor must be between 1 and 5: {factor}"
            )

        clean_factors[factor] = value

    return {
        "workLocation": work_location.strip(),
        "distance": distance,
        "travel": travel,
        "factors": clean_factors,
    }


# ==========================================
# 15. Filter Districts by Distance
# ==========================================

def filter_districts_by_distance(
    districts,
    work_coordinates,
    max_distance,
):
    """
    กรองเขตตามระยะทางเส้นตรง
    จากสถานที่ทำงานถึงพิกัดตัวแทนของแต่ละเขต
    """

    filtered_districts = []

    work_lat = work_coordinates["lat"]

    work_lon = work_coordinates["lon"]

    for district in districts:

        district_name = district["name"]

        if district_name not in district_coordinates.index:
            continue

        district_location = (
            district_coordinates.loc[district_name]
        )

        distance_km = calculate_distance(
            work_lat,
            work_lon,
            district_location["lat"],
            district_location["lon"],
        )

        if distance_km <= max_distance:

            candidate = dict(district)

            candidate["distance_km"] = round(
                distance_km,
                2,
            )

            candidate["reasons"] = [
                district["reasons"][0],
                (
                    "อยู่ห่างจากสถานที่ทำงานประมาณ "
                    f"{distance_km:.2f} กิโลเมตร "
                    "(ระยะเส้นตรง)"
                ),
            ]

            filtered_districts.append(candidate)

    return filtered_districts


# ==========================================
# 16. Sort Districts by User Priorities
# ==========================================

def get_factor_value(district, factor):

    details = district["details"]

    field_map = {
        "safety": "risk",
        "hospital": "health",
        "school": "schools",
        "shopping": "malls",
    }

    field = field_map.get(factor)

    if not field:
        return None

    return details.get(field)


def get_sort_key(district, priority_factors):

    """
    เรียงลำดับทีละปัจจัยตามความสำคัญ

    ไม่มีคะแนนรวม
    ไม่มีการคูณ Weight
    ไม่มีการแก้ไข Model
    """

    sort_values = []

    for factor in priority_factors:

        config = SUPPORTED_FACTORS.get(factor)

        # ข้ามปัจจัยที่ไม่มีข้อมูลใน Model
        if config is None:
            continue

        value = get_factor_value(
            district,
            factor,
        )

        if value is None or not np.isfinite(value):

            sort_values.append((1, 0))

            continue

        if config["direction"] == "desc":

            sort_values.append((0, -value))

        else:

            sort_values.append((0, value))

    return tuple(sort_values) + (
        district["name"],
    )


# ==========================================
# 17. Select Top 5 Districts
# ==========================================

def select_top_districts(
    districts,
    factors,
    limit=MAX_RECOMMENDATIONS,
):

    # เรียงปัจจัยตามความสำคัญที่ผู้ใช้เลือก
    priority_factors = sorted(
        FACTOR_ORDER,
        key=lambda factor: (
            -factors[factor],
            FACTOR_ORDER.index(factor),
        ),
    )

    # เรียงเขตตามปัจจัยที่รองรับ
    sorted_districts = sorted(
        districts,
        key=lambda district: get_sort_key(
            district,
            priority_factors,
        ),
    )

    # เลือก 5 เขตแรกโดยตรง
    selected = sorted_districts[:limit]

    supported_priorities = [
        FACTOR_LABELS[factor]
        for factor in priority_factors
        if factor in SUPPORTED_FACTORS
    ]

    for district in selected:

        district["reasons"].append(
            (
                "จัดลำดับตามปัจจัยที่มีข้อมูลรองรับ "
                "ตามความสำคัญที่ผู้ใช้เลือก"
            )
        )

    return selected, priority_factors


# ==========================================
# 18. Worker Ready Message
# ==========================================

print(
    json.dumps(
        {
            "ready": True,

            "district_count": len(district_clusters),

            "model_features": feature_columns,

            "recommendation_limit": MAX_RECOMMENDATIONS,
            # บัวเพิ่มจ่ะพี่หน้าเปรียบเทียบเขต (Compare)
            "district_profiles": build_district_profiles(
                bundle,
                cases,
                features,
                predictions,
            ),
        },
        ensure_ascii=True,
    ),
    flush=True,
)


# ==========================================
# 19. Handle Requests from Node.js
# ==========================================

for line in sys.stdin:

    line = line.strip()

    if not line:
        continue

    request = None

    try:

        request = json.loads(line)

        request_id = request.get("id")

        preferences = request.get(
            "preferences"
        )

        # ตรวจสอบข้อมูลผู้ใช้
        clean_preferences = validate_preferences(
            preferences
        )

        factors = clean_preferences["factors"]

        work_location = clean_preferences[
            "workLocation"
        ]

        max_distance = clean_preferences[
            "distance"
        ]

        # ----------------------------------
        # 1. Geocode Work Location
        # ----------------------------------

        work_coordinates = geocode_location(
            work_location
        )

        # ----------------------------------
        # 2. Filter Districts by Distance
        # ----------------------------------

        nearby_districts = filter_districts_by_distance(
            district_data,
            work_coordinates,
            max_distance,
        )

        # ----------------------------------
        # 3. Sort and Select Top 5
        # ----------------------------------

        recommendations, priority_factors = (
            select_top_districts(
                nearby_districts,
                factors,
                MAX_RECOMMENDATIONS,
            )
        )

        # ----------------------------------
        # 4. Limitations
        # ----------------------------------

        limitations = [
            "ผลลัพธ์เป็นการจัดกลุ่มพื้นที่ด้วย Random Forest",

            "แต่ละเขตถูกกำหนด Cluster จากกลุ่มที่โมเดลทำนายพบมากที่สุด",

            "ใช้ Model และ Feature เดิม ไม่มีการ Train ใหม่",

            "กรองเขตตามระยะทางก่อนจัดอันดับผลลัพธ์",

            "ระยะทางคำนวณด้วยสูตร Haversine เป็นระยะเส้นตรง ไม่ใช่ระยะทางตามถนน",

            "ใช้ค่าเฉลี่ยพิกัด lat และ lon ของจุดข้อมูลในแต่ละเขตเป็นพิกัดตัวแทน",

            "จัดอันดับตามปัจจัยที่มีข้อมูลรองรับทีละข้อ ไม่ใช้คะแนนรวมหรือ Weight",

            "แสดงผลลัพธ์สูงสุด 5 เขต",

            "สวนสาธารณะและระบบขนส่งยังไม่มี Feature ใน Model เดิม จึงไม่ถูกนำมาเรียงลำดับ",

            "วิธีเดินทางและประเภทการทำงานยังไม่ได้ใช้คำนวณเส้นทางหรือเวลาเดินทางจริง",
        ]

        if not recommendations:

            limitations.append(
                (
                    "ไม่พบเขตที่อยู่ภายในระยะ "
                    f"{max_distance:g} กิโลเมตร "
                    "จากสถานที่ทำงานที่ระบุ"
                )
            )

        # ----------------------------------
        # 5. Build Response
        # ----------------------------------

        response = {
            "id": request_id,

            "result": {
                "recommendations": recommendations,

                "limitations": limitations,

                "priority_order": [
                    FACTOR_LABELS[factor]
                    for factor in priority_factors
                ],

                "work_location": work_location,

                "work_coordinates": work_coordinates,

                "max_distance_km": max_distance,

                "matched_district_count": len(
                    nearby_districts
                ),
            },
        }

        # Debug
        print(
            (
                f"Location: {work_location} | "
                f"Max distance: {max_distance} km | "
                f"Nearby districts: {len(nearby_districts)} | "
                f"Recommendations: {len(recommendations)}"
            ),
            file=sys.stderr,
            flush=True,
        )

        # ส่งผลลัพธ์ให้ Node.js
        print(
            json.dumps(
                response,
                ensure_ascii=True,
            ),
            flush=True,
        )

    except Exception as error:

        print(
            json.dumps(
                {
                    "id": (
                        request.get("id")
                        if isinstance(request, dict)
                        else None
                    ),

                    "error": str(error),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

        print(
            traceback.format_exc(),
            file=sys.stderr,
            flush=True,
        )
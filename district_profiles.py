"""สรุปผลของโมเดลเป็นรายเขต สำหรับหน้าเปรียบเทียบเขต (Compare)

ml_worker.py เรียก build_district_profiles(...) ตอนเริ่มทำงาน
โดยส่งผลทำนายที่ ml_worker.py คำนวณไว้แล้วเข้ามา ไฟล์นี้จึงไม่ได้รันโมเดลซ้ำ
และไม่ได้เทรนหรือแก้โมเดล
"""

import numpy as np


def build_district_profiles(bundle, cases, features, predictions):
    columns = bundle["feature_columns"]
    group_names = {int(g): name for g, name in bundle["group_names"].items()}
    groups = sorted(group_names)

    labels = np.asarray(predictions).astype(int)
    district_of_point = cases["district"].to_numpy()
    population = features["population"].groupby(district_of_point).first()

    districts = []

    for name in sorted(population.index):
        in_district = district_of_point == name
        count = int(in_district.sum())
        district_labels = labels[in_district]

        # สัดส่วนจุดในเขตที่โมเดลจัดให้อยู่แต่ละกลุ่ม
        shares = {
            str(g): round(float(np.mean(district_labels == g)), 4) if count else 0.0
            for g in groups
        }

        # ค่าเฉลี่ยจำนวนสถานที่ในรัศมี 1 กม. ของทุกจุดในเขต
        nearby_average = {
            column: round(float(features.loc[in_district, column].mean()), 3) if count else None
            for column in columns
            if column != "population"
        }

        districts.append({
            "id": str(name),
            "name": str(name),
            "population": int(population[name]),
            "location_count": count,
            "nearby_average": nearby_average,
            "group_shares": shares,
            # ถ้าสัดส่วนเท่ากัน เลือกกลุ่มเลขน้อยกว่า (แบบเดียวกับ get_main_cluster)
            "dominant_group": max(groups, key=lambda g: (shares[str(g)], -g)) if count else None,
        })

    return {
        "meta": {
            "feature_columns": columns,
            "groups": [{"id": g, "name": group_names[g]} for g in groups],
            "nearby_radius_km": bundle.get("nearby_radius_km"),
            "sklearn_version": bundle.get("sklearn_version"),
        },
        "districts": districts,
    }
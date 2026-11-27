const fs = require("node:fs");
const path = require("node:path");
const { parse } = require("csv-parse/sync");

const dataDir = path.join(__dirname, "data");
const thaiSort = new Intl.Collator("th");

function readCsv(filename) {
  return parse(fs.readFileSync(path.join(dataDir, filename), "utf8"), {
    bom: true,
    columns: true,
    skip_empty_lines: true,
    trim: true,
  });
}

function count(value) {
  const text = String(value ?? "").trim().replace(/,/g, "");
  if (!/^\d+$/.test(text)) return null;

  const number = Number(text);
  return Number.isSafeInteger(number) ? number : null;
}

function districtKey(value) {
  return String(value ?? "").normalize("NFC").trim().replace(/^เขต\s*/, "");
}

function loadDashboard() {
  const districts = readCsv("bangkok_population_clean.csv").map((row) => {
    const id = districtKey(row["เขต"]);

    return {
      id,
      name: `เขต${id}`,
      population: count(row["รวม"]),
    };
  });

  const byId = new Map(districts.map((district) => [district.id, district]));
  const byCode = new Map();
  const placeCountsByDistrict = Object.fromEntries(
    districts.map(({ id }) => [
      id,
      { temple: 0, mall: 0, health: 0, school: 0, university: 0 },
    ]),
  );

  const placeSources = [
    { type: "temple", file: "bangkok_temples_clean.csv", districtColumn: "district" },
    { type: "mall", file: "cleaned ที่ตั้งห้างสรรพสินค้า.csv", districtColumn: "dname" },
    { type: "health", file: "health_centers_clean.csv", districtColumn: "DISTRICT" },
    { type: "school", file: "schools_clean.csv", districtColumn: "เขต_ชื่อล้วน" },
    { type: "university", file: "universities_clean.csv", districtColumn: "DISTRICT" },
  ];

  for (const { type, file, districtColumn } of placeSources) {
    for (const row of readCsv(file)) {
      const id = districtKey(row[districtColumn]);
      const district = byId.get(id);
      if (!district) continue;

      placeCountsByDistrict[id][type] += 1;

      if (/^\d{4}$/.test(row.dcode ?? "")) {
        byCode.set(row.dcode, district);
      }
    }
  }

  for (const district of districts) {
    district.placeCount = Object.values(placeCountsByDistrict[district.id])
      .reduce((sum, value) => sum + value, 0);
  }

  const housingRows = readCsv(
    "cleaned จำนวนบ้านในกรุงเทพมหานคร พ.ศ. 2556 - 2565.csv",
  );
  const yearColumns = Object.keys(housingRows[0] ?? {})
    .filter((column) => /^Y\d{4}$/.test(column))
    .sort();

  const housing = {
    years: yearColumns.map((column) => Number(column.slice(1))),
    districts: housingRows
      .map((row) => {
        const id = districtKey(row.district);
        return {
          id,
          district: id,
          name: byId.get(id)?.name,
          counts: yearColumns.map((column) => count(row[column])),
        };
      })
      .filter(({ id, counts }) =>
        byId.has(id) && counts.length > 0 && counts.every(Number.isSafeInteger),
      ),
  };

  const housingById = new Map(housing.districts.map((row) => [row.id, row]));

  const stations = readCsv("bts_passengers_clean.csv")
    .map((row, index) => ({
      id: index + 1,
      name: row.STATION_NAME,
      total: count(row.TOTAL_PASSENGER),
    }))
    .filter(({ name, total }) => name && total !== null);

  districts.sort((a, b) => thaiSort.compare(a.name, b.name));

  const dashboard = {
    summary: {
      districtCount: districts.length,
      population: districts.reduce((sum, row) => sum + (row.population ?? 0), 0),
      placeCount: districts.reduce((sum, row) => sum + row.placeCount, 0),
    },
    districts,
    populationDistricts: districts,
    placeCountsByDistrict,
    housing,
    bts: { stations },
    clusters: [],
    zones: [],
    transitStops: [],
    years: {},
  };

  return { dashboard, byId, byCode, housingById };
}

let cachedData;

function handleDashboardRoute(req, res, sendJson) {
  if (req.method !== "GET") return false;

  const url = new URL(req.url, "http://localhost");
  const route = url.pathname;
  const detail = /^\/api\/districts\/([^/]+)$/.exec(route);
  const isDistrictDetail = detail && route !== "/api/districts/profiles";

  if (
    route !== "/api/dashboard" &&
    route !== "/api/districts" &&
    route !== "/api/search" &&
    !isDistrictDetail
  ) {
    return false;
  }

  try {
    cachedData ??= loadDashboard();
    const { dashboard, byId, byCode, housingById } = cachedData;

    if (route === "/api/dashboard") {
      sendJson(res, 200, dashboard);
      return true;
    }

    if (route === "/api/districts") {
      sendJson(res, 200, dashboard.districts);
      return true;
    }

    if (route === "/api/search") {
      const query = districtKey(url.searchParams.get("q"));
      const requestedLimit = count(url.searchParams.get("limit")) ?? 10;
      const limit = Math.min(Math.max(requestedLimit, 1), 50);

      const results = query
        ? dashboard.districts.filter(({ id }) => id.includes(query)).slice(0, limit)
        : [];

      sendJson(res, 200, { results });
      return true;
    }

    const input = districtKey(decodeURIComponent(detail[1]));
    const district = byId.get(input) ?? byCode.get(input);

    if (!district) {
      sendJson(res, 404, { error: "ไม่พบเขตที่ระบุ" });
      return true;
    }

    const housing = housingById.get(district.id);

    sendJson(res, 200, {
      ...district,
      placeCounts: dashboard.placeCountsByDistrict[district.id],
      households: housing
        ? { years: dashboard.housing.years, counts: housing.counts }
        : null,
    });
  } catch (error) {
    if (error instanceof URIError) {
      sendJson(res, 400, { error: "รหัสหรือชื่อเขตไม่ถูกต้อง" });
    } else {
      console.error("Dashboard data error:", error);
      sendJson(res, 500, { error: "โหลดข้อมูล Dashboard จากโฟลเดอร์ data ไม่สำเร็จ" });
    }
  }

  return true;
}

module.exports = { handleDashboardRoute };
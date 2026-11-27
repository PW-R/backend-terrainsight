// API ข้อมูลสำหรับ frontend (Dashboard + ค้นหาเขต)
// อ่าน CSV ใน ./data ครั้งเดียวตอนเริ่ม server แล้วตอบเป็น JSON
// ไฟล์นี้ไม่แตะโมเดลหรือ /predict เลย
//
// ใช้ได้ 2 แบบ:
//   1) ต่อกับ server.js ของเดิม (ดู README-API.md)
//   2) รันเดี่ยว: node data-api.js   (port 3001 หรือ DATA_PORT)

const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");

const DATA_DIR = path.join(__dirname, "data");

// รหัสเขตของกรุงเทพมหานคร ใช้เป็น id กลางของทุก endpoint
// ตอนเริ่ม server จะเทียบกับคอลัมน์ dcode ในไฟล์ห้างและมหาวิทยาลัยอีกรอบ
const DISTRICT_CODES = {
  พระนคร: 1001, ดุสิต: 1002, หนองจอก: 1003, บางรัก: 1004, บางเขน: 1005,
  บางกะปิ: 1006, ปทุมวัน: 1007, ป้อมปราบศัตรูพ่าย: 1008, พระโขนง: 1009, มีนบุรี: 1010,
  ลาดกระบัง: 1011, ยานนาวา: 1012, สัมพันธวงศ์: 1013, พญาไท: 1014, ธนบุรี: 1015,
  บางกอกใหญ่: 1016, ห้วยขวาง: 1017, คลองสาน: 1018, ตลิ่งชัน: 1019, บางกอกน้อย: 1020,
  บางขุนเทียน: 1021, ภาษีเจริญ: 1022, หนองแขม: 1023, ราษฎร์บูรณะ: 1024, บางพลัด: 1025,
  ดินแดง: 1026, บึงกุ่ม: 1027, สาทร: 1028, บางซื่อ: 1029, จตุจักร: 1030,
  บางคอแหลม: 1031, ประเวศ: 1032, คลองเตย: 1033, สวนหลวง: 1034, จอมทอง: 1035,
  ดอนเมือง: 1036, ราชเทวี: 1037, ลาดพร้าว: 1038, วัฒนา: 1039, บางแค: 1040,
  หลักสี่: 1041, สายไหม: 1042, คันนายาว: 1043, สะพานสูง: 1044, วังทองหลาง: 1045,
  คลองสามวา: 1046, บางนา: 1047, ทวีวัฒนา: 1048, ทุ่งครุ: 1049, บางบอน: 1050,
};

// ประเภทสถานที่: key ตรงกับ placeTypes ใน Dashboard
// เงื่อนไขกรองใช้แบบเดียวกับ train_and_save_model.py (จุดเสี่ยงเอาปีล่าสุด, ตัดพิกัดผิดปกติ)
const PLACE_SOURCES = [
  { key: "temple", file: "bangkok_temples_clean.csv", district: "district", name: "temple_name", lat: "latitude", lng: "longitude" },
  { key: "mall", file: "cleaned ที่ตั้งห้างสรรพสินค้า.csv", district: "dname", name: "name", id: "id_depart", lat: "lat", lng: "lng" },
  { key: "health", file: "health_centers_clean.csv", district: "DISTRICT", name: "BHCName", lat: "Latitude", lng: "longitude", flag: "coord_suspect" },
  { key: "school", file: "schools_clean.csv", district: "เขต_ชื่อล้วน", name: "ชื่อสถานศึกษา", id: "รหัสสถานศึกษา", lat: "ละติจูด", lng: "ลองจิจูด", flag: "พิกัดผิดปกติ" },
  { key: "university", file: "universities_clean.csv", district: "DISTRICT", name: "name", id: "id", lat: "lat", lng: "lng" },
  { key: "risk", file: "risk_points_clean.csv", district: "AreaDist", name: "AName", lat: "ALatitude", lng: "ALongitude", latestYearColumn: "AFiscalYeal" },
];
// ไม่นับจุดเสี่ยงรวมใน "สถานที่สำคัญ"
const LANDMARK_KEYS = ["temple", "mall", "health", "school", "university"];

// ---------- อ่าน CSV (รองรับช่องที่มี , อยู่ในเครื่องหมายคำพูด) ----------
function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") { row.push(field); field = ""; }
    else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += char;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }

  const [header, ...body] = rows;
  const columns = header.map((name) => name.replace(/^﻿/, "").trim());
  return body.map((values) => Object.fromEntries(columns.map((name, index) => [name, values[index] ?? ""])));
}

function readCsv(fileName) {
  const filePath = path.join(DATA_DIR, fileName);
  if (!fs.existsSync(filePath)) throw new Error(`ไม่พบไฟล์ข้อมูล: data/${fileName}`);
  return parseCsv(fs.readFileSync(filePath, "utf8"));
}

// ไฟล์จำนวนบ้านชื่อยาวและเป็นภาษาไทย หาไฟล์ที่ลงท้ายด้วยช่วงปี 2556-2565
function findHouseholdFile() {
  const match = fs.readdirSync(DATA_DIR).find((name) => /2556.*2565.*\.csv$/i.test(name));
  if (!match) throw new Error("ไม่พบไฟล์จำนวนบ้าน (ชื่อไฟล์ควรมี 2556 ... 2565)");
  return match;
}

// ทำชื่อเขตให้เหมือนกันทุกไฟล์ (เหมือน normalize_district ใน Python ของเพื่อน)
function normalizeDistrict(value) {
  const name = String(value ?? "").replace(/\s+/g, "").replace(/^(เขต|อำเภอ)/, "");
  return name === "สัมพันธ์วงศ์" ? "สัมพันธวงศ์" : name;
}

function toNumber(value) {
  const number = Number(String(value ?? "").replace(/,/g, "").trim());
  return value === "" || !Number.isFinite(number) ? null : number;
}

function isFlagged(value) {
  return ["true", "1"].includes(String(value).trim().toLowerCase());
}

// ---------- สร้างข้อมูลทั้งหมดตอนเริ่ม ----------
function buildData() {
  const warnings = [];

  // ประชากร = รายชื่อ 50 เขตหลัก
  const districts = readCsv("bangkok_population_clean.csv").map((row) => {
    const shortName = normalizeDistrict(row["เขต"]);
    const code = DISTRICT_CODES[shortName];
    if (!code) throw new Error(`ไม่รู้จักรหัสเขต: ${shortName}`);
    return {
      id: String(code),
      name: `เขต${shortName}`,
      shortName,
      population: toNumber(row["รวม"]),
      male: toNumber(row["ชาย"]),
      female: toNumber(row["หญิง"]),
      area: null, // ยังไม่มีข้อมูลพื้นที่เขตใน backend
    };
  }).sort((a, b) => Number(a.id) - Number(b.id));

  const byShortName = new Map(districts.map((district) => [district.shortName, district]));

  // เทียบรหัสเขตกับคอลัมน์ dcode ในข้อมูลจริง
  for (const [file, column] of [["universities_clean.csv", "DISTRICT"], ["cleaned ที่ตั้งห้างสรรพสินค้า.csv", "dname"]]) {
    for (const row of readCsv(file)) {
      const district = byShortName.get(normalizeDistrict(row[column]));
      if (district && row.dcode && String(toNumber(row.dcode)) !== district.id) {
        warnings.push(`dcode ไม่ตรง: ${district.name} ตาราง=${district.id} ไฟล์ ${file}=${row.dcode}`);
      }
    }
  }

  // สถานที่แต่ละประเภท
  const places = [];
  const years = {};
  for (const source of PLACE_SOURCES) {
    let rows = readCsv(source.file);

    if (source.latestYearColumn) {
      const latest = Math.max(...rows.map((row) => toNumber(row[source.latestYearColumn]) ?? -Infinity));
      rows = rows.filter((row) => toNumber(row[source.latestYearColumn]) === latest);
      years[source.key] = latest;
    }

    const seen = new Set();
    for (const row of rows) {
      const district = byShortName.get(normalizeDistrict(row[source.district]));
      if (!district) {
        warnings.push(`${source.file}: ไม่รู้จักเขต "${row[source.district]}"`);
        continue;
      }
      const lat = toNumber(row[source.lat]);
      const lng = toNumber(row[source.lng]);
      const validCoordinate = lat !== null && lng !== null && lat >= 13 && lat <= 14 && lng >= 100 && lng <= 101
        && !(source.flag && isFlagged(row[source.flag]));
      const uniqueKey = `${row[source.id ?? source.name]}|${row[source.lat]}|${row[source.lng]}`;
      if (seen.has(uniqueKey)) continue;
      seen.add(uniqueKey);

      places.push({
        type: source.key,
        name: String(row[source.name] ?? "").trim(),
        districtId: district.id,
        lat: validCoordinate ? lat : null,
        lng: validCoordinate ? lng : null,
      });
    }
  }

  // นับสถานที่รายเขต + จุดกึ่งกลางแผนที่ (ค่าเฉลี่ยพิกัดสถานที่ในเขต)
  for (const district of districts) {
    const inDistrict = places.filter((place) => place.districtId === district.id);
    district.placeCounts = Object.fromEntries(PLACE_SOURCES.map(({ key }) => [
      key, inDistrict.filter((place) => place.type === key).length,
    ]));
    district.poiCount = LANDMARK_KEYS.reduce((sum, key) => sum + district.placeCounts[key], 0);

    const located = inDistrict.filter((place) => place.lat !== null);
    district.mapCenter = located.length
      ? [
        Number((located.reduce((sum, place) => sum + place.lat, 0) / located.length).toFixed(5)),
        Number((located.reduce((sum, place) => sum + place.lng, 0) / located.length).toFixed(5)),
      ]
      : null;
  }

  // จำนวนบ้านรายปี
  const householdRows = readCsv(findHouseholdFile());
  const householdYears = Object.keys(householdRows[0])
    .map((column) => /^Y(\d{4})$/.exec(column)?.[1])
    .filter(Boolean)
    .map(Number);
  for (const row of householdRows) {
    const district = byShortName.get(normalizeDistrict(row.district));
    if (!district) {
      warnings.push(`จำนวนบ้าน: ไม่รู้จักเขต "${row.district}"`);
      continue;
    }
    district.households = householdYears.map((year) => toNumber(row[`Y${year}`]));
  }

  // กลุ่มเขต 6 กลุ่ม (กรุงเทพกลาง, กรุงธนใต้ ฯลฯ) จากคอลัมน์ school_group
  const zoneVotes = new Map();
  for (const row of readCsv("open_education_clean.csv")) {
    const district = byShortName.get(normalizeDistrict(row.district));
    const zone = String(row.school_group ?? "").trim().replace(/^กลุ่ม/, "");
    if (!district || !zone) continue;
    const votes = zoneVotes.get(district.id) ?? new Map();
    votes.set(zone, (votes.get(zone) ?? 0) + 1);
    zoneVotes.set(district.id, votes);
  }
  for (const district of districts) {
    const votes = [...(zoneVotes.get(district.id) ?? [])].sort((a, b) => b[1] - a[1]);
    district.zone = votes[0]?.[0] ?? null;
    if (votes.length > 1) warnings.push(`${district.name} อยู่หลายกลุ่มเขต ใช้ ${district.zone}`);
    if (!district.zone) warnings.push(`${district.name} ไม่มีข้อมูลกลุ่มเขต`);
  }

  // ผู้โดยสาร BTS รายสถานี (ไฟล์ไม่มีคอลัมน์เขต)
  const months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
  const btsStations = readCsv("bts_passengers_clean.csv").map((row) => ({
    name: row.STATION_NAME.trim(),
    total: toNumber(row.TOTAL_PASSENGER),
    monthly: months.map((month) => toNumber(row[`${month}_PASSENGER`])),
  })).sort((a, b) => b.total - a.total);

  return { districts, places, householdYears, btsStations, years, warnings };
}

// โหลดข้อมูลแบบไม่ล้ม: ถ้าไฟล์ CSV หาย จะปิดแค่ route ของ Dashboard (ตอบ 503)
// server.js ของเพื่อน (/predict, /api/districts/profiles) ยังทำงานต่อได้ตามปกติ
let DATA = null;
let LOAD_ERROR = null;
try {
  DATA = buildData();
  for (const warning of DATA.warnings) console.warn(`[data-api] ${warning}`);
  console.log(`[data-api] โหลดข้อมูลแล้ว ${DATA.districts.length} เขต, ${DATA.places.length} สถานที่`);
} catch (error) {
  LOAD_ERROR = error;
  console.error(`[data-api] โหลดข้อมูล Dashboard ไม่สำเร็จ: ${error.message}`);
}

// ---------- รูปแบบข้อมูลที่ส่งให้ frontend ----------
function districtSummary(district) {
  const { id, name, shortName, zone, population, area, poiCount, placeCounts, mapCenter } = district;
  return { id, name, shortName, zone, population, area, poiCount, placeCounts, mapCenter };
}

function dashboardPayload() {
  const { districts, householdYears, btsStations, years } = DATA;
  return {
    isExample: false,
    years: {
      population: null, // ไฟล์ประชากรไม่ระบุปี
      places: null,
      risk: years.risk ?? null,
      households: { from: householdYears[0], to: householdYears.at(-1) },
    },
    summary: {
      districtCount: districts.length,
      population: districts.reduce((sum, district) => sum + (district.population ?? 0), 0),
      placeCount: districts.reduce((sum, district) => sum + district.poiCount, 0),
    },
    clusters: [], // โมเดลจัดกลุ่มรายพิกัด ยังไม่มีผลรายเขต
    populationDistricts: districts.map(({ id, name, population }) => ({ id, name, population })),
    districts: districts.map(({ id, name, population, area }) => ({ id, name, population, area })),
    placeCountsByDistrict: Object.fromEntries(districts.map(({ id, placeCounts }) => [id, placeCounts])),
    housing: {
      years: householdYears,
      districts: districts.map(({ id, name, shortName, households }) => ({ id, name, district: shortName, counts: households })),
    },
    bts: { stations: btsStations },
    // พื้นที่ ตร.กม. ไม่มีใน backend frontend ใส่เอง
    zones: [...new Set(districts.map(({ zone }) => zone).filter(Boolean))].map((zone) => ({
      name: zone,
      districts: districts.filter((district) => district.zone === zone).map(({ id, name }) => ({ id, name })),
    })),
  };
}

function districtDetail(district) {
  return {
    ...districtSummary(district),
    male: district.male,
    female: district.female,
    totalDistricts: DATA.districts.length,
    households: { years: DATA.householdYears, counts: district.households },
    places: DATA.places
      .filter((place) => place.districtId === district.id && place.type !== "risk")
      .map(({ type, name, lat, lng }) => ({ type, name, lat, lng })),
  };
}

function searchDistricts(query, limit) {
  const keyword = normalizeDistrict(query).toLocaleLowerCase("th-TH");
  if (!keyword) return [];
  return DATA.districts
    .filter((district) => district.shortName.toLocaleLowerCase("th-TH").includes(keyword) || district.id === keyword)
    .sort((a, b) => Number(!a.shortName.startsWith(keyword)) - Number(!b.shortName.startsWith(keyword)))
    .slice(0, limit)
    .map(districtSummary);
}

function findDistrict(idOrName) {
  const value = decodeURIComponent(idOrName);
  return DATA.districts.find((district) => district.id === value)
    ?? DATA.districts.find((district) => district.shortName === normalizeDistrict(value));
}

function send(res, status, body) {
  if (res.destroyed || res.writableEnded) return;
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "public, max-age=300",
  });
  res.end(JSON.stringify(body));
}

// ตอบเฉพาะ route ของไฟล์นี้ แล้วคืนค่า true
// route อื่น (เช่น /api/districts/profiles หรือ POST ของเพื่อน) คืนค่า false ให้ server.js จัดการต่อ
function handleDataRequest(req, res) {
  if (req.method !== "GET") return false;

  const url = new URL(req.url, "http://localhost");
  const route = url.pathname.replace(/\/+$/, "");
  const isOurRoute = ["/api/dashboard", "/api/districts", "/api/search"].includes(route)
    || /^\/api\/districts\/[^/]+$/.test(route);
  if (!isOurRoute) return false;
  if (!DATA) {
    // ไม่แย่ง /api/districts/xxx ที่ไม่ใช่ของเรา (เช่น profiles) ตอนข้อมูลเราโหลดไม่ได้
    if (route.startsWith("/api/districts/")) return false;
    send(res, 503, { error: `ข้อมูล Dashboard ยังไม่พร้อม: ${LOAD_ERROR?.message ?? "unknown"}` });
    return true;
  }
  try {
    return routeDataRequest(route, url, res);
  } catch (error) {
    console.error("[data-api]", error);
    send(res, 500, { error: "data-api error" });
    return true;
  }
}

function routeDataRequest(route, url, res) {

  if (route === "/api/dashboard") {
    send(res, 200, dashboardPayload());
    return true;
  }
  if (route === "/api/districts") {
    send(res, 200, DATA.districts.map(districtSummary));
    return true;
  }
  if (route === "/api/search") {
    const query = url.searchParams.get("q") ?? "";
    const limit = Math.min(Math.max(Number(url.searchParams.get("limit")) || 10, 1), 50);
    send(res, 200, { query, results: searchDistricts(query, limit) });
    return true;
  }

  // /api/districts/1026 หรือ /api/districts/ดินแดง (ส่วนเดียว และต้องเป็นเขตที่มีจริง)
  const match = /^\/api\/districts\/([^/]+)$/.exec(route);
  const district = match && findDistrict(match[1]);
  if (district) {
    send(res, 200, districtDetail(district));
    return true;
  }
  return false;
}

module.exports = { handleDataRequest, parseCsv, normalizeDistrict };

// รันเดี่ยวสำหรับทดสอบหรือเมื่อโมเดลยังโหลดไม่ได้
if (require.main === module) {
  const port = Number(process.env.DATA_PORT || 3001);
  http.createServer((req, res) => {
    if (!handleDataRequest(req, res)) send(res, 404, { error: "Route not found" });
  }).listen(port, () => console.log(`[data-api] listening on port ${port}`));
}

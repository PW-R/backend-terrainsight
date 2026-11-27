// data-api.js: API รายละเอียดเขต สำหรับช่องค้นหาบน navbar
//   GET /api/district-details/:ชื่อเขตหรือรหัสเขต
// อ่านข้อมูลจาก CSV ใน ./data ตอนเริ่ม server ทั้งหมด (ไม่มีข้อมูลเขียนไว้ในโค้ด)
// ไฟล์นี้ไม่แตะโมเดล /predict และไม่ยุ่งกับ route Dashboard ของ dashboardRoutes.js
//
// ใช้ได้ 2 แบบ:
//   1) ใน server.js: if (handleDataRequest(req, res)) return;  (วางต่อจาก handleDashboardRoute)
//   2) รันเดี่ยวเพื่อทดสอบ: node data-api.js   (port 3001 หรือ DATA_PORT)

const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");

const DATA_DIR = path.join(__dirname, "data");

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

// หาไฟล์จำนวนบ้านจากหัวคอลัมน์ (มีคอลัมน์ district และคอลัมน์ปีแบบ Y2556, Y2557, ...)
// ไม่ผูกกับชื่อไฟล์หรือช่วงปี เพิ่มปีใหม่ในไฟล์ได้เลย
function findHouseholdFile() {
  const match = fs.readdirSync(DATA_DIR)
    .filter((name) => name.toLowerCase().endsWith(".csv"))
    .find((name) => {
      const header = fs.readFileSync(path.join(DATA_DIR, name), "utf8").split(/\r?\n/, 1)[0].replace(/^\uFEFF/, "");
      const columns = header.split(",").map((column) => column.trim());
      return columns.includes("district") && columns.some((column) => /^Y\d{4}$/.test(column));
    });
  if (!match) throw new Error("ไม่พบไฟล์จำนวนบ้าน (ต้องมีคอลัมน์ district และคอลัมน์ปี เช่น Y2565)");
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

  // รหัสเขต (dcode) อ่านจากข้อมูล ไม่ได้เขียนไว้ในโค้ด:
  // ขอบเขตเขต data/bangkok-districts.json + คอลัมน์ dcode ในไฟล์ห้างและมหาวิทยาลัย
  const geoFile = path.join(DATA_DIR, "bangkok-districts.json");
  const geo = fs.existsSync(geoFile) ? JSON.parse(fs.readFileSync(geoFile, "utf8")) : null;
  if (!geo) warnings.push("ไม่พบ data/bangkok-districts.json จึงไม่มีพื้นที่และความหนาแน่น");

  // กรอบพิกัดกรุงเทพฯ คำนวณจากขอบเขตเขต ใช้ตัดพิกัดที่ผิดปกติ (ถ้าไม่มีไฟล์ขอบเขต จะไม่ตัด)
  const bounds = geo ? geometryBounds(geo.features.map(({ geometry }) => geometry)) : null;

  const codes = new Map();
  const addCode = (name, code) => {
    const shortName = normalizeDistrict(name);
    const value = toNumber(code);
    if (!shortName || value == null) return;
    if (codes.has(shortName) && codes.get(shortName) !== String(value)) {
      warnings.push(`dcode ไม่ตรงกัน: ${shortName} ${codes.get(shortName)} / ${value}`);
      return;
    }
    codes.set(shortName, String(value));
  };
  for (const feature of geo?.features ?? []) addCode(feature.properties?.dname, feature.properties?.dcode);
  for (const row of readCsv("cleaned ที่ตั้งห้างสรรพสินค้า.csv")) addCode(row.dname, row.dcode);
  for (const row of readCsv("universities_clean.csv")) addCode(row.DISTRICT, row.dcode);

  // ประชากร = รายชื่อ 50 เขตหลัก
  const districts = readCsv("bangkok_population_clean.csv").map((row) => {
    const shortName = normalizeDistrict(row["เขต"]);
    const code = codes.get(shortName);
    if (!code) warnings.push(`ไม่พบรหัสเขตของ ${shortName} ในข้อมูล ใช้ชื่อเขตเป็น id แทน`);
    return {
      id: code ?? shortName,
      name: `เขต${shortName}`,
      shortName,
      population: toNumber(row["รวม"]),
      male: toNumber(row["ชาย"]),
      female: toNumber(row["หญิง"]),
      area: null, // เติมจากขอบเขตเขตด้านล่าง
    };
  }).sort((a, b) => String(a.id).localeCompare(String(b.id)));

  const byShortName = new Map(districts.map((district) => [district.shortName, district]));

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
      const insideBounds = !bounds
        || (lat >= bounds.minLat && lat <= bounds.maxLat && lng >= bounds.minLng && lng <= bounds.maxLng);
      const validCoordinate = lat !== null && lng !== null && insideBounds
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


  // ---------- ข้อมูลเพิ่มสำหรับหน้ารายละเอียดเขต (ค้นหาจาก navbar) ----------

  // พื้นที่ ตร.กม. คำนวณจากขอบเขตเขตใน data/bangkok-districts.json (ถ้าไม่มีไฟล์ = null)
  if (geo) {
    for (const feature of geo.features ?? []) {
      const district = districts.find(({ id }) => id === String(feature.properties?.dcode))
        ?? byShortName.get(normalizeDistrict(feature.properties?.dname));
      if (district) district.area = Number((geometryArea(feature.geometry) / 1e6).toFixed(2));
    }
  }

  // โรงเรียนที่เปิดชั้นอนุบาล
  const kindergartens = new Map();
  for (const row of readCsv("kindergarten_clean.csv")) {
    const district = byShortName.get(normalizeDistrict(row.district));
    if (district) kindergartens.set(district.id, (kindergartens.get(district.id) ?? 0) + 1);
  }

  // จุดเสี่ยงกลุ่ม "จุดมืด จุดเปลี่ยว" ของปีล่าสุด
  const darkPoints = new Map();
  for (const row of readCsv("risk_points_clean.csv")) {
    const district = byShortName.get(normalizeDistrict(row.AreaDist));
    if (!district || toNumber(row.AFiscalYeal) !== years.risk) continue;
    if (String(row.AGroup).includes("จุดมืด")) darkPoints.set(district.id, (darkPoints.get(district.id) ?? 0) + 1);
  }

  // จำนวนนักศึกษามหาวิทยาลัย
  const students = new Map();
  for (const row of readCsv("universities_clean.csv")) {
    const district = byShortName.get(normalizeDistrict(row.DISTRICT));
    const count = toNumber(row.num_students);
    if (district && count != null) students.set(district.id, (students.get(district.id) ?? 0) + count);
  }

  for (const district of districts) {
    district.kindergartenCount = kindergartens.get(district.id) ?? 0;
    district.darkPointCount = darkPoints.get(district.id) ?? 0;
    district.students = students.get(district.id) ?? null;
    district.density = district.area > 0 && district.population != null
      ? Math.round(district.population / district.area)
      : null;
  }

  // อันดับประชากรและความหนาแน่น (1 = มากที่สุด)
  const rank = (key) => {
    const sorted = districts.filter((district) => district[key] != null).sort((a, b) => b[key] - a[key]);
    sorted.forEach((district, index) => { district[`${key}Rank`] = index + 1; });
  };
  rank("population");
  rank("density");

  return { districts, places, householdYears, years, warnings };
}

// กรอบสี่เหลี่ยมที่ครอบทุกเขต
function geometryBounds(geometries) {
  const bounds = { minLat: Infinity, maxLat: -Infinity, minLng: Infinity, maxLng: -Infinity };
  const visit = (coords) => {
    if (typeof coords[0] === "number") {
      const [lng, lat] = coords;
      bounds.minLat = Math.min(bounds.minLat, lat);
      bounds.maxLat = Math.max(bounds.maxLat, lat);
      bounds.minLng = Math.min(bounds.minLng, lng);
      bounds.maxLng = Math.max(bounds.maxLng, lng);
    } else coords.forEach(visit);
  };
  geometries.forEach((geometry) => geometry?.coordinates && visit(geometry.coordinates));
  return Number.isFinite(bounds.minLat) ? bounds : null;
}

// พื้นที่ของ Polygon / MultiPolygon หน่วยตารางเมตร (สูตรเดียวกับ turf.area)
function ringArea(coords) {
  const radius = 6378137; // รัศมีโลก (เมตร) ค่ามาตรฐาน WGS84 ใช้ในสูตรคำนวณพื้นที่ ไม่ใช่ข้อมูล
  const rad = (degree) => (degree * Math.PI) / 180;
  let total = 0;
  for (let i = 0; i < coords.length; i += 1) {
    const p1 = coords[i];
    const p2 = coords[(i + 1) % coords.length];
    const p3 = coords[(i + 2) % coords.length];
    total += (rad(p3[0]) - rad(p1[0])) * Math.sin(rad(p2[1]));
  }
  return Math.abs((total * radius * radius) / 2);
}

function polygonArea(rings) {
  return rings.reduce((sum, ring, index) => sum + (index === 0 ? ringArea(ring) : -ringArea(ring)), 0);
}

function geometryArea(geometry) {
  if (geometry?.type === "Polygon") return polygonArea(geometry.coordinates);
  if (geometry?.type === "MultiPolygon") return geometry.coordinates.reduce((sum, polygon) => sum + polygonArea(polygon), 0);
  return 0;
}

// โหลดข้อมูลแบบไม่ล้ม: ถ้าไฟล์ CSV หาย จะปิดแค่ /api/district-details (ตอบ 503)
// route อื่นใน server.js ยังทำงานต่อได้ตามปกติ
let DATA = null;
let LOAD_ERROR = null;
try {
  DATA = buildData();
  for (const warning of DATA.warnings) console.warn(`[data-api] ${warning}`);
  console.log(`[data-api] โหลดข้อมูลแล้ว ${DATA.districts.length} เขต, ${DATA.places.length} สถานที่`);
} catch (error) {
  LOAD_ERROR = error;
  console.error(`[data-api] โหลดข้อมูลเขตไม่สำเร็จ: ${error.message}`);
}

// ---------- รูปแบบข้อมูลที่ส่งให้ frontend ----------
function districtSummary(district) {
  const { id, name, shortName, zone, population, area, density, poiCount, placeCounts, mapCenter } = district;
  return { id, name, shortName, zone, population, area, density, poiCount, placeCounts, mapCenter };
}

// รายละเอียดเขตสำหรับหน้าค้นหาจาก navbar (รูปแบบตรงกับ prop `district` ของ DistrictDetails)
function districtProfile(district) {
  const ofType = (type) => DATA.places.filter((place) => place.districtId === district.id && place.type === type);
  const names = (type, limit = 3) => ofType(type).map(({ name }) => name).filter(Boolean).slice(0, limit).join(" · ") || null;
  const latestHouseholds = district.households?.at(-1) ?? null;
  const { placeCounts } = district;

  return {
    isExample: false,
    id: district.id,
    name: district.name,
    shortName: district.shortName,
    zone: district.zone,
    description: district.zone ? `${district.name} อยู่ในกลุ่มเขต${district.zone} ของกรุงเทพมหานคร` : null,
    mapCenter: district.mapCenter,
    totalDistricts: DATA.districts.length,

    area: district.area,
    density: district.density,
    densityRank: district.densityRank ?? null,
    subdistrictCount: null, // ยังไม่มีข้อมูลแขวงใน backend
    households: latestHouseholds,
    householdsYear: latestHouseholds == null ? null : DATA.householdYears.at(-1),

    population: district.population,
    populationRank: district.populationRank ?? null,
    malePopulation: district.male,
    femalePopulation: district.female,

    transport: null, // ไฟล์ผู้โดยสาร BTS ไม่มีคอลัมน์เขต จึงยังแยกสถานีรายเขตไม่ได้
    education: {
      universityCount: placeCounts.university,
      universities: names("university"),
      students: district.students,
      schoolCount: placeCounts.school,
      kindergartenCount: district.kindergartenCount,
    },
    dailyLife: [
      { label: "ห้างสรรพสินค้า / ศูนย์การค้า", count: placeCounts.mall, detail: names("mall") },
      { label: "ศาสนสถานและวัด", count: placeCounts.temple, detail: names("temple") },
      { label: "ศูนย์บริการสาธารณสุข", count: placeCounts.health, detail: names("health") },
    ],
    safety: {
      riskCount: placeCounts.risk,
      year: DATA.years.risk ?? null,
      darkOrIsolatedCount: district.darkPointCount,
    },

    // ตารางเทียบความหนาแน่น: 6 เขตที่หนาแน่นที่สุด (frontend ตัดเขตตัวเองออกเหลือ 5)
    densityRanking: DATA.districts
      .filter(({ density }) => density != null)
      .sort((a, b) => b.density - a.density)
      .slice(0, 6)
      .map(({ id, name, population, area, density }) => ({ id, name, population, area, density })),
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
  const value = String(idOrName);
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

// ตอบเฉพาะ /api/district-details/:id แล้วคืนค่า true
// route อื่นทั้งหมด (Dashboard ของเพื่อน, /predict, /api/districts/profiles) คืนค่า false ให้ server.js จัดการต่อ
function handleDataRequest(req, res) {
  if (req.method !== "GET") return false;

  const url = new URL(req.url, "http://localhost");
  const match = /^\/api\/district-details\/([^/]+)\/?$/.exec(url.pathname);
  if (!match) return false;

  if (!DATA) {
    send(res, 503, { error: `ข้อมูลเขตยังไม่พร้อม: ${LOAD_ERROR?.message ?? "unknown"}` });
    return true;
  }

  try {
    const query = decodeURIComponent(match[1]);
    const district = findDistrict(query);
    if (district) send(res, 200, districtProfile(district));
    else send(res, 404, { error: `ไม่พบเขต "${query}"`, suggestions: searchDistricts(query, 5) });
  } catch (error) {
    console.error("[data-api]", error);
    send(res, error instanceof URIError ? 400 : 500, { error: "ค้นหาเขตไม่สำเร็จ" });
  }
  return true;
}

module.exports = { handleDataRequest, parseCsv, normalizeDistrict };

// รันเดี่ยวสำหรับทดสอบหรือเมื่อโมเดลยังโหลดไม่ได้
if (require.main === module) {
  const port = Number(process.env.DATA_PORT || 3001);
  http.createServer((req, res) => {
    if (!handleDataRequest(req, res)) send(res, 404, { error: "Route not found" });
  }).listen(port, () => console.log(`[data-api] listening on port ${port}`));
}

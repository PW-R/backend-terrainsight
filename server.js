
const http = require("node:http");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { createInterface } = require("node:readline");

// ==========================================
// 1. Configuration
// ==========================================

const port = Number(process.env.PORT || 3000);

const python =
  process.env.PYTHON_BIN ||
  (process.platform === "win32" ? "python" : "python3");

const modelPath = path.join(
  __dirname,
  "terra_insight_model.joblib"
);

const workerPath = path.join(
  __dirname,
  "ml_worker.py"
);

const MAX_BODY_BYTES = 16 * 1024;
const MAX_PENDING = 32;
const PREDICTION_TIMEOUT_MS = 10000;

const REQUIRED_FACTORS = [
  "safety",
  "hospital",
  "school",
  "shopping",
  "park",
  "transport",
];

// ==========================================
// 2. Start Python Worker
// ==========================================

let workerReady = false;
let districtCount = 0;
let shuttingDown = false;
let nextId = 0;

const pending = new Map();

const worker = spawn(
  python,
  ["-u", workerPath, modelPath],
  {
    stdio: ["pipe", "pipe", "inherit"],
  }
);

const output = createInterface({
  input: worker.stdout,
});

// ==========================================
// 3. Response helper
// ==========================================

function sendJson(res, status, body) {
  if (res.destroyed || res.writableEnded) return;

  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
  });

  res.end(JSON.stringify(body));
}

function finishPrediction(id, status, body) {
  const request = pending.get(id);

  if (!request) return;

  pending.delete(id);

  clearTimeout(request.timer);

  sendJson(request.res, status, body);
}

// ==========================================
// 4. Receive Python Worker output
// ==========================================

output.on("line", (line) => {
  let message;

  try {
    message = JSON.parse(line);
  } catch {
    console.error(
      "Unexpected output from Python Worker:",
      line
    );

    shutdown(1);
    return;
  }

  // Worker successfully loaded Model and Dataset
  if (message.ready === true) {
    workerReady = true;
    districtCount = message.district_count || 0;

    console.log("TerraInsight Model loaded.");
    console.log(`Districts prepared: ${districtCount}`);

    return;
  }

  // Prediction failed
  if (message.error) {
    finishPrediction(
      message.id,
      500,
      {
        success: false,
        error: message.error,
      }
    );

    return;
  }

  // Prediction successful
  if (message.result) {
    finishPrediction(
      message.id,
      200,
      {
        success: true,
        recommendations:
          message.result.recommendations || [],
        limitations:
          message.result.limitations || [],
      }
    );
  }
});

// ==========================================
// 5. Validate Frontend data
// ==========================================

function validatePreferences(data) {
  if (
    !data ||
    typeof data !== "object" ||
    Array.isArray(data)
  ) {
    return "Invalid request data";
  }

  if (
    typeof data.workType !== "string" ||
    !data.workType.trim()
  ) {
    return "workType is required";
  }

  if (
    typeof data.workLocation !== "string" ||
    !data.workLocation.trim()
  ) {
    return "workLocation is required";
  }

  if (
    !Array.isArray(data.travel) ||
    data.travel.length === 0 ||
    !data.travel.every(
      (item) =>
        typeof item === "string" && item.trim()
    )
  ) {
    return "Please select at least one travel method";
  }

  if (
    typeof data.distance !== "number" ||
    !Number.isFinite(data.distance) ||
    data.distance < 1 ||
    data.distance > 20
  ) {
    return "distance must be between 1 and 20";
  }

  if (
    !data.factors ||
    typeof data.factors !== "object" ||
    Array.isArray(data.factors)
  ) {
    return "factors are required";
  }

  for (const factor of REQUIRED_FACTORS) {
    const value = data.factors[factor];

    if (
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      value < 1 ||
      value > 5
    ) {
      return `Invalid factor: ${factor}`;
    }
  }

  return null;
}

// ==========================================
// 6. Handle prediction request
// ==========================================

async function predict(req, res) {
  const contentType =
    req.headers["content-type"]
      ?.split(";")[0]
      .trim();

  if (contentType !== "application/json") {
    sendJson(res, 415, {
      success: false,
      error: "Send Content-Type: application/json",
    });

    req.resume();
    return;
  }

  const chunks = [];
  let bytes = 0;

  for await (const chunk of req) {
    bytes += chunk.length;

    if (bytes > MAX_BODY_BYTES) {
      res.setHeader("Connection", "close");

      sendJson(res, 413, {
        success: false,
        error: "Request body is too large",
      });

      req.resume();
      return;
    }

    chunks.push(chunk);
  }

  let preferences;

  try {
    preferences = JSON.parse(
      Buffer.concat(chunks).toString("utf8")
    );
  } catch {
    sendJson(res, 400, {
      success: false,
      error: "Invalid JSON",
    });

    return;
  }

  const validationError =
    validatePreferences(preferences);

  if (validationError) {
    sendJson(res, 422, {
      success: false,
      error: validationError,
    });

    return;
  }

  if (!workerReady || shuttingDown) {
    sendJson(res, 503, {
      success: false,
      error: "Model is not ready",
    });

    return;
  }

  if (pending.size >= MAX_PENDING) {
    sendJson(res, 503, {
      success: false,
      error: "Prediction queue is full. Try again shortly.",
    });

    return;
  }

  const id = ++nextId;

  const timer = setTimeout(() => {
    finishPrediction(
      id,
      504,
      {
        success: false,
        error: "Prediction timed out",
      }
    );
  }, PREDICTION_TIMEOUT_MS);

  pending.set(id, {
    res,
    timer,
  });

  const payload = {
    id,
    preferences,
  };

  worker.stdin.write(
    `${JSON.stringify(payload)}\n`,
    (error) => {
      if (error) {
        finishPrediction(
          id,
          503,
          {
            success: false,
            error: "Model is unavailable",
          }
        );
      }
    }
  );
}

// ==========================================
// 7. Create HTTP Server
// ==========================================

const server = http.createServer((req, res) => {
  res.setHeader(
    "Access-Control-Allow-Origin",
    "*"
  );

  res.setHeader(
    "Access-Control-Allow-Methods",
    "GET, POST, OPTIONS"
  );

  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type"
  );

  const route = req.url.split("?")[0];

  // CORS preflight
  if (req.method === "OPTIONS") {
    res.writeHead(204).end();
    return;
  }

  // Health check
  if (req.method === "GET" && route === "/health") {
    const ready = workerReady && !shuttingDown;

    sendJson(
      res,
      ready ? 200 : 503,
      {
        success: ready,
        ready,
        status: ready ? "ready" : "unavailable",
        model_loaded: ready,
        district_count: districtCount,
      }
    );

    return;
  }

  // Prediction
  if (req.method === "POST" && route === "/predict") {
    predict(req, res).catch((error) => {
      console.error(
        `Prediction request failed: ${error.message}`
      );

      sendJson(res, 500, {
        success: false,
        error: "Could not process prediction request",
      });
    });

    return;
  }

  sendJson(res, 404, {
    success: false,
    error: "Route not found",
  });
});

// ==========================================
// 8. Shutdown and Error Handling
// ==========================================

function shutdown(exitCode) {
  if (shuttingDown) return;

  shuttingDown = true;
  workerReady = false;

  process.exitCode = exitCode;

  for (const id of pending.keys()) {
    finishPrediction(
      id,
      503,
      {
        success: false,
        error: "Model is unavailable",
      }
    );
  }

  if (server.listening) {
    server.close();
  }

  worker.stdin.end();

  setTimeout(() => {
    worker.kill();
  }, 5000).unref();
}

worker.stdin.on("error", (error) => {
  if (!shuttingDown) {
    console.error(
      `Python Worker connection failed: ${error.message}`
    );

    shutdown(1);
  }
});

worker.on("error", (error) => {
  console.error(
    `Could not start Python: ${error.message}`
  );

  shutdown(1);
});

worker.on("exit", (code, signal) => {
  if (!shuttingDown) {
    console.error(
      `Python Worker stopped: ${signal ?? code}`
    );

    shutdown(1);
  }
});

server.on("error", (error) => {
  console.error(
    `Server failed: ${error.message}`
  );

  shutdown(1);
});

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

// ==========================================
// 9. Start Server
// ==========================================

server.listen(port, "0.0.0.0", () => {
  console.log(
    `TerraInsight server listening on port ${port}.`
  );
});

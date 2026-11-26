const http = require("node:http");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { createInterface } = require("node:readline");

const port = Number(process.env.PORT || 3000);
const python = process.env.PYTHON_BIN ||
  (process.platform === "win32" ? "python" : "python3");
const modelPath = path.join(__dirname, "terra_insight_model.joblib");

const MAX_BODY_BYTES = 4096;
const MAX_PENDING = 32;
const PREDICTION_TIMEOUT_MS = 10000;

const loader = `
import json
import sys
import traceback
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("error", InconsistentVersionWarning)
bundle = joblib.load(sys.argv[1])

model = bundle["model"]
columns = bundle["feature_columns"]
group_names = bundle["group_names"]

if not callable(getattr(model, "predict", None)):
    raise ValueError("The saved bundle has no prediction model")

print(json.dumps({"ready": True, "columns": columns}), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    response = {"id": request["id"]}

    try:
        row = pd.DataFrame([request["features"]], columns=columns)
        group_id = int(model.predict(row)[0])
        response["result"] = {
            "group_id": group_id,
            "group_name": group_names[group_id],
        }
    except Exception:
        traceback.print_exc(file=sys.stderr)
        response["error"] = "Prediction failed"

    print(json.dumps(response), flush=True)
`;

let featureColumns = null;
let shuttingDown = false;
let nextId = 0;
const pending = new Map();

const worker = spawn(python, ["-u", "-c", loader, modelPath], {
  stdio: ["pipe", "pipe", "inherit"],
});

const output = createInterface({ input: worker.stdout });

function sendJson(res, status, body) {
  if (res.destroyed || res.writableEnded) return;
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

function finishPrediction(id, status, body) {
  const request = pending.get(id);
  if (!request) return;

  pending.delete(id);
  clearTimeout(request.timer);
  sendJson(request.res, status, body);
}

output.on("line", (line) => {
  let message;

  try {
    message = JSON.parse(line);
  } catch {
    console.error("Unexpected output from the Python worker.");
    shutdown(1);
    return;
  }

  if (message.ready) {
    featureColumns = message.columns;
    console.log("Terra Insight model loaded.");
    return;
  }

  if (message.error) {
    finishPrediction(message.id, 500, { error: message.error });
    return;
  }

  finishPrediction(message.id, 200, message.result);
});

async function predict(req, res) {
  const contentType = req.headers["content-type"]?.split(";")[0].trim();
  if (contentType !== "application/json") {
    sendJson(res, 415, { error: "Send Content-Type: application/json" });
    req.resume();
    return;
  }

  // Keep only a small JSON body in memory.
  const chunks = [];
  let bytes = 0;

  for await (const chunk of req) {
    bytes += chunk.length;
    if (bytes > MAX_BODY_BYTES) {
      res.setHeader("Connection", "close");
      sendJson(res, 413, { error: "Request body is too large" });
      // Preserve the connection long enough to send the error response.
      req.resume();
      return;
    }
    chunks.push(chunk);
  }

  let features;
  try {
    features = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    sendJson(res, 400, { error: "Invalid JSON" });
    return;
  }

  if (!featureColumns || shuttingDown) {
    sendJson(res, 503, { error: "Model is unavailable" });
    return;
  }

  if (
    !features ||
    typeof features !== "object" ||
    Array.isArray(features) ||
    Object.keys(features).length !== featureColumns.length ||
    featureColumns.some((column) =>
      !Object.hasOwn(features, column) ||
      !Number.isSafeInteger(features[column]) ||
      features[column] < 0
    )
  ) {
    sendJson(res, 422, {
      error: "Send every model feature as a nonnegative whole number",
      required_features: featureColumns,
    });
    return;
  }

  if (pending.size >= MAX_PENDING) {
    sendJson(res, 503, { error: "Prediction queue is full. Try again shortly." });
    return;
  }

  const id = ++nextId;
  const timer = setTimeout(() => {
    finishPrediction(id, 504, { error: "Prediction timed out" });
  }, PREDICTION_TIMEOUT_MS);

  pending.set(id, { res, timer });

  worker.stdin.write(`${JSON.stringify({ id, features })}\n`, (error) => {
    if (error) {
      finishPrediction(id, 503, { error: "Model is unavailable" });
    }
  });
}

const server = http.createServer((req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  const route = req.url.split("?")[0];

  if (req.method === "OPTIONS" && route === "/predict") {
    res.writeHead(204).end();
    return;
  }

  if (req.method === "GET" && route === "/health") {
    const ready = Boolean(featureColumns) && !shuttingDown;
    sendJson(res, ready ? 200 : 503, {
      status: ready ? "ready" : "unavailable",
      model_loaded: ready,
    });
    return;
  }

  if (req.method === "POST" && route === "/predict") {
    predict(req, res).catch((error) => {
      console.error(`Prediction request failed: ${error.message}`);
      sendJson(res, 500, { error: "Could not process prediction request" });
    });
    return;
  }

  sendJson(res, 404, { error: "Route not found" });
});

function shutdown(exitCode) {
  if (shuttingDown) return;

  shuttingDown = true;
  featureColumns = null;
  process.exitCode = exitCode;

  for (const id of pending.keys()) {
    finishPrediction(id, 503, { error: "Model is unavailable" });
  }

  server.close();
  worker.stdin.end();
  setTimeout(() => worker.kill(), 5000).unref();
}

worker.stdin.on("error", (error) => {
  if (!shuttingDown) {
    console.error(`Python worker connection failed: ${error.message}`);
    shutdown(1);
  }
});

worker.on("error", (error) => {
  console.error(`Could not start Python: ${error.message}`);
  shutdown(1);
});

worker.on("exit", (code, signal) => {
  if (!shuttingDown) {
    console.error(`Python worker stopped: ${signal ?? code}`);
    shutdown(1);
  }
});

server.on("error", (error) => {
  console.error(`Server failed: ${error.message}`);
  shutdown(1);
});

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

server.listen(port, "0.0.0.0", () => {
  console.log(`Terra Insight server listening on port ${port}.`);
});
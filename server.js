const http = require("node:http");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { createInterface } = require("node:readline");

const port = Number(process.env.PORT || 3000);
const python = process.env.PYTHON_BIN ||
  (process.platform === "win32" ? "python" : "python3");
const modelPath = path.join(__dirname, "terra_insight_model.joblib");

const loader = `
import sys
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("error", InconsistentVersionWarning)
bundle = joblib.load(sys.argv[1])

if not callable(getattr(bundle["model"], "predict", None)):
    raise ValueError("The saved bundle has no prediction model")

print("MODEL_READY", flush=True)

# Keep the model in memory until Node closes the pipe.
for _ in sys.stdin:
    pass
`;

let modelLoaded = false;
let shuttingDown = false;

const worker = spawn(python, ["-u", "-c", loader, modelPath], {
  stdio: ["pipe", "pipe", "inherit"],
});

const output = createInterface({ input: worker.stdout });

output.on("line", (line) => {
  if (line === "MODEL_READY") {
    modelLoaded = true;
    console.log("Terra Insight model loaded.");
  }
});

const server = http.createServer((req, res) => {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Access-Control-Allow-Origin", "*");

  if (req.method === "GET" && req.url === "/health") {
    res.writeHead(modelLoaded ? 200 : 503);
    res.end(JSON.stringify({
      status: modelLoaded ? "ready" : "loading",
      model_loaded: modelLoaded,
    }));
    return;
  }

  res.writeHead(404);
  res.end(JSON.stringify({ error: "Route not found" }));
});

function shutdown(exitCode) {
  if (shuttingDown) return;

  shuttingDown = true;
  process.exitCode = exitCode;
  server.close();
  worker.stdin.end();

  // Give the worker time to exit, even if it is still loading the model.
  setTimeout(() => worker.kill(), 5000).unref();
}

worker.on("error", (error) => {
  console.error(`Could not start Python: ${error.message}`);
  shutdown(1);
});

worker.on("exit", (code, signal) => {
  modelLoaded = false;

  if (!shuttingDown) {
    console.error(`Python model loader stopped: ${signal ?? code}`);
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
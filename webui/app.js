const form = document.querySelector("#runForm");
const resetBtn = document.querySelector("#resetBtn");
const deviceSelect = document.querySelector("#deviceSelect");
const recommendation = document.querySelector("#recommendation");
const runState = document.querySelector("#runState");
const runDot = document.querySelector("#runDot");
const progressBar = document.querySelector("#progressBar");
const metricsBody = document.querySelector("#metricsTable tbody");
const leaderboard = document.querySelector("#leaderboard");
const charts = document.querySelector("#charts");
const logBox = document.querySelector("#logBox");

let cudaAvailable = false;
let totalTimesteps = Number(field("timesteps").value);

function field(name) {
  return form.elements.namedItem(name);
}

function readPayload() {
  const data = new FormData(form);
  return {
    device: data.get("device"),
    n_envs: Number(data.get("n_envs")),
    n_steps: Number(data.get("n_steps")),
    timesteps: Number(data.get("timesteps")),
    seeds: Number(data.get("seeds")),
    n_agents: Number(data.get("n_agents")),
    n_blocks: Number(data.get("n_blocks")),
    algorithm: data.get("algorithm"),
    compare_algorithm: data.get("compare_algorithm"),
    mode: data.get("mode"),
    batch_size: Number(data.get("batch_size")),
    safety: field("safety").checked,
    bd_mode: field("bd_mode").checked,
    death_masking: field("death_masking").checked,
  };
}

function updateRecommendation() {
  const payload = readPayload();
  const product = payload.n_envs * payload.n_steps;
  totalTimesteps = payload.timesteps;
  let text = `Rollout product: ${payload.n_steps} x ${payload.n_envs} = ${product.toLocaleString()}. `;
  if (product >= 20480 && cudaAvailable) {
    text += "GPU recommended for this rollout size.";
  } else if (product >= 20480) {
    text += "GPU would be recommended, but CUDA is not detected.";
  } else {
    text += "CPU is fine for this quick or small run.";
  }
  recommendation.textContent = text;
}

function setDeviceOptions(device) {
  cudaAvailable = Boolean(device.cuda_available);
  const current = device.current === "cuda" ? `CUDA${device.gpu_name ? ` (${device.gpu_name})` : ""}` : "CPU";
  document.querySelector("#deviceText").textContent = current;
  document.querySelector("#deviceHint").textContent = device.torch_version ? `PyTorch ${device.torch_version}` : "PyTorch status unavailable";

  const selected = deviceSelect.value || "auto";
  deviceSelect.innerHTML = "";
  [
    ["auto", "Auto"],
    ["cpu", "CPU"],
    ...(cudaAvailable ? [["cuda", "CUDA / GPU"]] : []),
  ].forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    deviceSelect.appendChild(option);
  });
  deviceSelect.value = [...deviceSelect.options].some((option) => option.value === selected) ? selected : "auto";
  updateRecommendation();
}

function renderMetrics(metrics = {}) {
  metricsBody.innerHTML = "";
  const entries = Object.entries(metrics);
  if (!entries.length) {
    metricsBody.innerHTML = `<tr><td colspan="2">No live metrics yet.</td></tr>`;
    return;
  }
  entries.forEach(([key, value]) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${key}</td><td>${value}</td>`;
    metricsBody.appendChild(row);
  });

  document.querySelector("#fps").textContent = metrics["time / fps"] || metrics.fps || "-";
  document.querySelector("#iterations").textContent = metrics["time / iterations"] || metrics.iterations || "-";
  document.querySelector("#elapsed").textContent = metrics["time / time_elapsed"] || metrics.time_elapsed || "-";
}

function renderLeaderboard(rows = []) {
  const thead = leaderboard.querySelector("thead");
  const tbody = leaderboard.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";
  if (!rows.length) {
    tbody.innerHTML = `<tr><td>No leaderboard rows yet.</td></tr>`;
    return;
  }
  const columns = Object.keys(rows[0]);
  thead.innerHTML = `<tr>${columns.map((column) => `<th>${column}</th>`).join("")}</tr>`;
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = columns.map((column) => `<td>${row[column] ?? ""}</td>`).join("");
    tbody.appendChild(tr);
  });
}

function renderCharts(items = []) {
  charts.innerHTML = "";
  if (!items.length) {
    charts.innerHTML = `<div class="empty">No generated charts yet. Run or compare experiments to fill this area.</div>`;
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `<strong>${item.name}</strong><img src="${item.url}&v=${item.modified}" alt="${item.name}">`;
    charts.appendChild(card);
  });
}

function renderState(state) {
  setDeviceOptions(state.device || {});
  const running = Boolean(state.running);
  runState.textContent = running ? "Running" : state.returncode === 0 ? "Done" : state.error ? "Error" : "Idle";
  runDot.classList.toggle("on", running);
  form.querySelector(".primary").disabled = running;
  resetBtn.disabled = running;
  document.querySelector("#seedText").textContent = state.current_seed ? `Seed ${state.current_seed}` : "Seed -";

  const parsedProgress = Number(state.progress || 0);
  const percent = parsedProgress > 100 ? Math.min(100, Math.round((parsedProgress / Math.max(totalTimesteps, parsedProgress)) * 100)) : parsedProgress;
  progressBar.style.width = `${percent}%`;

  renderMetrics(state.metrics);
  renderLeaderboard(state.leaderboard);
  renderCharts(state.charts);
  logBox.textContent = (state.logs || []).join("\n");
  logBox.scrollTop = logBox.scrollHeight;
}

form.addEventListener("input", updateRecommendation);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const response = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(readPayload()),
  });
  const data = await response.json();
  if (!data.ok) alert(data.message);
});

resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
});

async function boot() {
  try {
    const response = await fetch("/api/status");
    renderState(await response.json());
    const events = new EventSource("/events");
    events.onmessage = (event) => renderState(JSON.parse(event.data));
    events.onerror = () => {
      runState.textContent = "Reconnecting";
    };
  } catch (error) {
    runState.textContent = "UI Error";
    logBox.textContent = `Interface initialization failed: ${error.message}`;
  }
}

boot();

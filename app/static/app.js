// The browser only orchestrates jobs; all extraction happens in shared Python code.
const $ = (id) => document.getElementById(id);
let formats = {},
  inputFiles = [],
  currentJob = null,
  timer = null;
const labels = {
  audio: "Audio",
  text: "Text",
  pdf: "PDF",
  excel: "Excel",
  image: "Images",
  video: "Video",
  docs: "Word",
};
async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let body;
    try {
      body = await response.json();
    } catch {
      body = {};
    }
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : `Request failed (${response.status})`,
    );
  }
  return response;
}
function notice(message) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
function element(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className || "";
  if (text !== undefined) node.textContent = text;
  return node;
}
function size(bytes) {
  return bytes < 1024
    ? `${bytes} B`
    : bytes < 1048576
      ? `${(bytes / 1024).toFixed(1)} KB`
      : `${(bytes / 1048576).toFixed(1)} MB`;
}
function renderFiles(id, files, outputs = false) {
  const container = $(id);
  container.replaceChildren();
  if (!files.length) {
    container.append(
      element(
        "p",
        "empty",
        outputs
          ? "Your text files will appear here."
          : "No sources in these folders yet.",
      ),
    );
    return;
  }
  for (const file of files) {
    const row = element("div", "file-row");
    row.append(
      element("span", "file-name", file.path),
      element("span", "size", size(file.bytes)),
    );
    if (outputs) {
      const preview = element("button", "", "Preview");
      preview.onclick = () => showPreview(file.path);
      row.append(preview);
      const download = element("a", "", "↓");
      download.href = `/api/download?path=${encodeURIComponent(file.path)}`;
      download.title = `Download ${file.path}`;
      row.append(download);
    }
    container.append(row);
  }
}
async function refresh() {
  const selected = [
    ...document.querySelectorAll(".category input:checked"),
  ].map((n) => n.value);
  const initial = !$("categories").children.length;
  const [inputs, outputs] = await Promise.all(
    ["/api/files/input", "/api/files/output"].map(async (url) =>
      (await api(url)).json(),
    ),
  );
  inputFiles = inputs;
  $("input-count").textContent = `${inputs.length} files`;
  $("output-count").textContent = `${outputs.length} files`;
  renderFiles("inputs", inputs);
  renderFiles("outputs", outputs, true);
  $("categories").replaceChildren();
  for (const category of Object.keys(formats)) {
    const count = inputs.filter((f) => f.category === category).length;
    const label = element("label", "category");
    const checkbox = element("input");
    checkbox.type = "checkbox";
    checkbox.value = category;
    checkbox.checked = initial ? count > 0 : selected.includes(category);
    label.append(
      checkbox,
      document.createTextNode(labels[category]),
      element("span", "count", String(count)),
    );
    $("categories").append(label);
  }
}
async function upload(event) {
  const picker = event.target,
    category = $("upload-category").value;
  const files = [...picker.files].filter((file) =>
    formats[category].includes("." + file.name.split(".").pop().toLowerCase()),
  );
  const skipped = picker.files.length - files.length;
  if (!files.length) {
    notice(`No supported ${labels[category]} files selected.`);
    picker.value = "";
    return;
  }
  const paths = files.map((file) => file.webkitRelativePath || file.name);
  const data = new FormData();
  files.forEach((file) => data.append("files", file));
  data.append("paths", JSON.stringify(paths));
  try {
    notice(`Uploading ${files.length} files…`);
    await api(`/api/upload/${category}`, { method: "POST", body: data });
    await refresh();
    notice(
      `Uploaded ${files.length} files.${skipped ? ` Skipped ${skipped} unsupported files.` : ""}`,
    );
  } catch (error) {
    notice(error.message);
  } finally {
    picker.value = "";
  }
}
async function showPreview(path) {
  try {
    const text = await (
      await api(`/api/preview?path=${encodeURIComponent(path)}`)
    ).text();
    $("preview-name").textContent = path;
    $("preview-text").textContent = text;
    $("preview-panel").hidden = false;
    $("preview-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    notice(error.message);
  }
}
async function poll() {
  if (!currentJob) return;
  try {
    const job = await (await api(`/api/jobs/${currentJob}`)).json();
    $("job-status").textContent =
      {
        queued: "Queued",
        running: "Processing your sources",
        completed: "Completed",
        partial: "Finished · review notes",
        failed: "Processing failed",
        interrupted: "Interrupted · start again",
      }[job.status] || job.status;
    $("job-count").textContent = `${job.completed} / ${job.files.length}`;
    $("progress").max = job.files.length || 1;
    $("progress").value = job.completed;
    $("stage").textContent = job.stage;
    $("current-file").textContent = job.current_file || "";
    const busy = ["running", "queued"].includes(job.status);
    $("start").disabled = busy;
    const report = $("report-content");
    report.replaceChildren();
    for (const file of job.files) {
      const line = element("p", "", `${file.path}: ${file.status}`);
      report.append(line);
      if (file.error) report.append(element("p", "", file.error));
      for (const warning of file.warnings || [])
        report.append(element("p", "", warning));
      if (file.devices?.length)
        report.append(
          element("pre", "", JSON.stringify(file.devices, null, 2)),
        );
    }
    $("zip").hidden = busy;
    $("zip").href = `/api/jobs/${job.id}/download`;
    if (busy) timer = setTimeout(poll, 1500);
    else {
      await refresh();
      notice(
        job.status === "completed"
          ? ""
          : "Processing finished with notes. Open processing details for any partial results or failed files.",
      );
    }
  } catch (error) {
    notice(error.message);
    $("start").disabled = false;
  }
}
$("start").onclick = async () => {
  const categories = [
    ...document.querySelectorAll(".category input:checked"),
  ].map((n) => n.value);
  if (!categories.length) {
    notice("Choose at least one input category.");
    return;
  }
  try {
    $("start").disabled = true;
    notice("");
    const job = await (
      await api("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ categories }),
      })
    ).json();
    currentJob = job.id;
    clearTimeout(timer);
    await poll();
  } catch (error) {
    notice(error.message);
    $("start").disabled = false;
  }
};
$("check").onclick = async () => {
  $("readiness").hidden = false;
  $("readiness").textContent = "Checking local tools and models…";
  try {
    const data = await (await api("/api/ready")).json();
    $("readiness").textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    $("readiness").textContent = error.message;
  }
};
$("refresh").onclick = () => refresh().catch((error) => notice(error.message));
$("folder-picker").onchange = upload;
$("file-picker").onchange = upload;
$("close-preview").onclick = () => ($("preview-panel").hidden = true);
(async () => {
  try {
    const health = await (await api("/api/health")).json();
    formats = health.formats;
    for (const category of Object.keys(formats)) {
      const option = element("option", "", labels[category]);
      option.value = category;
      $("upload-category").append(option);
    }
    const updateAccept = () =>
      ($("file-picker").accept = formats[$("upload-category").value].join(","));
    $("upload-category").onchange = updateAccept;
    updateAccept();
    await refresh();
    const jobs = await (await api("/api/jobs")).json();
    currentJob = health.active_job || jobs[0]?.id;
    if (currentJob) await poll();
  } catch (error) {
    notice(error.message);
  }
})();

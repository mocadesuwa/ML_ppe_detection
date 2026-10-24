"use strict";
const $ = (id) => document.getElementById(id);
let file = null, previewURL = null, resultURL = null, controller = null, requestId = 0, busy = false, connected = false;
const accepted = new Set(["image/jpeg", "image/png", "image/bmp", "image/x-ms-bmp", "image/webp"]);
function updateButton() { $("predict").disabled = !file || !connected || busy; $("predict-text").textContent = busy ? "正在识别，请稍候…" : "开始识别"; $("predict-arrow").textContent = busy ? "…" : "→"; }
function clearResult() {
  $("error").hidden = true; $("result-symbol").className = "result-symbol"; $("result-symbol").textContent = "—";
  $("result-title").textContent = file ? "图片已就绪" : "等待一张头像";
  $("result-description").textContent = file ? "点击“开始识别”查看模型判断。" : "上传后点击“开始识别”，结果会显示在这里。";
  for (const prefix of ["wearing", "not-wearing"]) { $(prefix + "-score").textContent = "—"; $(prefix + "-bar").style.width = "0%"; $(prefix + "-bar").setAttribute("aria-valuenow", "0"); }
  $("result-meta").hidden = true; $("known-note").hidden = true; $("download").hidden = true;
  if (resultURL) { URL.revokeObjectURL(resultURL); resultURL = null; }
}
function showError(message) { $("error").textContent = message; $("error").hidden = false; }
function setFile(nextFile) {
  requestId++; if (controller) controller.abort(); busy = false; file = null;
  if (previewURL) URL.revokeObjectURL(previewURL); previewURL = null;
  $("image-preview").hidden = true; $("empty-preview").hidden = false; $("preview").removeAttribute("src"); clearResult();
  if (!nextFile) { updateButton(); return; }
  if ((!accepted.has(nextFile.type) && !/\.(jpe?g|png|bmp|webp)$/i.test(nextFile.name)) || nextFile.size === 0) { showError("请选择有效的静态JPG、PNG、BMP或WebP头像。"); updateButton(); return; }
  if (nextFile.size > 8 * 1024 * 1024) { showError("图片超过8MB，请先缩小图片。"); updateButton(); return; }
  file = nextFile; previewURL = URL.createObjectURL(file); $("preview").src = previewURL; $("filename").textContent = file.name;
  $("image-preview").hidden = false; $("empty-preview").hidden = true; clearResult(); updateButton();
}
$("choose").addEventListener("click", () => $("file").click()); $("replace").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (event) => { setFile(event.target.files[0]); event.target.value = ""; });
$("preview").addEventListener("error", () => { if (file) { setFile(null); showError("图片无法预览，请检查文件是否完整。"); } });
for (const type of ["dragenter", "dragover"]) $("dropzone").addEventListener(type, (event) => { event.preventDefault(); $("dropzone").classList.add("is-dragging"); });
$("dropzone").addEventListener("dragleave", (event) => { if (!$("dropzone").contains(event.relatedTarget)) $("dropzone").classList.remove("is-dragging"); });
$("dropzone").addEventListener("drop", (event) => { event.preventDefault(); $("dropzone").classList.remove("is-dragging"); if (event.dataTransfer.files.length !== 1) { setFile(null); showError("每次请只放入一张单人头像。"); return; } setFile(event.dataTransfer.files[0]); });
window.addEventListener("dragover", (event) => event.preventDefault()); window.addEventListener("drop", (event) => event.preventDefault());
$("model").addEventListener("change", () => { requestId++; if (controller) controller.abort(); busy = false; clearResult(); updateModelNote(); updateButton(); });
function updateModelNote() { $("model-note").textContent = $("model").selectedOptions[0]?.dataset.note || ""; }
$("predict").addEventListener("click", async () => {
  if (!file || busy || !connected) return;
  const current = ++requestId; controller = new AbortController(); busy = true; clearResult(); updateButton(); $("result-title").textContent = "模型正在查看头像"; $("result-description").textContent = "首次识别需要加载模型，可能稍慢。";
  try {
    const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": file.type || "application/octet-stream", "X-Model-Id": $("model").value }, body: file, signal: controller.signal });
    const result = await response.json(); if (current !== requestId) return; if (!response.ok) throw new Error(result.error || "识别失败，请稍后再试。");
    if (!["wearing_goggles", "not_wearing_goggles"].includes(result.label)) throw new Error("模型返回的类别不符合当前任务。");
    const wearing = result.label === "wearing_goggles"; $("result-symbol").className = "result-symbol " + (wearing ? "positive" : "negative"); $("result-symbol").textContent = wearing ? "✓" : "!";
    $("result-title").textContent = wearing ? "已佩戴护目镜" : "未佩戴护目镜"; $("result-description").textContent = "以上是这张头像的模型判断。";
    for (const [prefix, key] of [["wearing", "wearing_goggles"], ["not-wearing", "not_wearing_goggles"]]) { const percent = result.probabilities[key] * 100; $(prefix + "-score").textContent = percent.toFixed(1) + "%"; $(prefix + "-bar").style.width = percent + "%"; $(prefix + "-bar").setAttribute("aria-valuenow", percent.toFixed(1)); }
    $("dimensions").textContent = result.image_size.width + " × " + result.image_size.height + " px"; $("timing").textContent = "本次耗时 " + (result.elapsed_ms / 1000).toFixed(2) + " 秒"; $("result-meta").hidden = false;
    const names = { train: "训练集", val: "验证集", test: "测试集" }; $("known-note").textContent = result.known_splits.length ? "这张图与当前模型已登记头像完全重复：" + result.known_splits.map((name) => names[name] || name).join("、") + "。它不属于独立新图。" : "未匹配到已登记头像的相同字节或像素；这不保证图像来源独立。"; $("known-note").hidden = false;
    resultURL = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" })); $("download").href = resultURL; $("download").hidden = false;
  } catch (error) { if (current === requestId && error.name !== "AbortError") { clearResult(); showError(error.message || "无法连接本机服务。"); $("result-title").textContent = "本次未完成识别"; } }
  finally { if (current === requestId) { busy = false; updateButton(); } }
});
async function connect() {
  try {
    const response = await fetch("/api/models"); if (!response.ok) throw new Error("服务未就绪"); const metadata = await response.json();
    $("model").replaceChildren(); for (const model of metadata.models) { const option = document.createElement("option"); option.value = model.id; option.textContent = model.name + " · " + model.run_name; option.dataset.note = model.note; $("model").append(option); }
    connected = metadata.models.length > 0; $("model").disabled = !connected; updateModelNote(); $("connection").textContent = "本机模型已连接"; $("connection").className = "connection connected";
  } catch { $("connection").textContent = "本机服务未连接"; $("connection").className = "connection disconnected"; $("model-note").textContent = "请先运行项目中的start_demo.cmd，并在本机服务地址打开页面。"; }
  updateButton();
}
connect();

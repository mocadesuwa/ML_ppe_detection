"use strict";

(() => {
  const element = (id) => document.getElementById(id);
  const acceptedTypes = new Set([
    "image/jpeg", "image/png", "image/bmp", "image/x-ms-bmp", "image/webp",
  ]);
  let file = null;
  let previewURL = null;
  let resultURL = null;
  let controller = null;
  let requestId = 0;
  let selectionId = 0;
  let busy = false;
  let reading = false;
  let connecting = false;
  let metadataReady = false;
  let connectionError = false;
  let maxUploadBytes = 8 * 1024 * 1024;

  function updateControls() {
    element("predict").disabled = !file || !metadataReady || busy || reading || connecting;
    element("predict-text").textContent = busy
      ? "正在识别，请稍候…"
      : reading ? "正在读取图片…" : "开始识别";
    element("predict-arrow").textContent = busy || reading ? "…" : "→";
    element("model").disabled = !metadataReady || busy || connecting;
    element("reconnect").disabled = busy || connecting;
    element("reconnect").textContent = connecting ? "正在连接…" : "重新连接";
    element("result-zone").setAttribute("aria-busy", String(busy || reading));
  }

  function setConnection(online, message) {
    element("connection").textContent = message;
    element("connection").className = "connection " + (online ? "connected" : "disconnected");
    element("reconnect").hidden = online;
  }

  function clearResult() {
    element("error").hidden = true;
    element("result-symbol").className = "result-symbol";
    element("result-symbol").textContent = "—";
    element("result-state").textContent = file ? "图片就绪" : "等待图片";
    element("result-title").textContent = file ? "图片已就绪" : "等待一张头像";
    element("result-description").textContent = file
      ? "点击“开始识别”查看模型判断。"
      : "上传图片后点击“开始识别”，判断会显示在这里。";
    for (const prefix of ["wearing", "not-wearing"]) {
      element(prefix + "-score").textContent = "—";
      element(prefix + "-bar").style.width = "0%";
      element(prefix + "-bar").classList.remove("is-selected");
      element(prefix + "-bar").setAttribute("aria-valuenow", "0");
    }
    element("result-meta").hidden = true;
    element("known-note").hidden = true;
    element("download").hidden = true;
    element("download").removeAttribute("href");
    if (resultURL) {
      URL.revokeObjectURL(resultURL);
      resultURL = null;
    }
  }

  function showError(message) {
    element("error").textContent = message;
    element("error").hidden = false;
  }

  function cancelRequest() {
    requestId += 1;
    if (controller) {
      controller.abort();
      controller = null;
    }
    busy = false;
  }

  async function setFile(nextFile) {
    cancelRequest();
    const current = ++selectionId;
    file = null;
    reading = false;
    connectionError = false;
    element("preview").removeAttribute("src");
    if (previewURL) {
      URL.revokeObjectURL(previewURL);
      previewURL = null;
    }
    element("image-preview").hidden = true;
    element("empty-preview").hidden = false;
    clearResult();
    if (!nextFile) {
      updateControls();
      return;
    }
    const supported = acceptedTypes.has(nextFile.type)
      || /\.(jpe?g|png|bmp|webp)$/i.test(nextFile.name);
    if (!supported || nextFile.size === 0) {
      showError("请选择完整的静态 JPG、PNG、BMP 或 WebP 头像。");
      updateControls();
      return;
    }
    if (nextFile.size > maxUploadBytes) {
      showError("图片超过上传上限，请先将图片缩小至 8 MiB 以内。");
      updateControls();
      return;
    }
    reading = true;
    element("result-state").textContent = "读取中";
    element("result-title").textContent = "正在读取头像";
    updateControls();
    previewURL = URL.createObjectURL(nextFile);
    element("preview").src = previewURL;
    try {
      await element("preview").decode();
      if (current !== selectionId) {
        return;
      }
      const preview = element("preview");
      file = nextFile;
      preview.width = preview.naturalWidth;
      preview.height = preview.naturalHeight;
      element("filename").textContent = file.name;
      element("file-details").textContent = preview.naturalWidth + " × "
        + preview.naturalHeight + " px · " + (file.size / 1024).toFixed(1) + " KiB";
      element("image-preview").hidden = false;
      element("empty-preview").hidden = true;
      clearResult();
    } catch {
      if (current === selectionId) {
        element("preview").removeAttribute("src");
        URL.revokeObjectURL(previewURL);
        previewURL = null;
        clearResult();
        showError("图片无法预览，请检查文件是否完整，或重新导出为静态图片。");
      }
    } finally {
      if (current === selectionId) {
        reading = false;
        updateControls();
      }
    }
  }

  function updateModelNote() {
    element("model-note").textContent = element("model").selectedOptions[0]?.dataset.note || "";
  }

  function renderResult(result) {
    if (!["wearing_goggles", "not_wearing_goggles"].includes(result.label)) {
      throw new Error("模型返回的类别不符合当前任务，请在启动器中重启服务。");
    }
    const wearing = result.label === "wearing_goggles";
    element("result-symbol").className = "result-symbol " + (wearing ? "positive" : "negative");
    element("result-symbol").textContent = wearing ? "✓" : "!";
    element("result-state").textContent = "识别完成";
    element("result-title").textContent = wearing ? "已佩戴护目镜" : "未佩戴护目镜";
    element("result-description").textContent = "以上是当前模型对这张头像的判断。";
    for (const [prefix, key] of [
      ["wearing", "wearing_goggles"],
      ["not-wearing", "not_wearing_goggles"],
    ]) {
      const percent = result.probabilities[key] * 100;
      element(prefix + "-score").textContent = percent.toFixed(1) + "%";
      element(prefix + "-bar").style.width = percent + "%";
      element(prefix + "-bar").classList.toggle("is-selected", result.label === key);
      element(prefix + "-bar").setAttribute("aria-valuenow", percent.toFixed(1));
    }
    element("dimensions").textContent = result.image_size.width + " × "
      + result.image_size.height + " px";
    element("timing").textContent = "本次耗时 " + (result.elapsed_ms / 1000).toFixed(2)
      + " 秒 · " + result.device;
    element("result-meta").hidden = false;
    const names = { train: "训练集", val: "验证集", test: "测试集" };
    element("known-note").textContent = result.known_splits.length
      ? "这张图与已登记头像完全重复：" + result.known_splits
        .map((name) => names[name] || name).join("、") + "。它不属于独立新图。"
      : "未匹配到已登记头像的相同字节或像素；这不保证图像来源独立。";
    element("known-note").hidden = false;
    resultURL = URL.createObjectURL(
      new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
    );
    element("download").href = resultURL;
    element("download").hidden = false;
  }

  async function predict() {
    if (!file || !metadataReady || busy || reading || connecting) {
      return;
    }
    const current = ++requestId;
    controller = new AbortController();
    busy = true;
    connectionError = false;
    clearResult();
    element("result-state").textContent = "识别中";
    element("result-symbol").textContent = "…";
    element("result-title").textContent = "模型正在查看头像";
    element("result-description").textContent = "首次识别需要加载模型，可能稍慢。请保留本机启动器。";
    updateControls();
    let receivedResponse = false;
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": file.type || "application/octet-stream",
          "X-Model-Id": element("model").value,
        },
        body: file,
        signal: controller.signal,
      });
      receivedResponse = true;
      const result = await response.json();
      if (current !== requestId) {
        return;
      }
      setConnection(true, "本机服务已连接");
      if (!response.ok) {
        throw new Error(result.error || "识别失败，请稍后再试。");
      }
      renderResult(result);
    } catch (error) {
      if (current === requestId && error.name !== "AbortError") {
        clearResult();
        connectionError = !receivedResponse && error instanceof TypeError;
        const message = connectionError
          ? "无法连接本机服务。请在启动器中确认服务已启动，再点“开始识别”重试。"
          : error instanceof SyntaxError
            ? "服务返回内容无法读取，请在启动器中查看日志并重启服务。"
            : error.message || "本次识别失败，请稍后再试。";
        showError(message);
        element("result-state").textContent = "未完成";
        element("result-title").textContent = "本次未完成识别";
        element("result-description").textContent = message;
        if (connectionError) {
          setConnection(false, "本机服务连接中断");
        }
      }
    } finally {
      if (current === requestId) {
        busy = false;
        controller = null;
        updateControls();
      }
    }
  }

  async function connect() {
    if (connecting || busy) {
      return;
    }
    connecting = true;
    element("connection").textContent = "正在连接本机服务";
    element("connection").className = "connection";
    updateControls();
    const timeoutController = new AbortController();
    const timeout = setTimeout(() => timeoutController.abort(), 10000);
    try {
      const response = await fetch("/api/models", { signal: timeoutController.signal });
      if (!response.ok) {
        throw new Error("服务未就绪");
      }
      const metadata = await response.json();
      if (!Array.isArray(metadata.models) || !metadata.models.length) {
        throw new Error("服务没有提供可用模型");
      }
      const previousModel = element("model").value;
      const options = metadata.models.map((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.name + " · " + model.run_name;
        option.dataset.note = model.note;
        return option;
      });
      element("model").replaceChildren(...options);
      if (metadata.models.some((model) => model.id === previousModel)) {
        element("model").value = previousModel;
      } else if (metadataReady) {
        cancelRequest();
        clearResult();
      }
      metadataReady = true;
      maxUploadBytes = metadata.max_upload_bytes || maxUploadBytes;
      updateModelNote();
      setConnection(true, "本机服务已连接");
      if (connectionError) {
        clearResult();
        connectionError = false;
      }
    } catch {
      setConnection(false, "本机服务未连接");
      element("model-note").textContent = "请先在图形启动器中启动服务，再点“重新连接”。";
    } finally {
      clearTimeout(timeout);
      connecting = false;
      updateControls();
    }
  }

  element("choose").addEventListener("click", () => element("file").click());
  element("replace").addEventListener("click", () => element("file").click());
  element("clear").addEventListener("click", () => {
    setFile(null);
    element("choose").focus();
  });
  element("file").addEventListener("change", (event) => {
    const selected = event.target.files[0];
    event.target.value = "";
    if (selected) {
      setFile(selected);
    }
  });
  for (const type of ["dragenter", "dragover"]) {
    element("dropzone").addEventListener(type, (event) => {
      event.preventDefault();
      element("dropzone").classList.add("is-dragging");
    });
  }
  element("dropzone").addEventListener("dragleave", (event) => {
    if (!element("dropzone").contains(event.relatedTarget)) {
      element("dropzone").classList.remove("is-dragging");
    }
  });
  element("dropzone").addEventListener("drop", (event) => {
    event.preventDefault();
    element("dropzone").classList.remove("is-dragging");
    if (event.dataTransfer.files.length !== 1) {
      setFile(null);
      showError("每次请只放入一张单人头像。");
      return;
    }
    setFile(event.dataTransfer.files[0]);
  });
  for (const type of ["dragover", "drop"]) {
    window.addEventListener(type, (event) => event.preventDefault());
  }
  element("model").addEventListener("change", () => {
    cancelRequest();
    connectionError = false;
    clearResult();
    updateModelNote();
    updateControls();
  });
  element("predict").addEventListener("click", predict);
  element("reconnect").addEventListener("click", connect);
  window.addEventListener("pagehide", () => {
    selectionId += 1;
    cancelRequest();
    if (previewURL) {
      URL.revokeObjectURL(previewURL);
    }
    if (resultURL) {
      URL.revokeObjectURL(resultURL);
    }
  });
  connect();
})();

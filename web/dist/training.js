"use strict";

(() => {
  const element = (id) => document.getElementById(id);
  const percentage = (value) => (value * 100).toFixed(2) + "%";
  const svgNamespace = "http://www.w3.org/2000/svg";
  let summary = null;

  function svgElement(tag, attributes, text) {
    const node = document.createElementNS(svgNamespace, tag);
    for (const [key, value] of Object.entries(attributes)) {
      node.setAttribute(key, String(value));
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function drawCurve() {
    if (!summary) {
      return;
    }
    const run = summary.experiments.find((item) => item.id === element("curve-run").value);
    const loss = element("curve-metric").value === "loss";
    const series = loss
      ? [["train_loss", "训练损失", "#1d7a68"], ["val_loss", "验证损失", "#c38e68"]]
      : [["val_macro_f1", "验证 macro-F1", "#1d7a68"]];
    const maximum = loss
      ? Math.max(...run.history.flatMap((row) => [row.train_loss, row.val_loss])) * 1.1
      : 1;
    const chart = element("training-chart");
    const width = Math.max(230, chart.getBoundingClientRect().width);
    chart.setAttribute("viewBox", "0 0 " + width + " 240");
    chart.replaceChildren();
    chart.setAttribute("aria-label", run.title + (loss ? "训练与验证损失" : "验证macro-F1"));
    const x = (epoch) => 48 + (epoch - 1) * (width - 68) / Math.max(1, run.history.length - 1);
    const y = (value) => 191 - value / maximum * 163;
    for (let tick = 0; tick <= 4; tick++) {
      const value = maximum * tick / 4;
      chart.append(svgElement("line", {
        x1: 48, x2: width - 20, y1: y(value), y2: y(value), stroke: "#e5ece7",
      }));
      chart.append(svgElement("text", {
        x: 37, y: y(value) + 4, "text-anchor": "end", class: "chart-label",
      }, value.toFixed(2)));
    }
    for (const [field, title, color] of series) {
      const points = run.history.map((row) => `${x(row.epoch)},${y(row[field])}`).join(" ");
      chart.append(svgElement("polyline", {points, fill: "none", stroke: color, "stroke-width": 2.5}));
      for (const row of run.history) {
        const dot = svgElement("circle", {cx: x(row.epoch), cy: y(row[field]), r: 3.5, fill: color});
        dot.append(svgElement("title", {}, `第${row.epoch}轮 · ${title} ${row[field].toFixed(4)}`));
        chart.append(dot);
      }
    }
    const labelStep = Math.ceil(run.history.length / 6);
    for (const row of run.history) {
      if ((row.epoch - 1) % labelStep !== 0 && row.epoch !== run.history.length) {
        continue;
      }
      chart.append(svgElement("text", {
        x: x(row.epoch), y: 211, "text-anchor": "middle", class: "chart-label",
      }, row.epoch));
    }
    chart.append(svgElement("text", {
      x: width / 2, y: 236, "text-anchor": "middle", class: "chart-label",
    }, "训练轮次 / epoch"));
    element("curve-caption").textContent = loss
      ? "绿色：训练损失；棕色：验证损失。增强与类别加权会影响损失，不能单独用它代表识别正确率。"
      : `共${run.history.length}轮、${run.updates}次参数更新；第${run.best_epoch}轮验证macro-F1最佳，为${run.validation.macro_f1.toFixed(4)}。`;
  }

  function renderSummary(data) {
    summary = data;
    const totals = Object.entries(data.counts).map(([name, counts]) => ({
      name,
      ...counts,
      total: counts.not_wearing_goggles + counts.wearing_goggles,
    }));
    element("data-count").textContent = totals.reduce((sum, item) => sum + item.total, 0) + " 张";
    element("split-counts").textContent = totals.map((item) => `${item.name} ${item.total}`).join(" / ");
    const control = data.experiments.find((run) => run.selected);
    element("validation-accuracy").textContent = percentage(control.validation.accuracy);
    element("validation-f1").textContent = "验证 macro-F1 " + control.validation.macro_f1.toFixed(4);
    element("test-accuracy").textContent = percentage(data.historical_test.accuracy);
    element("test-recall").textContent = "历史佩戴召回 " + percentage(data.historical_test.wearing_recall) + " · 已封存";
    for (const run of data.experiments) {
      const option = document.createElement("option");
      option.value = run.id;
      option.textContent = run.title;
      element("curve-run").append(option);
      const row = document.createElement("tr");
      if (run.selected) {
        row.className = "selected-run";
      }
      const values = [run.title, run.history.length, run.best_epoch,
        percentage(run.validation.accuracy), run.validation.macro_f1.toFixed(4),
        percentage(run.validation.wearing_recall)];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      element("experiment-rows").append(row);
    }
    const names = {train: "训练集", val: "验证集", test: "已封存测试"};
    for (const item of totals) {
      const box = document.createElement("p");
      const title = document.createElement("strong");
      title.textContent = `${names[item.name]} · ${item.total}张`;
      const detail = document.createElement("span");
      detail.textContent = `未佩戴 ${item.not_wearing_goggles} / 佩戴 ${item.wearing_goggles}`;
      box.append(title, detail);
      element("data-breakdown").append(box);
    }
    drawCurve();
  }

  element("curve-run").addEventListener("change", drawCurve);
  element("curve-metric").addEventListener("change", drawCurve);
  window.addEventListener("resize", drawCurve);
  fetch("/training-summary.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error("训练摘要未就绪");
      }
      return response.json();
    })
    .then(renderSummary)
    .catch(() => {
      element("training-error").textContent = "训练记录暂时无法读取，请检查应用文件是否完整。识别服务状态见页面右上角。";
      element("training-error").hidden = false;
    });
})();

import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const TEMPLATE = "D:\\MI_project\\inputs\\template_zky_purple.pptx";
const OUT = "D:\\MI_project\\outputs\\LLM_sycophancy_locate_steer_improve_presentation.pptx";
const QA_DIR = "C:\\Users\\30484\\AppData\\Local\\Temp\\codex-presentations\\019f5bdb-mi-sycophancy\\tmp\\final-qa";

const PURPLE = "#78006E";
const DARK = "#333333";
const MUTED = "#666666";

function shape(slide, index) {
  return slide.shapes.items[index];
}

function setText(target, text) {
  if (!target) return;
  target.text = text;
}

function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize: style.fontSize ?? 20,
    bold: style.bold ?? false,
    color: style.color ?? DARK,
    typeface: "Microsoft YaHei",
    alignment: style.alignment ?? "left",
  };
  return box;
}

function cover(slide, position, fill = "white") {
  return slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function addPanel(slide, position, fill = "#FFFFFF") {
  return slide.shapes.add({
    geometry: "roundRect",
    position,
    fill,
    line: { style: "solid", fill: "#E8D7E8", width: 1 },
    borderRadius: "rounded-lg",
    shadow: "shadow-sm",
  });
}

function addKpi(slide, title, value, note, position) {
  addPanel(slide, position);
  addText(slide, title, { left: position.left + 18, top: position.top + 16, width: position.width - 36, height: 28 }, { fontSize: 18, bold: true, color: DARK });
  addText(slide, value, { left: position.left + 18, top: position.top + 48, width: position.width - 36, height: 54 }, { fontSize: 34, bold: true, color: PURPLE, alignment: "center" });
  addText(slide, note, { left: position.left + 18, top: position.top + 104, width: position.width - 36, height: 44 }, { fontSize: 14, color: MUTED, alignment: "center" });
}

function addResultTable(slide, values, position) {
  cover(slide, position, "white");
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: position.left,
    top: position.top,
    width: position.width,
    height: position.height,
    values,
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  for (let c = 0; c < values[0].length; c += 1) {
    table.getCell(0, c).fill = PURPLE;
    table.getCell(0, c).text.style = { fontSize: 14, bold: true, color: "white", typeface: "Microsoft YaHei" };
  }
  for (let r = 1; r < values.length; r += 1) {
    for (let c = 0; c < values[0].length; c += 1) {
      table.getCell(r, c).text.style = { fontSize: 12, color: "#111111", typeface: "Microsoft YaHei" };
    }
  }
  table.borders.assign({ style: "solid", fill: "#D8C4D8", width: 1 });
  return table;
}

function addMiniBars(slide, items, position, maxValue = 1) {
  const rowH = position.height / items.length;
  const labelW = Math.min(132, position.width * 0.5);
  const gap = 10;
  const valueW = 44;
  const barW = Math.max(42, position.width - labelW - gap - valueW);
  items.forEach((item, i) => {
    const y = position.top + i * rowH;
    addText(slide, item.label, { left: position.left, top: y + 1, width: labelW, height: 22 }, { fontSize: 12, bold: true });
    const barLeft = position.left + labelW + gap;
    cover(slide, { left: barLeft, top: y + 5, width: barW, height: 14 }, "#EFE6EF");
    cover(slide, { left: barLeft, top: y + 5, width: Math.max(2, barW * item.value / maxValue), height: 14 }, PURPLE);
    addText(slide, item.display ?? String(item.value), { left: barLeft + barW + 6, top: y, width: valueW, height: 22 }, { fontSize: 11, color: MUTED });
  });
}

function clearLocalText(slide, keepShapeIndices = []) {
  slide.shapes.items.forEach((item, idx) => {
    if (keepShapeIndices.includes(idx)) return;
    if (item.text?.toString?.().trim()) item.text = "";
  });
  if (slide.tables?.items) {
    slide.tables.items.forEach((table) => {
      if (typeof table.delete === "function") table.delete();
    });
  }
}

function speaker(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.rm(QA_DIR, { recursive: true, force: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const presentation = await PresentationFile.importPptx(await FileBlob.load(TEMPLATE));
  const originals = [...presentation.slides.items];
  const sourceMap = [2, 5, 9, 31, 12, 36, 16, 36, 12, 16, 54];
  const slides = sourceMap.map((sourceSlide, idx) => {
    const dup = originals[sourceSlide - 1].duplicate();
    dup.moveTo(idx);
    return dup;
  });
  for (const s of [...presentation.slides.items]) {
    if (!slides.includes(s)) s.delete();
  }

  // 1. Cover
  setText(shape(slides[0], 0), "LLM 迎合现象\n机制可解释性复现");
  setText(shape(slides[0], 1), "Locate-Steer-Improve | Qwen2.5-0.5B CPU 实验");
  setText(shape(slides[0], 2), "南京大学 | 智能科学与技术学院");
  setText(shape(slides[0], 3), "报告人：MI Project");
  setText(shape(slides[0], 4), "时间：2026.07.14");
  speaker(slides[0], "本报告以 sycophancy 为具体现象，展示 100 样本下的 locate、steer、improve 三段闭环。");

  // 2. Contents
  ["问题与复现对象", "Locate：定位关键层", "Steer/Improve：干预与缓解", "总结、局限与拓展"].forEach((txt, i) => {
    setText(shape(slides[1], [2, 6, 10, 14][i]), txt);
  });

  // 3. Background
  setText(shape(slides[2], 0), "项目问题与复现对象");
  setText(shape(slides[2], 1), "PART  01");
  setText(shape(slides[2], 2), "01");
  setText(
    shape(slides[2], 3),
    "问题：用户给出错误观点时，小模型是否会把事实判断让位给迎合？\n复现：参考 sycophancy 机制论文与 actionable MI 框架，在 Qwen2.5-0.5B 上跑通可操作的 Locate-Steer-Improve 链路。\n规模：100 条 MMLU 样本，覆盖 56 个 subject，本地 CPU 完成。",
  );

  // 4. Pipeline
  setText(shape(slides[3], 11), "Locate-Steer-Improve 实验链路");
  const pipelineText = [
    ["现象构造", "MMLU 题目 + 用户错误观点，形成 plain / opinion / persona 三类条件"],
    ["Logit Lens", "逐层读取正确答案与用户观点选项概率，观察层间偏置"],
    ["Patching", "用 plain 表示替换 opinion 表示，估计因果恢复层"],
    ["Steering", "plain-opinion 向量注入推理过程，测试行为改变"],
    ["Improve", "比较反迎合提示语，并提出层级 sweep 拓展"],
  ];
  [2, 4, 6, 8, 10].forEach((idx, i) => setText(shape(slides[3], idx), `${pipelineText[i][0]}\n${pipelineText[i][1]}`));
  ["数据", "定位", "因果", "干预", "缓解"].forEach((txt, i) => setText(shape(slides[3], [27, 28, 29, 30, 31][i]), txt));

  // 5. Locate method
  setText(shape(slides[4], 1), "1.1 Locate：从层级趋势到因果恢复");
  const locateBullets = [
    "行为层：比较 plain、opinion_only、prefix_and_opinion 的 accuracy 与 sycophancy_rate。",
    "Logit Lens：对每层最后 token hidden state 解码，跟踪 p(correct) 与 p(opinion)。",
    "Activation Patching：把 opinion prompt 某层 residual 替换为 plain prompt 表示。",
    "定位结论：logit lens 指向第 20 层，patching 指向第 20-23 层后段 residual 区域。",
  ];
  [2, 5, 8, 11].forEach((idx, i) => setText(shape(slides[4], idx), locateBullets[i]));

  // 6. Locate result
  setText(shape(slides[5], 1), "1.2 Locate：100 样本关键层结果");
  clearLocalText(slides[5], [1]);
  cover(slides[5], { left: 44, top: 112, width: 1192, height: 560 }, "#FFFFFF");
  addKpi(slides[5], "样本规模", "100 条", "56 个 MMLU subject", { left: 66, top: 130, width: 255, height: 158 });
  addKpi(slides[5], "关键区间", "20-23", "Logit Lens 与 Patching 对齐", { left: 350, top: 130, width: 255, height: 158 });
  addKpi(slides[5], "最高恢复", "+1.256", "layer 22 patch_delta", { left: 634, top: 130, width: 255, height: 158 });
  addPanel(slides[5], { left: 918, top: 130, width: 295, height: 158 }, "#F8F1F8");
  addText(slides[5], "核心发现", { left: 944, top: 152, width: 240, height: 30 }, { fontSize: 22, bold: true, color: PURPLE, alignment: "center" });
  addText(slides[5], "用户观点相关表示主要在后段 residual layers 影响输出决策；第 20 层是 logit lens 中最清晰的竞争点。", { left: 946, top: 195, width: 238, height: 66 }, { fontSize: 16, color: DARK, alignment: "center" });
  addResultTable(slides[5], [
    ["方法", "条件/层", "主要指标", "观察"],
    ["行为", "三类 prompt", "acc 0.31-0.35", "最终答案无强迎合放大"],
    ["Logit Lens", "opinion L20", "gap=0.0307", "观点选项优势最高"],
    ["Patching", "L22", "delta=+1.256", "plain 表示恢复 margin"],
    ["结论", "L20-23", "一致高响应", "后段 residual 优先干预"],
  ], { left: 82, top: 330, width: 820, height: 248 });
  addMiniBars(slides[5], [
    { label: "L20 logit gap", value: 0.0307, display: "0.031" },
    { label: "L22 patch", value: 1.2562, display: "+1.256" },
    { label: "opinion syc", value: 0.23, display: "0.23" },
  ], { left: 940, top: 370, width: 260, height: 130 }, 1.2562);
  addText(slides[5], "解释：logit lens 给出层级趋势，activation patching 给出替换 plain 表示后的因果恢复。", { left: 948, top: 528, width: 245, height: 56 }, { fontSize: 15, color: DARK, alignment: "center" });
  speaker(slides[5], "Locate 的关键不是只看最终答案，而是把每层对正确答案和用户观点选项的支持拆出来。");

  // 7. Steer
  setText(shape(slides[6], 1), "2.1 Steer：用向量算术做推理时干预");
  setText(shape(slides[6], 4), "干预向量");
  setText(shape(slides[6], 5), "");
  addText(slides[6], "steering_vector = mean(hidden_plain - hidden_opinion)\n在指定层最后 token 表示上注入 alpha * vector。", { left: 555, top: 262, width: 575, height: 72 }, { fontSize: 20, color: "white" });
  setText(shape(slides[6], 8), "当前结果");
  setText(shape(slides[6], 9), "");
  addText(slides[6], "layer 23 的 alpha=-2/0/2 均为 accuracy=0.32、sycophancy_rate=0.23。\n负结果说明单层小幅线性注入还不足以改变最终答案。", { left: 560, top: 542, width: 575, height: 92 }, { fontSize: 20, color: "white" });
  speaker(slides[6], "Steer 部分已经完成推理时向量注入。结果没有改变答案，但为后续第 20-23 层密集 sweep 提供了方向。");

  // 8. Improve
  setText(shape(slides[7], 1), "2.2 Improve：提示词缓解结果");
  clearLocalText(slides[7], [1]);
  cover(slides[7], { left: 44, top: 112, width: 1192, height: 560 }, "#FFFFFF");
  addKpi(slides[7], "样本规模", "100 条", "同一 MMLU 子集", { left: 66, top: 130, width: 255, height: 158 });
  addKpi(slides[7], "最佳准确率", "0.32", "none / truth_priority", { left: 350, top: 130, width: 255, height: 158 });
  addKpi(slides[7], "迎合率", "0.23", "四种模式基本一致", { left: 634, top: 130, width: 255, height: 158 });
  addPanel(slides[7], { left: 918, top: 130, width: 295, height: 158 }, "#F8F1F8");
  addText(slides[7], "结论", { left: 944, top: 152, width: 240, height: 30 }, { fontSize: 22, bold: true, color: PURPLE, alignment: "center" });
  addText(slides[7], "简单提示“不要迎合”没有带来可见收益；对小模型难题，需要结合内部层定位做干预。", { left: 946, top: 195, width: 238, height: 66 }, { fontSize: 16, color: DARK, alignment: "center" });
  addResultTable(slides[7], [
    ["mitigation", "n", "accuracy", "sycophancy"],
    ["none", "100", "0.32", "0.23"],
    ["truth_priority", "100", "0.32", "0.23"],
    ["anti_sycophancy", "100", "0.31", "0.23"],
    ["verify_then_answer", "100", "0.31", "0.23"],
  ], { left: 82, top: 330, width: 820, height: 248 });
  addMiniBars(slides[7], [
    { label: "none", value: 0.23, display: "0.23" },
    { label: "truth", value: 0.23, display: "0.23" },
    { label: "anti", value: 0.23, display: "0.23" },
    { label: "verify", value: 0.23, display: "0.23" },
  ], { left: 948, top: 370, width: 245, height: 130 }, 0.30);
  addText(slides[7], "下一步：把 mitigation 与 layer 20-23 steering sweep 结合，并扩展 attention/MLP 粒度。", { left: 948, top: 528, width: 245, height: 56 }, { fontSize: 15, color: DARK, alignment: "center" });
  speaker(slides[7], "Improve 的结论是 prompt-only 作用有限，这也符合可操作 MI 的动机：先定位，再干预。");

  // 9. Paper reproduction
  setText(shape(slides[8], 1), "3.1 顶会/Arxiv 机制可解释性复现");
  const paperBullets = [
    "复现对象：sycophancy 机制论文中“用户观点覆盖事实判断”的内部起源分析。",
    "方法对齐：Logit Lens 观察层间偏置，Activation Patching 给出更强因果证据。",
    "框架对齐：Locate -> Steer -> Improve，对应 actionable MI survey 的实践链路。",
    "拓展：整理成本地 CPU 可运行的小模型实验，并补充 100 样本 patching 与中文材料。",
  ];
  [2, 5, 8, 11].forEach((idx, i) => setText(shape(slides[8], idx), paperBullets[i]));

  // 10. Analysis
  setText(shape(slides[9], 1), "4.1 个人分析、局限与下一步");
  setText(shape(slides[9], 4), "我的分析");
  setText(shape(slides[9], 5), "");
  addText(slides[9], "Sycophancy 更像是用户观点选项在后段 residual stream 中获得竞争优势。\nLogit Lens 看趋势，Patching 验证因果。", { left: 555, top: 262, width: 575, height: 72 }, { fontSize: 20, color: "white" });
  setText(shape(slides[9], 8), "下一步");
  setText(shape(slides[9], 9), "");
  addText(slides[9], "扩展到 300-500 条样本；围绕第 20-23 层 sweep alpha；再拆到 attention / MLP 模块。", { left: 560, top: 552, width: 575, height: 76 }, { fontSize: 20, color: "white" });

  // 11. Ending
  setText(shape(slides[10], 0), "谢谢各位老师指导");
  addText(slides[10], "代码、100 样本结果、报告与 PPT 已整理在 D:\\MI_project\\outputs", { left: 360, top: 475, width: 620, height: 44 }, { fontSize: 20, bold: true, color: PURPLE, alignment: "center" });

  const finalPptx = await PresentationFile.exportPptx(presentation);
  await finalPptx.save(OUT);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(QA_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  await writeBlob(path.join(QA_DIR, "final-montage.webp"), await presentation.export({
    format: "webp",
    montage: { columns: 3, slideWidth: 420, padding: 20, gap: 12, background: "#ffffff" },
    scale: 1,
  }));
  const inspect = await presentation.inspect({ kind: "slide,textbox,shape,table,chart,notes,layout", maxChars: 50000 });
  await fs.writeFile(path.join(QA_DIR, "final-inspect.ndjson"), inspect.ndjson, "utf8");
  console.log(OUT);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});


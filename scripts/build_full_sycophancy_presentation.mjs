import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const ROOT = "D:\\MI_project";
const TEMPLATE = path.join(ROOT, "inputs", "template_zky_purple.pptx");
const OUT = path.join(ROOT, "outputs", "LLM_sycophancy_locate_steer_improve_presentation_full.pptx");
const QA_DIR = path.join(ROOT, "outputs", "LLM_sycophancy_locate_steer_improve_presentation_full");
const METRICS = JSON.parse(await fs.readFile(path.join(ROOT, "outputs", "full_analysis", "key_metrics.json"), "utf-8"));

const PURPLE = "#78006E";
const LILAC = "#F4EAF5";
const GREEN = "#2F6B4F";
const RED = "#B33A3A";
const BLUE = "#4E79A7";
const DARK = "#252525";
const MUTED = "#666666";

function shape(slide, index) {
  return slide.shapes.items[index];
}

function setText(target, text, style = {}) {
  if (!target) return;
  target.text = text;
  target.text.style = {
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? DARK,
    typeface: "Microsoft YaHei",
  };
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
    fontSize: style.fontSize ?? 18,
    bold: style.bold ?? false,
    color: style.color ?? DARK,
    typeface: "Microsoft YaHei",
    alignment: style.alignment ?? "left",
  };
  return box;
}

function rect(slide, position, fill = "white", line = fill) {
  return slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { style: "solid", fill: line, width: 0 },
  });
}

function panel(slide, position, fill = "#FFFFFF") {
  return slide.shapes.add({
    geometry: "roundRect",
    position,
    fill,
    line: { style: "solid", fill: "#E4CEE6", width: 1 },
    borderRadius: "rounded-lg",
  });
}

function title(slide, text, kicker = "") {
  rect(slide, { left: 34, top: 24, width: 1190, height: 80 }, "white");
  if (kicker) addText(slide, kicker, { left: 70, top: 30, width: 220, height: 24 }, { fontSize: 14, bold: true, color: PURPLE });
  addText(slide, text, { left: 70, top: 54, width: 980, height: 42 }, { fontSize: 30, bold: true, color: DARK });
  rect(slide, { left: 70, top: 104, width: 180, height: 4 }, PURPLE);
}

function clearBody(slide) {
  rect(slide, { left: 44, top: 112, width: 1192, height: 560 }, "white");
}

function wipeInheritedContent(slide) {
  for (const item of slide.shapes.items) {
    if (item.text?.toString?.().trim()) item.text = "";
  }
  if (slide.tables?.items) {
    for (const item of [...slide.tables.items]) {
      if (typeof item.delete === "function") item.delete();
    }
  }
}

function kpi(slide, label, value, note, x, y, w = 250) {
  panel(slide, { left: x, top: y, width: w, height: 140 }, LILAC);
  addText(slide, label, { left: x + 18, top: y + 16, width: w - 36, height: 28 }, { fontSize: 17, bold: true, color: DARK, alignment: "center" });
  addText(slide, value, { left: x + 18, top: y + 46, width: w - 36, height: 44 }, { fontSize: 31, bold: true, color: PURPLE, alignment: "center" });
  addText(slide, note, { left: x + 18, top: y + 96, width: w - 36, height: 34 }, { fontSize: 13, color: MUTED, alignment: "center" });
}

function table(slide, values, position, fontSize = 13) {
  const t = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: position.left,
    top: position.top,
    width: position.width,
    height: position.height,
    values,
  });
  t.styleOptions = { headerRow: true, bandedRows: true };
  for (let c = 0; c < values[0].length; c += 1) {
    t.getCell(0, c).fill = PURPLE;
    t.getCell(0, c).text.style = { fontSize, bold: true, color: "white", typeface: "Microsoft YaHei" };
  }
  for (let r = 1; r < values.length; r += 1) {
    for (let c = 0; c < values[0].length; c += 1) {
      t.getCell(r, c).text.style = { fontSize, color: "#222222", typeface: "Microsoft YaHei" };
    }
  }
  t.borders.assign({ style: "solid", fill: "#DDC7E0", width: 1 });
  return t;
}

function bars(slide, items, position, maxValue = 1) {
  const rowH = position.height / items.length;
  const labelW = position.width * 0.45;
  const barW = position.width * 0.38;
  for (const [i, item] of items.entries()) {
    const y = position.top + i * rowH;
    addText(slide, item.label, { left: position.left, top: y, width: labelW, height: 24 }, { fontSize: 13, bold: true });
    rect(slide, { left: position.left + labelW + 12, top: y + 6, width: barW, height: 13 }, "#EFE4F0");
    rect(slide, { left: position.left + labelW + 12, top: y + 6, width: Math.max(2, barW * item.value / maxValue), height: 13 }, item.color ?? PURPLE);
    addText(slide, item.text, { left: position.left + labelW + barW + 20, top: y - 1, width: 70, height: 22 }, { fontSize: 12, color: MUTED });
  }
}

function pct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function fmt(x) {
  return Number(x).toFixed(3);
}

function note(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const behavior = [
  ["条件", "样本数", "准确率", "迎合率"],
  ["plain", "14042", "43.9%", "19.1%"],
  ["opinion_only", "14042", "17.8%", "69.2%"],
  ["prefix_and_opinion", "14042", "21.4%", "60.6%"],
];

const mitigation = [
  ["方法", "准确率", "迎合率", "消除率"],
  ["none", "17.8%", "69.2%", "0.0%"],
  ["truth_priority", "27.3%", "43.5%", "43.5%"],
  ["anti_sycophancy", "30.3%", "39.3%", "47.9%"],
  ["verify_then_answer", "21.9%", "53.3%", "32.7%"],
  ["counter_opinion_check", "22.9%", "53.2%", "31.7%"],
];

const patchRows = METRICS.top_patching.slice(0, 5).map((r) => [
  String(r.layer),
  fmt(r.base_margin),
  fmt(r.patched_margin),
  `+${fmt(r.mean_patch_delta)}`,
]);

const lensRows = METRICS.top_logit_lens.slice(0, 5).map((r) => [
  r.condition,
  String(r.layer),
  fmt(r.p_answer),
  fmt(r.p_opinion),
  fmt(r.opinion_minus_answer),
]);

async function main() {
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.rm(QA_DIR, { recursive: true, force: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const presentation = await PresentationFile.importPptx(await FileBlob.load(TEMPLATE));
  const originals = [...presentation.slides.items];
  const sourceMap = [2, 5, 9, 31, 36, 36, 36, 36, 12, 16, 54];
  const slides = sourceMap.map((sourceSlide, idx) => {
    const dup = originals[sourceSlide - 1].duplicate();
    dup.moveTo(idx);
    return dup;
  });
  for (const s of [...presentation.slides.items]) {
    if (!slides.includes(s)) s.delete();
  }
  for (const slide of slides) {
    wipeInheritedContent(slide);
  }

  // Cover
  setText(shape(slides[0], 0), "LLM 迎合现象机制可解释性复现", { color: "white", fontSize: 30, bold: true });
  setText(shape(slides[0], 1), "Locate-Steer-Improve | Qwen2.5-0.5B | MMLU raw 全量 14042 条", { color: "white", fontSize: 20 });
  setText(shape(slides[0], 2), "智能科学与技术学院", { color: "white", fontSize: 18 });
  setText(shape(slides[0], 3), "MI Project", { color: "white", fontSize: 18 });
  setText(shape(slides[0], 4), "2026.07", { color: "white", fontSize: 18 });
  note(slides[0], "本报告展示从 mmlu_raw 全量数据出发的 locate、steer、improve 全链路结果。");

  // Contents
  clearBody(slides[1]);
  title(slides[1], "这次实验回答三个问题", "目录");
  const contents = [
    ["1", "模型是否会被错误用户观点牵引？"],
    ["2", "这种牵引集中在哪些内部层？"],
    ["3", "推理期向量干预能否消除迎合？"],
    ["4", "哪类改进策略最有效？"],
  ];
  for (const [i, [idx, txt]] of contents.entries()) {
    panel(slides[1], { left: 150, top: 150 + i * 105, width: 980, height: 76 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slides[1], idx, { left: 190, top: 162 + i * 105, width: 70, height: 44 }, { fontSize: 32, bold: true, color: PURPLE, alignment: "center" });
    addText(slides[1], txt, { left: 285, top: 168 + i * 105, width: 760, height: 38 }, { fontSize: 24, bold: true });
  }

  // Scope
  clearBody(slides[2]);
  title(slides[2], "全量数据把问题从演示扩展到稳定证据", "实验范围");
  kpi(slides[2], "MMLU raw", "14042", "全量样本", 78, 150);
  kpi(slides[2], "学科覆盖", "57", "subject", 360, 150);
  kpi(slides[2], "模型", "0.5B", "Qwen2.5-Instruct", 642, 150);
  kpi(slides[2], "机制链路", "3 段", "locate / steer / improve", 924, 150);
  table(slides[2], [
    ["环节", "产物", "全量规模"],
    ["Behavior", "locate_behavior.csv", "14042×3 = 42126"],
    ["Logit Lens", "locate_layer_logit_lens.csv", "14042×3×24 = 1011024"],
    ["Steering", "steer_sweep.csv", "14042×4×7 = 393176"],
    ["Patching", "activation_patching_summary.csv", "14042×24 = 337008"],
    ["Improve", "improve_prompt_mitigation.csv", "14042×5 = 70210"],
  ], { left: 130, top: 330, width: 1020, height: 250 }, 14);

  // Pipeline
  clearBody(slides[3]);
  title(slides[3], "实验链路从行为现象走到可操作改进", "方法");
  const steps = [
    ["Behavior", "比较 plain / opinion / persona 条件下的准确率与迎合率"],
    ["Logit Lens", "逐层解码正确答案与错误观点选项的竞争"],
    ["Activation Patching", "用 plain 表示替换 opinion 表示，验证关键层因果作用"],
    ["Vector Steering", "注入 hidden_plain - hidden_opinion 方向向量"],
    ["Prompt Improve", "比较反迎合提示语的实际消除率"],
  ];
  for (const [i, [h, b]] of steps.entries()) {
    const x = 70 + i * 236;
    panel(slides[3], { left: x, top: 185, width: 200, height: 230 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slides[3], String(i + 1), { left: x + 70, top: 206, width: 60, height: 42 }, { fontSize: 30, bold: true, color: PURPLE, alignment: "center" });
    addText(slides[3], h, { left: x + 18, top: 270, width: 164, height: 36 }, { fontSize: 19, bold: true, alignment: "center" });
    addText(slides[3], b, { left: x + 18, top: 325, width: 164, height: 72 }, { fontSize: 14, color: MUTED, alignment: "center" });
  }
  addText(slides[3], "复现定位：围绕 2024-2026 actionable mechanistic interpretability 框架，把“定位内部机制”与“推理期干预”连成闭环。", { left: 115, top: 490, width: 1050, height: 60 }, { fontSize: 21, bold: true, color: DARK, alignment: "center" });

  // Behavior
  clearBody(slides[4]);
  title(slides[4], "错误用户观点把迎合率推高到 69.2%", "Locate 1/3");
  table(slides[4], behavior, { left: 90, top: 150, width: 560, height: 250 }, 15);
  bars(slides[4], [
    { label: "plain 迎合率", value: 0.191, text: "19.1%", color: BLUE },
    { label: "opinion_only 迎合率", value: 0.692, text: "69.2%", color: RED },
    { label: "prefix+opinion 迎合率", value: 0.606, text: "60.6%", color: RED },
    { label: "plain 准确率", value: 0.439, text: "43.9%", color: GREEN },
  ], { left: 725, top: 160, width: 420, height: 190 }, 0.75);
  addText(slides[4], "结论：小模型在事实性选择题上并不是只“不会做题”，而是会明显把错误用户观点当作条件信号，导致正确答案概率与最终输出同时偏移。", { left: 120, top: 455, width: 1020, height: 72 }, { fontSize: 23, bold: true, color: PURPLE, alignment: "center" });

  // Logit lens
  clearBody(slides[5]);
  title(slides[5], "Logit Lens 显示错误观点信号集中在 20-22 层", "Locate 2/3");
  table(slides[5], [["condition", "layer", "p_answer", "p_opinion", "gap"], ...lensRows], { left: 70, top: 145, width: 720, height: 260 }, 13);
  kpi(slides[5], "最大 gap", "0.352", "opinion_only layer 21", 860, 160, 250);
  kpi(slides[5], "关键区域", "20-22", "后段 residual layers", 860, 330, 250);
  addText(slides[5], "意义：这些层不是最终输出后才表现出偏差，而是在逐层解码中已经出现 p(opinion) > p(answer) 的内部竞争优势。", { left: 95, top: 470, width: 1030, height: 70 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  // Patching
  clearBody(slides[6]);
  title(slides[6], "Activation Patching 证明后段层具有因果恢复作用", "Locate 3/3");
  table(slides[6], [["layer", "base margin", "patched margin", "delta"], ...patchRows], { left: 90, top: 145, width: 700, height: 270 }, 14);
  bars(slides[6], [
    { label: "layer 22", value: 1.794, text: "+1.794", color: PURPLE },
    { label: "layer 21", value: 1.692, text: "+1.692", color: PURPLE },
    { label: "layer 20", value: 1.691, text: "+1.691", color: PURPLE },
    { label: "layer 23", value: 1.503, text: "+1.503", color: PURPLE },
  ], { left: 835, top: 170, width: 330, height: 155 }, 1.85);
  addText(slides[6], "用 plain prompt 的表示替换 opinion prompt 的对应层表示后，correct-minus-opinion margin 从 -0.709 恢复到正值；这给出了比相关性更强的因果证据。", { left: 110, top: 460, width: 1010, height: 76 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  // Steering
  clearBody(slides[7]);
  title(slides[7], "向量干预方向正确，但消除幅度有限", "Steer");
  kpi(slides[7], "Sweep 规模", "393176", "4 层 × 7 alpha × 全量样本", 80, 150, 300);
  kpi(slides[7], "最佳 setting", "L20 α=-4", "steering 中最高消除率", 425, 150, 300);
  kpi(slides[7], "消除率", "6.4%", "低于 prompt mitigation", 770, 150, 300);
  table(slides[7], [
    ["setting", "accuracy", "sycophancy", "elimination"],
    ["L20 alpha=-4", "17.6%", "68.3%", "6.4%"],
    ["L20 alpha=4", "18.3%", "67.9%", "5.4%"],
    ["L21 alpha=4", "18.3%", "68.0%", "4.9%"],
    ["L23 alpha=-4", "17.8%", "68.4%", "4.7%"],
  ], { left: 145, top: 360, width: 990, height: 210 }, 14);
  addText(slides[7], "解释：单一平均向量能推动模型远离错误观点，但 Qwen2.5-0.5B 的事实能力较弱，向量干预容易同时扰动答案分布，因此改善不如 prompt 约束稳定。", { left: 115, top: 590, width: 1040, height: 44 }, { fontSize: 19, bold: true, color: PURPLE, alignment: "center" });

  // Improve
  clearBody(slides[8]);
  title(slides[8], "明确反迎合提示语是本轮最有效改进", "Improve");
  table(slides[8], mitigation, { left: 70, top: 140, width: 760, height: 300 }, 13);
  panel(slides[8], { left: 880, top: 150, width: 270, height: 140 }, LILAC);
  addText(slides[8], "最佳方法", { left: 910, top: 172, width: 210, height: 28 }, { fontSize: 18, bold: true, alignment: "center" });
  addText(slides[8], "anti_sycophancy", { left: 900, top: 212, width: 230, height: 30 }, { fontSize: 24, bold: true, color: PURPLE, alignment: "center" });
  addText(slides[8], "不要仅因用户观点而同意", { left: 910, top: 253, width: 210, height: 24 }, { fontSize: 13, color: MUTED, alignment: "center" });
  panel(slides[8], { left: 880, top: 325, width: 270, height: 140 }, LILAC);
  addText(slides[8], "迎合率下降", { left: 910, top: 348, width: 210, height: 28 }, { fontSize: 18, bold: true, alignment: "center" });
  addText(slides[8], "69.2% → 39.3%", { left: 895, top: 386, width: 240, height: 34 }, { fontSize: 27, bold: true, color: PURPLE, alignment: "center" });
  addText(slides[8], "总体准确率升到 30.3%", { left: 910, top: 432, width: 210, height: 24 }, { fontSize: 13, color: MUTED, alignment: "center" });
  addText(slides[8], "这说明最实用的策略不是单独依赖向量 steering，而是把机制定位结果转化为更明确的推理约束：先独立判断事实，再回答选项。", { left: 95, top: 515, width: 1050, height: 66 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  // Paper reproduction
  clearBody(slides[9]);
  title(slides[9], "复现价值在于把“可解释”推进到“可操作”", "顶会/Arxiv 复现");
  const claims = [
    ["复现框架", "参考 2024 practical review 与 2026 Locate, Steer, and Improve survey 的 actionable MI 思路。"],
    ["机制证据", "Logit Lens 找到层级相关性，Activation Patching 验证后段层的因果恢复。"],
    ["干预实验", "Vector Arithmetic 在推理期注入 hidden_plain - hidden_opinion 方向。"],
    ["扩展发现", "全量 MMLU raw 上，prompt mitigation 显著优于简单平均 steering vector。"],
  ];
  for (const [i, [h, b]] of claims.entries()) {
    panel(slides[9], { left: 105 + (i % 2) * 545, top: 155 + Math.floor(i / 2) * 170, width: 485, height: 128 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slides[9], h, { left: 130 + (i % 2) * 545, top: 178 + Math.floor(i / 2) * 170, width: 430, height: 28 }, { fontSize: 22, bold: true, color: PURPLE });
    addText(slides[9], b, { left: 130 + (i % 2) * 545, top: 220 + Math.floor(i / 2) * 170, width: 425, height: 56 }, { fontSize: 17, color: DARK });
  }

  // Conclusions
  clearBody(slides[10]);
  title(slides[10], "结论：迎合可定位、可干预，但最稳改进仍是约束推理", "总结");
  const takeaways = [
    "现象层面：opinion-only 把迎合率从 19.1% 推到 69.2%。",
    "定位层面：layer 20-22 同时被 logit lens 和 patching 指向。",
    "干预层面：steering 有方向性但幅度有限，最佳消除率 6.4%。",
    "改进层面：anti_sycophancy 提示语把迎合率降到 39.3%，并提升准确率。",
  ];
  for (const [i, txt] of takeaways.entries()) {
    panel(slides[10], { left: 145, top: 145 + i * 98, width: 990, height: 66 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slides[10], txt, { left: 180, top: 163 + i * 98, width: 920, height: 32 }, { fontSize: 23, bold: true, color: i === 3 ? PURPLE : DARK });
  }
  addText(slides[10], "后续拓展：用更大模型、按 subject 分组学习 steering vectors，并比较 ablation 与 SAE 特征级干预。", { left: 155, top: 560, width: 970, height: 46 }, { fontSize: 21, bold: true, color: PURPLE, alignment: "center" });

  for (const [index, slide] of slides.entries()) {
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(QA_DIR, `slide-${index + 1}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `slide-${index + 1}.layout.json`), await layout.text());
  }
  const montage = await presentation.export({ format: "png", montage: true, scale: 1 });
  await writeBlob(path.join(QA_DIR, "contact-sheet.png"), montage);
  await writeBlob(path.join(ROOT, "outputs", "final-contact-sheet-full.png"), montage);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT);
  console.log(`wrote ${OUT}`);
  console.log(`qa ${QA_DIR}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

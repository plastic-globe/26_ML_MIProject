import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const ROOT = "D:\\MI_project";
const TEMPLATE = path.join(ROOT, "inputs", "template_zky_purple.pptx");
const OUT = path.join(ROOT, "outputs", "LLM_sycophancy_locate_steer_improve_presentation_qwen15.pptx");
const QA_DIR = path.join(ROOT, "outputs", "LLM_sycophancy_locate_steer_improve_presentation_qwen15");
const CONTACT = path.join(ROOT, "outputs", "final-contact-sheet-qwen15.png");
const BASE = path.join(ROOT, "outputs", "qwen15_mmlu_raw_full_locate_steer_improve");
const METRICS = JSON.parse(await fs.readFile(path.join(ROOT, "outputs", "full_analysis_qwen15", "key_metrics.json"), "utf-8"));

const PURPLE = "#78006E";
const LILAC = "#F4EAF5";
const GREEN = "#2F6B4F";
const RED = "#B33A3A";
const BLUE = "#4E79A7";
const DARK = "#252525";
const MUTED = "#666666";

function pct(x) {
  return `${(Number(x) * 100).toFixed(1)}%`;
}

function fmt(x) {
  return Number(x).toFixed(3);
}

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
    alignment: style.alignment ?? "left",
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
  if (kicker) addText(slide, kicker, { left: 70, top: 30, width: 260, height: 24 }, { fontSize: 14, bold: true, color: PURPLE });
  addText(slide, text, { left: 70, top: 54, width: 1040, height: 42 }, { fontSize: 30, bold: true, color: DARK });
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

function note(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
}

function kpi(slide, label, value, noteText, x, y, w = 250) {
  panel(slide, { left: x, top: y, width: w, height: 140 }, LILAC);
  addText(slide, label, { left: x + 18, top: y + 16, width: w - 36, height: 28 }, { fontSize: 17, bold: true, color: DARK, alignment: "center" });
  addText(slide, value, { left: x + 18, top: y + 46, width: w - 36, height: 44 }, { fontSize: 31, bold: true, color: PURPLE, alignment: "center" });
  addText(slide, noteText, { left: x + 18, top: y + 96, width: w - 36, height: 34 }, { fontSize: 13, color: MUTED, alignment: "center" });
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
    addText(slide, item.text, { left: position.left + labelW + barW + 20, top: y - 1, width: 78, height: 22 }, { fontSize: 12, color: MUTED });
  }
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function rowsByCondition() {
  return [
    ["条件", "样本数", "准确率", "迎合率"],
    ["plain", "14042", "57.5%", "14.7%"],
    ["opinion_only", "14042", "47.1%", "32.5%"],
    ["prefix_and_opinion", "14042", "54.6%", "19.4%"],
  ];
}

const mitigationRows = [
  ["方法", "准确率", "迎合率", "消除率"],
  ["none", "47.1%", "32.5%", "0.0%"],
  ["truth_priority", "46.5%", "30.2%", "18.2%"],
  ["anti_sycophancy", "48.1%", "27.4%", "25.0%"],
  ["verify_then_answer", "34.7%", "51.4%", "2.0%"],
  ["counter_opinion_check", "37.3%", "44.0%", "6.5%"],
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
  for (const slide of slides) wipeInheritedContent(slide);

  setText(shape(slides[0], 0), "LLM 迎合现象机制可解释性复现", { color: "white", fontSize: 30, bold: true });
  setText(shape(slides[0], 1), "Locate-Steer-Improve | Qwen2.5-1.5B | MMLU raw 全量 14042 条", { color: "white", fontSize: 20 });
  setText(shape(slides[0], 2), "智能科学与技术学院", { color: "white", fontSize: 18 });
  setText(shape(slides[0], 3), "MI Project", { color: "white", fontSize: 18 });
  setText(shape(slides[0], 4), "2026.07", { color: "white", fontSize: 18 });
  note(slides[0], "本报告展示 Qwen2.5-1.5B-Instruct 在 mmlu_raw 全量数据上的 locate、steer、improve 结果。");

  clearBody(slides[1]);
  title(slides[1], "这次 1.5B 全量实验回答四个问题", "目录");
  const contents = [
    ["1", "较大模型是否仍会被错误用户观点牵引？"],
    ["2", "内部定位结果是否仍呈现同样的层级模式？"],
    ["3", "向量 steering 对 1.5B 是否更有效？"],
    ["4", "哪些 prompt mitigation 最实用？"],
  ];
  for (const [i, [idx, txt]] of contents.entries()) {
    panel(slides[1], { left: 150, top: 150 + i * 105, width: 980, height: 76 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slides[1], idx, { left: 190, top: 162 + i * 105, width: 70, height: 44 }, { fontSize: 32, bold: true, color: PURPLE, alignment: "center" });
    addText(slides[1], txt, { left: 285, top: 168 + i * 105, width: 760, height: 38 }, { fontSize: 24, bold: true });
  }

  clearBody(slides[2]);
  title(slides[2], "同一全量数据上，1.5B 提供了更强的对照证据", "实验范围");
  kpi(slides[2], "MMLU raw", "14042", "全量样本，不是 3000 条", 78, 150);
  kpi(slides[2], "学科覆盖", "57", "subjects", 360, 150);
  kpi(slides[2], "模型", "1.5B", "Qwen2.5-Instruct", 642, 150);
  kpi(slides[2], "框架", "CUDA", "PyTorch + Transformers", 924, 150);
  table(slides[2], [
    ["环节", "产物", "全量规模"],
    ["Behavior", "locate_behavior.csv", "14042×3 = 42126"],
    ["Logit Lens", "locate_layer_logit_lens.csv", "14042×3×28 = 1179528"],
    ["Steering", "steer_sweep.csv", "14042×4×7 = 393176"],
    ["Patching", "activation_patching_summary.csv", "14042×28 = 393176"],
    ["Improve", "improve_prompt_mitigation.csv", "14042×5 = 70210"],
  ], { left: 130, top: 330, width: 1020, height: 250 }, 14);

  clearBody(slides[3]);
  title(slides[3], "实验链路保持不变，便于与 0.5B 结果直接对照", "方法");
  const steps = [
    ["Behavior", "比较 plain / opinion / persona 条件下的准确率与迎合率"],
    ["Logit Lens", "逐层解码正确答案与错误观点选项的竞争"],
    ["Activation Patching", "用 plain 表示替换 opinion 表示，检验层级因果作用"],
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
  addText(slides[3], "核心对照：只替换模型规模，保持全量输入、locate-steer-improve 链路和评估定义一致。", { left: 115, top: 490, width: 1050, height: 60 }, { fontSize: 21, bold: true, color: DARK, alignment: "center" });

  clearBody(slides[4]);
  title(slides[4], "1.5B 仍会迎合，但幅度明显低于 0.5B", "Locate 1/3");
  table(slides[4], rowsByCondition(), { left: 90, top: 150, width: 560, height: 250 }, 15);
  bars(slides[4], [
    { label: "plain 迎合率", value: 0.147, text: "14.7%", color: BLUE },
    { label: "opinion_only 迎合率", value: 0.325, text: "32.5%", color: RED },
    { label: "prefix+opinion 迎合率", value: 0.194, text: "19.4%", color: RED },
    { label: "plain 准确率", value: 0.575, text: "57.5%", color: GREEN },
  ], { left: 725, top: 160, width: 420, height: 190 }, 0.65);
  addText(slides[4], "解释：错误用户观点仍然会拉低事实性回答，但 1.5B 的 opinion-only 迎合率为 32.5%，显著低于 0.5B 全量实验的 69.2%。", { left: 120, top: 455, width: 1020, height: 72 }, { fontSize: 23, bold: true, color: PURPLE, alignment: "center" });

  clearBody(slides[5]);
  title(slides[5], "Logit Lens 不再显示后段层峰值，而是弱的早层观点差异", "Locate 2/3");
  table(slides[5], [["condition", "layer", "p_answer", "p_opinion", "gap"], ...lensRows], { left: 70, top: 145, width: 720, height: 260 }, 13);
  kpi(slides[5], "最大 gap", "0.028", "prefix+opinion layer 8", 860, 160, 250);
  kpi(slides[5], "主要层位", "3/5/8", "早层信号较弱", 860, 330, 250);
  addText(slides[5], "意义：1.5B 的错误观点 logit 优势很小，不能沿用 0.5B 报告里“20-22 层最强”的结论；规模变大后，表征模式发生了明显变化。", { left: 95, top: 470, width: 1030, height: 70 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  clearBody(slides[6]);
  title(slides[6], "Activation Patching 仍指向中后段层，但作用是增强正确 margin", "Locate 3/3");
  table(slides[6], [["layer", "base margin", "patched margin", "delta"], ...patchRows], { left: 90, top: 145, width: 700, height: 270 }, 14);
  bars(slides[6], [
    { label: "layer 26", value: 1.693, text: "+1.693", color: PURPLE },
    { label: "layer 23", value: 1.576, text: "+1.576", color: PURPLE },
    { label: "layer 22", value: 1.565, text: "+1.565", color: PURPLE },
    { label: "layer 24", value: 1.516, text: "+1.516", color: PURPLE },
  ], { left: 835, top: 170, width: 330, height: 155 }, 1.75);
  addText(slides[6], "与 0.5B 不同，1.5B 的 base margin 已经为正：patching 更准确地说是在增强正确答案相对错误观点的优势，而不是从负 margin 中救回答案。", { left: 110, top: 460, width: 1010, height: 76 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  clearBody(slides[7]);
  title(slides[7], "向量 steering 有方向性，但在 1.5B 上几乎不能消除迎合", "Steer");
  kpi(slides[7], "Sweep 规模", "393176", "4 层 × 7 alpha × 全量样本", 80, 150, 300);
  kpi(slides[7], "最佳 setting", "L24 α=4", "steering 中最高消除率", 425, 150, 300);
  kpi(slides[7], "消除率", "0.9%", "明显低于 prompt mitigation", 770, 150, 300);
  table(slides[7], [
    ["setting", "accuracy", "sycophancy", "elimination"],
    ["L24 alpha=4", "47.1%", "32.4%", "0.9%"],
    ["L24 alpha=-4", "47.0%", "32.5%", "0.9%"],
    ["L25 alpha=4", "47.1%", "32.4%", "0.8%"],
    ["L26 alpha=4", "47.1%", "32.5%", "0.7%"],
  ], { left: 145, top: 360, width: 990, height: 210 }, 14);
  addText(slides[7], "解释：简单均值方向向量在 1.5B 上只带来极小变化；这说明模型规模提升后，迎合行为不是一个容易被单向量线性抹去的模式。", { left: 115, top: 590, width: 1040, height: 44 }, { fontSize: 19, bold: true, color: PURPLE, alignment: "center" });

  clearBody(slides[8]);
  title(slides[8], "明确反迎合提示语仍是最有效的工程改进", "Improve");
  table(slides[8], mitigationRows, { left: 70, top: 140, width: 760, height: 300 }, 13);
  panel(slides[8], { left: 880, top: 150, width: 270, height: 140 }, LILAC);
  addText(slides[8], "最佳方法", { left: 910, top: 172, width: 210, height: 28 }, { fontSize: 18, bold: true, alignment: "center" });
  addText(slides[8], "anti_sycophancy", { left: 900, top: 212, width: 230, height: 30 }, { fontSize: 24, bold: true, color: PURPLE, alignment: "center" });
  addText(slides[8], "不要仅因用户观点而同意", { left: 910, top: 253, width: 210, height: 24 }, { fontSize: 13, color: MUTED, alignment: "center" });
  panel(slides[8], { left: 880, top: 325, width: 270, height: 140 }, LILAC);
  addText(slides[8], "迎合率下降", { left: 910, top: 348, width: 210, height: 28 }, { fontSize: 18, bold: true, alignment: "center" });
  addText(slides[8], "32.5% → 27.4%", { left: 895, top: 386, width: 240, height: 34 }, { fontSize: 27, bold: true, color: PURPLE, alignment: "center" });
  addText(slides[8], "准确率提升到 48.1%", { left: 910, top: 432, width: 210, height: 24 }, { fontSize: 13, color: MUTED, alignment: "center" });
  addText(slides[8], "结论：prompt 约束的效果仍明显强于 activation steering；更可靠的路线是机制定位帮助解释问题，工程上用明确的事实优先约束抑制迎合。", { left: 95, top: 515, width: 1050, height: 66 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  clearBody(slides[9]);
  title(slides[9], "1.5B 对照让机制结论更细：不是所有规模都复现同一层级形态", "复现价值");
  const claims = [
    ["行为更稳", "准确率从 0.5B 的低水平显著提升，opinion-only 迎合率从 69.2% 降到 32.5%。"],
    ["Logit Lens 改变", "最大 opinion gap 只有 0.028，且集中在 layer 3/5/8，不能说后段层峰值。"],
    ["Patching 保留因果性", "layer 26/23/22 仍能显著增强 correct-minus-opinion margin。"],
    ["改进策略更清楚", "prompt-level mitigation 仍然强于简单 steering，是本轮最实用的方案。"],
  ];
  for (const [i, [h, b]] of claims.entries()) {
    panel(slides[9], { left: 105 + (i % 2) * 545, top: 155 + Math.floor(i / 2) * 170, width: 485, height: 128 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slides[9], h, { left: 130 + (i % 2) * 545, top: 178 + Math.floor(i / 2) * 170, width: 430, height: 28 }, { fontSize: 22, bold: true, color: PURPLE });
    addText(slides[9], b, { left: 130 + (i % 2) * 545, top: 220 + Math.floor(i / 2) * 170, width: 425, height: 56 }, { fontSize: 17, color: DARK });
  }

  clearBody(slides[10]);
  title(slides[10], "结论：1.5B 降低了迎合，但 prompt 约束仍最稳", "总结");
  const takeaways = [
    "现象层面：opinion-only 仍把迎合率从 14.7% 提高到 32.5%。",
    "定位层面：Logit Lens 信号弱且偏早层；patching 在 layer 26/23/22 最强。",
    "干预层面：steering 最佳消除率只有 0.9%，远弱于 prompt mitigation。",
    "改进层面：anti_sycophancy 把迎合率降到 27.4%，并把准确率提升到 48.1%。",
  ];
  for (const [i, txt] of takeaways.entries()) {
    panel(slides[10], { left: 145, top: 145 + i * 98, width: 990, height: 66 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slides[10], txt, { left: 180, top: 163 + i * 98, width: 920, height: 32 }, { fontSize: 23, bold: true, color: i === 3 ? PURPLE : DARK });
  }
  addText(slides[10], "后续拓展：按 subject 学习 steering vectors，比较 ablation 与 SAE 特征级干预，并加入 3B/7B 模型做规模曲线。", { left: 155, top: 560, width: 970, height: 46 }, { fontSize: 21, bold: true, color: PURPLE, alignment: "center" });

  for (const [index, slide] of slides.entries()) {
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(QA_DIR, `slide-${index + 1}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `slide-${index + 1}.layout.json`), await layout.text());
  }
  const montage = await presentation.export({ format: "png", montage: true, scale: 1 });
  await writeBlob(path.join(QA_DIR, "contact-sheet.png"), montage);
  await writeBlob(CONTACT, montage);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT);
  console.log(`wrote ${OUT}`);
  console.log(`qa ${QA_DIR}`);
  console.log(`source ${BASE}`);
  process.exit(0);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

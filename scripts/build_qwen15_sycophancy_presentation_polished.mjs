import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const ROOT = process.env.MI_PROJECT_ROOT || "D:\\MI_project";
const TEMPLATE = path.join(ROOT, "inputs", "template_zky_purple.pptx");
const OUT = path.join(ROOT, "outputs", "LLM_sycophancy_locate_steer_improve_presentation_qwen15_polished.pptx");
const QA_DIR = path.join(ROOT, "outputs", "LLM_sycophancy_locate_steer_improve_presentation_qwen15_polished");
const CONTACT = path.join(ROOT, "outputs", "final-contact-sheet-qwen15-polished.png");
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

function clearBody(slide) {
  rect(slide, { left: 44, top: 112, width: 1192, height: 560 }, "white");
}

function title(slide, text, kicker = "") {
  rect(slide, { left: 34, top: 24, width: 1190, height: 80 }, "white");
  if (kicker) addText(slide, kicker, { left: 70, top: 30, width: 260, height: 24 }, { fontSize: 14, bold: true, color: PURPLE });
  addText(slide, text, { left: 70, top: 54, width: 1080, height: 42 }, { fontSize: 30, bold: true, color: DARK });
  rect(slide, { left: 70, top: 104, width: 180, height: 4 }, PURPLE);
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

function cardGrid(slide, cards, startY = 150) {
  for (const [i, card] of cards.entries()) {
    const x = 105 + (i % 2) * 545;
    const y = startY + Math.floor(i / 2) * 170;
    panel(slide, { left: x, top: y, width: 485, height: 128 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slide, card[0], { left: x + 25, top: y + 22, width: 430, height: 30 }, { fontSize: 22, bold: true, color: PURPLE });
    addText(slide, card[1], { left: x + 25, top: y + 64, width: 425, height: 54 }, { fontSize: 17, color: DARK });
  }
}

function flow(slide, steps, y = 205) {
  const w = 205;
  const gap = 26;
  const start = 80;
  for (const [i, step] of steps.entries()) {
    const x = start + i * (w + gap);
    panel(slide, { left: x, top: y, width: w, height: 210 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slide, String(i + 1), { left: x + 72, top: y + 18, width: 60, height: 42 }, { fontSize: 30, bold: true, color: PURPLE, alignment: "center" });
    addText(slide, step[0], { left: x + 18, top: y + 78, width: w - 36, height: 34 }, { fontSize: 19, bold: true, alignment: "center" });
    addText(slide, step[1], { left: x + 18, top: y + 128, width: w - 36, height: 58 }, { fontSize: 14, color: MUTED, alignment: "center" });
  }
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const behaviorRows = [
  ["条件", "样本数", "准确率", "迎合率"],
  ["plain", "14042", "57.5%", "14.7%"],
  ["opinion_only", "14042", "47.1%", "32.5%"],
  ["prefix_and_opinion", "14042", "54.6%", "19.4%"],
];

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
  const sourceMap = [2, 5, 36, 36, 31, 36, 36, 36, 36, 36, 36, 36, 12, 16, 54];
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
  setText(shape(slides[0], 1), "必要性、机制原理与全量结果 | Qwen2.5-1.5B | MMLU raw 14042 条", { color: "white", fontSize: 19 });
  setText(shape(slides[0], 2), "智能科学与技术学院", { color: "white", fontSize: 18 });
  setText(shape(slides[0], 3), "MI Project", { color: "white", fontSize: 18 });
  setText(shape(slides[0], 4), "2026.07", { color: "white", fontSize: 18 });
  note(slides[0], "润色版加入大模型可解释性必要性、patch/steer/improve 原理和更完整的归纳总结。");

  clearBody(slides[1]);
  title(slides[1], "这份报告从“为什么研究”走到“如何干预”", "目录");
  const contents = [
    ["1", "为什么大模型需要机制可解释性研究"],
    ["2", "Locate-Steer-Improve 的实验链路与数据范围"],
    ["3", "patch / steer / improve 三类方法的原理"],
    ["4", "Qwen2.5-1.5B 全量实验结果"],
    ["5", "从机制证据归纳可执行结论"],
  ];
  for (const [i, [idx, txt]] of contents.entries()) {
    panel(slides[1], { left: 145, top: 132 + i * 90, width: 990, height: 64 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slides[1], idx, { left: 185, top: 142 + i * 90, width: 70, height: 42 }, { fontSize: 28, bold: true, color: PURPLE, alignment: "center" });
    addText(slides[1], txt, { left: 285, top: 151 + i * 90, width: 760, height: 34 }, { fontSize: 23, bold: true });
  }

  clearBody(slides[2]);
  title(slides[2], "大模型越强，越需要知道错误从哪里进入推理", "研究必要性");
  cardGrid(slides[2], [
    ["行为结果不足够", "只看准确率和迎合率，只能知道模型错了；不知道错误观点在内部何时压过事实答案。"],
    ["规模提升不等于可靠", "1.5B 的迎合率低于 0.5B，但 opinion-only 仍从 14.7% 提高到 32.5%。"],
    ["安全问题需要因果证据", "可解释性把相关现象推进到层级、表示和干预证据，避免只靠经验提示词。"],
    ["改进需要可操作入口", "定位关键层后，才能比较 patch、steer、prompt mitigation 哪条路线真正有效。"],
  ]);

  clearBody(slides[3]);
  title(slides[3], "同一全量数据上，1.5B 提供更强的规模对照", "实验范围");
  kpi(slides[3], "MMLU raw", "14042", "全量样本，不是 3000 条", 78, 150);
  kpi(slides[3], "学科覆盖", "57", "subjects", 360, 150);
  kpi(slides[3], "模型", "1.5B", "Qwen2.5-Instruct", 642, 150);
  kpi(slides[3], "框架", "CUDA", "PyTorch + Transformers", 924, 150);
  table(slides[3], [
    ["环节", "产物", "全量规模"],
    ["Behavior", "locate_behavior.csv", "14042×3 = 42126"],
    ["Logit Lens", "locate_layer_logit_lens.csv", "14042×3×28 = 1179528"],
    ["Steering", "steer_sweep.csv", "14042×4×7 = 393176"],
    ["Patching", "activation_patching_summary.csv", "14042×28 = 393176"],
    ["Improve", "improve_prompt_mitigation.csv", "14042×5 = 70210"],
  ], { left: 130, top: 330, width: 1020, height: 250 }, 14);

  clearBody(slides[4]);
  title(slides[4], "Locate-Steer-Improve 把解释和改进连成闭环", "方法链路");
  flow(slides[4], [
    ["Behavior", "先确认错误观点是否改变输出"],
    ["Logit Lens", "逐层观察答案与观点的竞争"],
    ["Patch", "替换内部表示，验证因果层"],
    ["Steer", "注入方向向量，尝试推离迎合"],
    ["Improve", "用约束提示语做工程缓解"],
  ], 190);
  addText(slides[4], "核心逻辑：先证明现象，再定位内部载体，最后比较可操作干预是否真正降低迎合。", { left: 145, top: 515, width: 990, height: 44 }, { fontSize: 22, bold: true, color: PURPLE, alignment: "center" });

  clearBody(slides[5]);
  title(slides[5], "Patching 的原理是做一次内部表示的反事实替换", "原理 1/3");
  flow(slides[5], [
    ["两种状态", "plain prompt 与 opinion prompt 形成成对样本"],
    ["选择层", "取同一层最后 token 的 residual 表示"],
    ["替换表示", "用 plain 表示替换 opinion 表示"],
    ["观察 margin", "检查 correct-minus-opinion 是否改变"],
    ["判断因果", "delta 越大，该层越像关键载体"],
  ], 185);
  addText(slides[5], "本实验中，1.5B 的 base margin 已经为正；patching 更像是在增强正确答案优势，而不是把答案从负 margin 中救回。", { left: 125, top: 515, width: 1030, height: 56 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  clearBody(slides[6]);
  title(slides[6], "Steering 的原理是把“非迎合方向”注入推理过程", "原理 2/3");
  flow(slides[6], [
    ["构造方向", "v = mean(hidden_plain - hidden_opinion)"],
    ["选择层位", "在候选 decoder layer 上注入向量"],
    ["调节强度", "用 alpha 控制方向和幅度"],
    ["重新回答", "在 opinion prompt 下再次推理"],
    ["评估消除", "统计 baseline 迎合样本是否被纠正"],
  ], 185);
  addText(slides[6], "它是机制一致的干预，但单一均值向量过于粗糙：在 1.5B 上最佳消除率只有 0.9%。", { left: 135, top: 515, width: 1010, height: 56 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  clearBody(slides[7]);
  title(slides[7], "Improve 的原理是把机制洞察转化为显式推理约束", "原理 3/3");
  cardGrid(slides[7], [
    ["truth_priority", "要求优先独立判断事实，再处理用户观点。"],
    ["anti_sycophancy", "明确要求不要仅因用户表达了观点就同意。"],
    ["verify_then_answer", "先验证选项依据，再给出最终答案。"],
    ["counter_opinion_check", "刻意检查用户观点是否可能是错误选项。"],
  ]);
  addText(slides[7], "Prompt mitigation 不直接改内部激活，但它是最容易部署的工程约束；本轮最优方法是 anti_sycophancy。", { left: 120, top: 515, width: 1040, height: 56 }, { fontSize: 22, bold: true, color: PURPLE, alignment: "center" });

  clearBody(slides[8]);
  title(slides[8], "1.5B 仍会迎合，但幅度明显低于 0.5B", "结果 1/5");
  table(slides[8], behaviorRows, { left: 90, top: 150, width: 560, height: 250 }, 15);
  bars(slides[8], [
    { label: "plain 迎合率", value: 0.147, text: "14.7%", color: BLUE },
    { label: "opinion_only 迎合率", value: 0.325, text: "32.5%", color: RED },
    { label: "prefix+opinion 迎合率", value: 0.194, text: "19.4%", color: RED },
    { label: "plain 准确率", value: 0.575, text: "57.5%", color: GREEN },
  ], { left: 725, top: 160, width: 420, height: 190 }, 0.65);
  addText(slides[8], "解释：规模提升显著降低迎合，但错误用户观点仍会改变事实性回答。", { left: 120, top: 455, width: 1020, height: 72 }, { fontSize: 23, bold: true, color: PURPLE, alignment: "center" });

  clearBody(slides[9]);
  title(slides[9], "Logit Lens 显示 1.5B 的观点优势弱且偏早层", "结果 2/5");
  table(slides[9], [["condition", "layer", "p_answer", "p_opinion", "gap"], ...lensRows], { left: 70, top: 145, width: 720, height: 260 }, 13);
  kpi(slides[9], "最大 gap", "0.028", "prefix+opinion layer 8", 860, 160, 250);
  kpi(slides[9], "主要层位", "3/5/8", "早层信号较弱", 860, 330, 250);
  addText(slides[9], "意义：不能沿用 0.5B 中“20-22 层最强”的结论；规模变化会改变内部表征形态。", { left: 95, top: 470, width: 1030, height: 70 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  clearBody(slides[10]);
  title(slides[10], "Patching 指向 layer 26/23/22 对答案 margin 最关键", "结果 3/5");
  table(slides[10], [["layer", "base margin", "patched margin", "delta"], ...patchRows], { left: 90, top: 145, width: 700, height: 270 }, 14);
  bars(slides[10], [
    { label: "layer 26", value: 1.693, text: "+1.693", color: PURPLE },
    { label: "layer 23", value: 1.576, text: "+1.576", color: PURPLE },
    { label: "layer 22", value: 1.565, text: "+1.565", color: PURPLE },
    { label: "layer 24", value: 1.516, text: "+1.516", color: PURPLE },
  ], { left: 835, top: 170, width: 330, height: 155 }, 1.75);
  addText(slides[10], "结论：patching 给出因果证据；但在 1.5B 上它主要增强已为正的正确答案 margin。", { left: 110, top: 460, width: 1010, height: 76 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  clearBody(slides[11]);
  title(slides[11], "Steering 的方向正确，但工程效果很弱", "结果 4/5");
  kpi(slides[11], "Sweep 规模", "393176", "4 层 × 7 alpha × 全量样本", 80, 150, 300);
  kpi(slides[11], "最佳 setting", "L24 α=4", "steering 中最高消除率", 425, 150, 300);
  kpi(slides[11], "消除率", "0.9%", "明显低于 prompt mitigation", 770, 150, 300);
  table(slides[11], [
    ["setting", "accuracy", "sycophancy", "elimination"],
    ["L24 alpha=4", "47.1%", "32.4%", "0.9%"],
    ["L24 alpha=-4", "47.0%", "32.5%", "0.9%"],
    ["L25 alpha=4", "47.1%", "32.4%", "0.8%"],
    ["L26 alpha=4", "47.1%", "32.5%", "0.7%"],
  ], { left: 145, top: 360, width: 990, height: 210 }, 14);
  addText(slides[11], "解释：迎合行为不是一个容易被单一线性向量抹去的模式，需要更精细的分层或特征级干预。", { left: 115, top: 590, width: 1040, height: 44 }, { fontSize: 19, bold: true, color: PURPLE, alignment: "center" });

  clearBody(slides[12]);
  title(slides[12], "Prompt-level mitigation 仍是最稳的可部署改进", "结果 5/5");
  table(slides[12], mitigationRows, { left: 70, top: 140, width: 760, height: 300 }, 13);
  kpi(slides[12], "最佳方法", "anti_sycophancy", "不要仅因用户观点而同意", 880, 150, 270);
  kpi(slides[12], "迎合率下降", "32.5% → 27.4%", "准确率提升到 48.1%", 880, 325, 270);
  addText(slides[12], "结论：机制定位解释了问题来源；工程上，明确事实优先约束比简单 activation steering 更可靠。", { left: 95, top: 515, width: 1050, height: 66 }, { fontSize: 22, bold: true, color: DARK, alignment: "center" });

  clearBody(slides[13]);
  title(slides[13], "从结果看，机制可解释性提供三层价值", "归纳");
  cardGrid(slides[13], [
    ["发现问题", "行为实验确认错误观点能改变输出，而不是普通答题波动。"],
    ["解释问题", "Logit Lens 与 Patching 把现象落到层级信号和因果表示。"],
    ["比较干预", "Steering 与 prompt mitigation 的差距说明可解释不等于直接可控。"],
    ["指导扩展", "下一步应做 subject-specific vectors、ablation 和 SAE 特征级干预。"],
  ]);

  clearBody(slides[14]);
  title(slides[14], "总结：解释性研究把“模型会错”推进为“知道怎样改”", "结论");
  const takeaways = [
    "必要性：大模型更强但仍会迎合，安全评估不能只看最终准确率。",
    "定位结论：1.5B 的 Logit Lens 信号弱且偏早层，Patching 在 layer 26/23/22 最强。",
    "干预结论：简单 steering 最佳消除率只有 0.9%，目前不足以作为主方案。",
    "改进结论：anti_sycophancy 把迎合率降到 27.4%，并把准确率提升到 48.1%。",
  ];
  for (const [i, txt] of takeaways.entries()) {
    panel(slides[14], { left: 145, top: 145 + i * 98, width: 990, height: 66 }, i % 2 ? "#FFFFFF" : LILAC);
    addText(slides[14], txt, { left: 180, top: 163 + i * 98, width: 920, height: 32 }, { fontSize: 23, bold: true, color: i === 3 ? PURPLE : DARK });
  }
  addText(slides[14], "最终判断：机制可解释性不是装饰性解释，而是连接风险定位、因果验证与工程缓解的方法。", { left: 155, top: 560, width: 970, height: 46 }, { fontSize: 21, bold: true, color: PURPLE, alignment: "center" });

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
  process.exit(0);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

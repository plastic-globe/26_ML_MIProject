import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const [pptx, slideNums] = process.argv.slice(2);
const presentation = await PresentationFile.importPptx(await FileBlob.load(pptx));
for (const n of slideNums.split(",")) {
  const slide = presentation.slides.items[Number(n) - 1];
  console.log(`\nSLIDE ${n}`);
  slide.shapes.items.forEach((shape, idx) => {
    const text = shape.text?.toString?.().trim();
    if (text) console.log(idx, shape.id, JSON.stringify(text));
  });
  slide.tables.items?.forEach((table, idx) => {
    console.log("table", idx, table.id);
  });
}

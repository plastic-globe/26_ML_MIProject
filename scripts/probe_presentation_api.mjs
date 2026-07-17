import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const pptx = process.argv[2];
const presentation = await PresentationFile.importPptx(await FileBlob.load(pptx));
const slide = presentation.slides.items[0];
console.log("slides methods", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides)).sort());
console.log("slide methods", Object.getOwnPropertyNames(Object.getPrototypeOf(slide)).sort());
console.log("slide keys", Object.keys(slide).sort());
console.log("shapes methods", Object.getOwnPropertyNames(Object.getPrototypeOf(slide.shapes)).sort());
console.log("shape count", slide.shapes.items?.length);
const shape = slide.shapes.items?.find((item) => item.text !== undefined) || slide.shapes.items?.[0];
if (shape) {
  console.log("shape methods", Object.getOwnPropertyNames(Object.getPrototypeOf(shape)).sort());
  console.log("shape keys", Object.keys(shape).sort());
  console.log("shape text type", typeof shape.text, String(shape.text).slice(0, 80));
  if (shape.text) {
    console.log("text methods", Object.getOwnPropertyNames(Object.getPrototypeOf(shape.text)).sort());
    console.log("text keys", Object.keys(shape.text).sort());
    console.log("text value", shape.text.toString?.());
  }
}

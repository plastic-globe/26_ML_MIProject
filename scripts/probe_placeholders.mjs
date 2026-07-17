import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const pptx = process.argv[2];
const presentation = await PresentationFile.importPptx(await FileBlob.load(pptx));
for (const [i, slide] of presentation.slides.items.entries()) {
  const placeholders = slide.getInheritedPlaceholderShapes?.() ?? [];
  if (placeholders.length) {
    console.log(`SLIDE ${i + 1}`, placeholders.length);
    placeholders.forEach((ph, idx) => {
      console.log(idx, ph.id, ph.placeholderType, JSON.stringify(ph.text?.toString?.() ?? ""));
    });
  }
}

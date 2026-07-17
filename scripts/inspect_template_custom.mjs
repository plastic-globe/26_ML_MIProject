import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  const [pptxPath, outDir] = process.argv.slice(2);
  if (!pptxPath || !outDir) {
    throw new Error("Usage: node inspect_template_custom.mjs <pptx> <outDir>");
  }
  await fs.mkdir(outDir, { recursive: true });
  const presentation = await PresentationFile.importPptx(await FileBlob.load(pptxPath));
  const inspect = await presentation.inspect({
    kind: "slide,textbox,shape,image,table,chart,notes,layout",
    maxChars: 50000,
  });
  await fs.writeFile(path.join(outDir, "template-inspect.ndjson"), inspect.ndjson, "utf8");

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(outDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(outDir, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  await writeBlob(path.join(outDir, "template-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
  await fs.writeFile(
    path.join(outDir, "template-manifest.json"),
    JSON.stringify({ slideCount: presentation.slides.items.length, source: pptxPath }, null, 2),
    "utf8",
  );
  console.log(`slides=${presentation.slides.items.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

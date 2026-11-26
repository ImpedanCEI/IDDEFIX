"""Prepare the published notebook copies and their attachment images for Sphinx."""

import base64
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
source_dir = repo_root / "examples"
dest_dir = Path(__file__).resolve().parent / "examples"
dest_dir.mkdir(parents=True, exist_ok=True)

# Use the example toctree as the list of notebooks to publish
notebooks = [
    Path(line.strip()).relative_to("examples").with_suffix(".ipynb")
    for line in (dest_dir.parent / "index.md").read_text(encoding="utf-8").splitlines()
    if line.startswith("examples/")
]

for name in notebooks:
    notebook = json.loads((source_dir / name).read_text(encoding="utf-8"))

    # Give each attachment its own path, including repeated filenames in different cells
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "markdown" or not cell.get("attachments"):
            continue
        source = "".join(cell["source"])
        for filename, bundle in cell["attachments"].items():
            mime = next((key for key in bundle if key.startswith("image/")), None)
            if mime is None:
                raise ValueError(f"Unsupported notebook attachment: {name}: {filename}")
            relative_path = (
                Path("_attachments")
                / Path(name).stem
                / f"{index}_{Path(filename).name}"
            )
            destination = dest_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = bundle[mime]
            payload = "".join(payload) if isinstance(payload, list) else payload
            image = (
                payload.encode("utf-8")
                if mime == "image/svg+xml"
                else base64.b64decode(payload)
            )
            destination.write_bytes(image)
            source = source.replace(f"attachment:{filename}", relative_path.as_posix())
        cell["source"] = source.splitlines(keepends=True)
        del cell["attachments"]

    # Keep the source notebooks and their saved code outputs unchanged
    (dest_dir / name).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"Prepared {name} in {dest_dir}")

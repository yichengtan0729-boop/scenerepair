from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


root = Path(__file__).resolve().parents[1] / "examples" / "toy"
root.mkdir(parents=True, exist_ok=True)
rows = []
for example_idx in range(2):
    image_paths = []
    for view in range(1, 4):
        image = Image.new("RGB", (256, 192), "white")
        draw = ImageDraw.Draw(image)
        offset = 20 * view
        draw.rectangle((20 + offset, 70, 80 + offset, 130), outline="black", width=4)
        draw.ellipse((150 - offset, 70, 200 - offset, 120), outline="black", width=4)
        draw.text((10, 10), f"Example {example_idx} View {view}", fill="black")
        path = root / f"example{example_idx}_view{view}.png"
        image.save(path)
        image_paths.append(path.name)
    rows.append(
        {
            "id": f"toy_{example_idx}",
            "task": "single_view_spatial_editing",
            "question": "If the rectangle moves to the right of the circle, what is its new relation to the circle?",
            "choices": ["A. left of", "B. right of", "C. above", "D. behind"],
            "answer": "B",
            "images": image_paths,
        }
    )
with (root / "toy.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\n")
print(root / "toy.jsonl")

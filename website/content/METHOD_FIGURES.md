# Original paper figures

The method section displays only the original Figure 2 (framework, PDF page 3)
from `assets/paper/OpenNWM.pdf`. Figure 3 (three-stage training, PDF page 7) is
retained as a source asset. The user requested a return to the static main figure.

Both PNGs are 432 DPI composite renders of the original PDF artwork, including
all raster, vector, and text layers. Their contents are not redrawn, relabeled,
recolored, or reconstructed. Exact crop bounds, dimensions, and SHA256 hashes
are recorded in `assets/paper/method-figures.json`.

The page preserves the complete framework figure and its aspect ratio at every
viewport width. Clicking the figure opens the unchanged full-resolution original.
There are no method animation overlays, stage controls, or playback controls.
A concise bilingual explanation summarizes latent-action pretraining and alignment
with physical controls for prediction and navigation.

Reproduce from the website directory using Python with PyMuPDF installed:

```python
import json
from pathlib import Path
import fitz

root = Path("assets/paper")
metadata = json.loads((root / "method-figures.json").read_text())
document = fitz.open(root / metadata["source"])
for figure in metadata["figures"]:
    image = document[figure["page"] - 1].get_pixmap(
        matrix=fitz.Matrix(figure["scale"], figure["scale"]),
        clip=fitz.Rect(figure["clipPoints"]),
        alpha=False,
        colorspace=fitz.csRGB,
    )
    image.save(root / figure["file"])
```

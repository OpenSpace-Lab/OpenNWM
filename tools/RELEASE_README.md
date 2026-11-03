# Release README generation

The root `README.md` owns the shared setup, data, models, evaluation, training,
and citation text for GitHub, Hugging Face, and ModelScope. Generate each card
locally using its existing README as the YAML metadata source:

```bash
conda run --no-capture-output -n nwm python tools/render_release_readme.py \
  --readme README.md --metadata-from /path/to/old-card.md \
  --appendix /path/to/resource-appendix.md --output /path/to/new-card.md
```

Omit `--appendix` if no resource-specific explanation is needed. Appendices
should include their own heading and are copied unchanged; they may use links
relative to the model or dataset repository. The generator only writes the
requested local file. Publication is a separate step.

Existing YAML metadata is preserved. An `en`/`zh` language field is reduced to
`en` when the body and appendix contain no Chinese characters; other language
values are left unchanged. The canonical body changes only to resolve relative
links:

- `website/` assets use the public `OpenSpace-Lab/OpenNWM-site` repository's
  `raw.githubusercontent.com` URLs, with the `website/` prefix removed.
- Other images use `OpenSpace-Lab/OpenNWM` raw URLs.
- Code and guide links use `OpenSpace-Lab/OpenNWM/blob/main/` URLs.
- Absolute URLs, page anchors, code examples, and BibTeX remain unchanged.

For local automation, import `render(readme, metadata_from="", appendix=None)`
from `tools.render_release_readme`; all arguments and the result are strings.

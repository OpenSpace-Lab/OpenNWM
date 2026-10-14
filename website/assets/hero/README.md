# Dataset video wall, Moon and Mars

`journey.mp4` is one silent 1920 × 1080, 24 fps H.264 film. It opens
with a three-second, fixed overview of the complete 15 × 15 dataset video
wall, then travels directly through two lunar and two Martian environments.
The wall contains **225 distinct trajectories from 15 NavAnywhere sources**
and has no text burned into its tiles. All CASIA-Nav `office_*` trajectories
are excluded; four new outdoor trajectories replace them.

These are source-data showcases, **not OpenNWM predictions**. The video
wall comes from NavAnywhere. Moon views are **LuSNAR simulation**. Mars views
use the **SynMars-TW** dataset: contiguous numbered frames from
the `forward_9` route's left camera in the testing split. SynMars-TW depicts
a Blender-generated Mars-like environment; it is not actual Mars mission
footage. No generated entities are composited into any scene.

The public `../../content/cinematic.json` records scene/planet timings,
bilingual labels, dataset provenance and exact selected frame filenames.
Its `journey.src` and `mosaic.src` both identify the same complete film;
`mosaic.startSeconds` is zero and identifies the start of the wall in that film.
The website draws scene labels and the interactive planet navigation above
the film. The wall is intentionally unlabeled.

## Planet textures

- `earth-texture.webp`: cropped [NASA Blue Marble surface map](https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.png), showing blue oceans and green continents.
- `moon-texture.webp`: LuSNAR lunar source frame.
- `mars-texture.webp`: SynMars-TW surface source frame.

The three textures are retained legacy presentation assets; they are not
model outputs. SynMars-TW source: [CVIR-Lab/SynMars, SynMars-TW branch](https://github.com/CVIR-Lab/SynMars/tree/SynMars-TW).

## Editing and reproduction

Source frames are center-cropped to 16:9 and resized. Within each journey
segment the closest source frame is shown without generated intermediate
frames. Brief crossfades connect planetary scenes. The wall has no zoom or
pan: all 225 tiles are visible from the first frame, and their clips keep
moving during all three seconds. Playback speeds are editorial choices,
not original camera timing or ground-truth actions. Short wall clips play
forward and backward, with frame blending to soften loops. The whole film
loops back to the dataset wall after Mars.

The current 606-frame film lasts 25.25 seconds. Chapter boundaries are:

| Chapter | Start (seconds) | End (seconds) |
| --- | ---: | ---: |
| Dataset video wall | 0 | 3 |
| Moon | 3 | 13.44 |
| Mars | 13.44 | 25.25 |

The October 10 refinement reuses the reviewed source frames and tile caches.
It removes the standalone Earth scenes and their transition into the Moon.
Both posters show the full video wall. Scene timings and the content-derived
media cache version are updated together in the public manifest.

With the datasets available locally, run from the source repository:

```bash
conda run --no-capture-output -n base python -u tools/build_website_cinematic.py sample
conda run --no-capture-output -n base python -u tools/build_website_cinematic.py prepare
conda run --no-capture-output -n base python -u tools/build_website_cinematic.py render
```

`prepare` reuses reviewed source-frame caches. `render` writes the merged
film to NAS, validates its format and frame count, then updates the exported
film, posters and manifest. The public manifest can reconstruct the selected
frames if the local `curation.json` is absent. The runtime uses OpenCV,
NumPy, Pillow and FFmpeg; encoding runs on CPU. Local defaults store caches
and review sheets on NAS. Other machines can set `OPENNWM_NAV_ROOT`,
`OPENNWM_LUSNAR_ROOT`, `OPENNWM_SYNMARS_ROOT` and `OPENNWM_MEDIA_WORK_DIR`.

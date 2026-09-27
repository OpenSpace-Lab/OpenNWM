# Paper content and provenance

`paper.json` is transcribed and summarized from the supplied 38-page manuscript,
`/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM.pdf`, inspected on 2026-09-27.
Page numbers refer to the PDF's one-based pages, which match its printed page
numbers. This file records source and interpretation details for maintainers;
the concise reader-facing text lives in `paper.json`.

## Source map

| Content | Manuscript source |
| --- | --- |
| Title, authors, review status, abstract | Page 1 |
| Contributions | Pages 1–2 |
| Latent action objectives and dynamics conditioning | Pages 4–6 |
| Three-stage world-model training | Figure 3, page 7 |
| Main direct prediction results | Table 1, page 7; interpretation on page 8 |
| Main navigation results | Table 2, page 7 |
| LAM reconstruction | Table 3, page 8 |
| Semantic consistency values | Figure 4, page 8 |
| Semantic consistency metric definition | Appendix F.3, page 24 |
| Time / latent conditioning ablation | Table 4, page 9 |
| LAM supervision ablation | Table 5, page 9 |
| NavAnywhere scene distribution and diversity | Figure 6, page 15 |
| NavAnywhere source descriptions | Pages 15–16 |
| NavAnywhere source statistics | Table 6, page 17 |
| Action-annotated statistics and dataset roles | Table 7 / Appendix B, page 18 |
| Additional held-out sets and temporal sampling | Appendix B, page 18 |
| Main architecture, training, inference and planning settings | Pages 19–20; Table 8, page 20 |
| TartanDrive navigation results | Table 10, page 22 |
| Unlabeled-data scaling | Figure 8, pages 22–23 |

## Metadata and citation

The manuscript names **Anonymous authors** and states **Paper under double-blind
review** and **Under review as a conference paper at ICLR 2027**. It does not state
real author names, affiliations, acceptance, a publication year, a DOI, or an
arXiv URL. Those fields must not be inferred from local directory names or PDF
creation timestamps. `citation.bibtex` is a provisional `@unpublished` entry
that preserves the anonymous review status and omits an unsupported year.

## Statistical and experimental interpretation

- NavAnywhere has **70,756 sequences**, **17,496,570 sampled RGB frames**, and
  **1,188.65 source-video hours** across **15 sources**, as reported by Table 6.
  The displayed source rows sum exactly to the sequence and frame totals. The
  separately rounded source-hour rows sum to 1,188.64 h; the published total is
  preserved. Source descriptions are qualitative, not measured per-source scene
  proportions.
- The five published scene percentages sum to **100.1%** because of rounding.
  Their source values are preserved without renormalizing the displayed labels.
  The figure also reports **17+ countries** and **97.3% real-world video**.
- Hours refer to source-video duration, while frame counts refer to sampled
  observations. Most videos use 4 Hz; EgoWalk uses approximately 40 Hz after
  sampling its 100 Hz source. A single frame-rate conversion cannot reconstruct
  the corpus's duration.
- Action-annotated datasets and the additional held-out sets are **excluded**
  from the NavAnywhere statistics. LAM grounding uses RECON, HuRoN/SACSoN, and
  SCAND. NWM post-training additionally uses TartanDrive. Go Stanford is held
  out. TartanDrive is therefore LAM OOD but NWM ID.
- LAM reconstruction ID averages cover RECON, SCAND, and HuRoN; its OOD result
  is Go Stanford. The semantic-consistency evaluation additionally tests
  TartanDrive as LAM OOD. NWM ID prediction averages cover RECON, SCAND, HuRoN,
  and TartanDrive.
- Table 8 reports an approximately **197M CDiT-B/2 backbone**. Tables 1–2 report
  a **280M benchmark model**. The JSON keeps these distinct; it does not invent
  a component-wise reconciliation.
- The LAM checkpoint used is at **60K steps**, under a **100K-step learning-rate
  schedule** with a **12.8K-step warmup**. These are different quantities.
- The main NWM pipeline uses **60K latent pretraining**, **3K adapter warmup**,
  and **100K joint post-training**. Ablation configurations differ and are
  recorded separately; they must not overwrite the main reproduction settings.
- Planning searches over physical **(Δx, Δy, Δψ)** actions, not over latent
  actions. Its reported configuration is DDPM250 and CEM with 80 candidates,
  5 elites, 1 update, 3 stochastic evaluations, 8 actions at 0.25 s intervals,
  and an LPIPS-Alex cost. The auxiliary LAM physical predictor is a training
  objective, not the online planner.
- Table 1's reported 9.1% LPIPS reduction, 13.7% DreamSim reduction, and
  0.342 dB PSNR improvement compare against the **strongest baseline for each
  metric**. They are rounded headline results, not a claim that each compares
  against NWM alone.
- OpenNWM leads OOD PSNR on **three of four** datasets. Office-Go2 is second,
  0.036 dB below NWM CDiT-XL + Ego4D. Do not describe every OOD metric as best.
- In Table 5, joint Pixel + Action leads five of six metrics; Pixel-only has
  the lowest RECON RPE. In Figure 4, Pixel + Action is not best on TartanDrive.
  Neither table supports a claim of universal superiority for every metric.
- Navigation `null` values mean **not reported**, never zero.
- Scaling contains four measured points (25%, 50%, 75%, 100%), all evaluated
  on the same 500 RECON windows at a 4 s horizon. A visualization should select
  actual points; interpolation must not be labeled as a measured experiment.

## Local media and demonstrations

Dataset media is described in `datasets.json`; prediction media is described
in `demos.json`. Those files own their assets and provenance. Dataset frames
illustrate the source material; they are not predictions. Precomputed model
rollouts are curated qualitative examples, not new measurements, aggregate
benchmark results, or verified copies of the paper's example figures.

The selected HuRoN and TartanDrive examples use extended 16 s predictions;
their 65-frame, 4 FPS files include an initial frame and last 16.25 s. The
selected Unitree Go2 example predicts 4 s; its 17-frame file lasts 4.25 s.
Use the media manifest's verified label for the local Unitree recording rather
than implying that an unverified individual trajectory is a specific paper
figure or a complete reproduction of the Office-Go2 evaluation set.

The media manifests and manuscript support different types of evidence. The
paper tables report the manuscript's experiments; local videos illustrate
existing artifact outputs. Earlier code-equivalence smoke tests do not validate
the paper's aggregate quality results and should not be presented as doing so.

## Content structure

`paper.json` uses the following top-level keys:

- `metadata`, `overview`, `contributions`, `citation`: display copy and metadata.
- `method`, `training`, `planning`: the method and main reproducibility settings.
- `datasets`: `navanywhere`, `action_annotated`, and `additional_ood`.
- `results`: metric directions and the prediction, navigation, reconstruction,
  ablation, semantic-consistency, and scaling tables.

Each content group carries a `source` field. Source IDs in
`datasets.navanywhere.sources` match `datasets.json` so the UI can join
statistics and media by `id`. The paper's In-house Collected source uses the
media ID `casia-nav`; HuRoN uses `sacson` in the annotated dataset rows.

PDF text extraction and temporary page renderings are outside the source
repository in
`/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts`.

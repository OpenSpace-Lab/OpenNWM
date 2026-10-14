# Restore the original main method figure

The user requested removing the animation and returning to the previous paper main figure. The method renderer and stylesheet are restored from published commit `7d9582e`: one unchanged Figure 2, complete aspect ratio, and a link to the full-resolution original. The original concise English/Chinese explanation is restored. The animation module, overlays, and controls are removed. Cache keys are updated to `main-figure-v8-20261010`.

Original image: `assets/paper/figure-2-framework.png`, 2352×720, SHA256 `06feb9ee8a10c096120d1d51ddab38d1561831df089ea2887754b912c8036392`.

JavaScript syntax and public-checkout diff checks passed. The restored renderer and stylesheet match the previous static implementation exactly. Earlier animation validation remains as historical evidence.

Local browser smoke checks passed 19/19, PID 673313, exit 0: one Figure 2, complete aspect ratio, original-image link opening, no animation container/controls/module request, and no JavaScript errors. English and Chinese at 1440/390 px were checked; screenshots were visually reviewed.

Evidence directory and working directory: `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/method_main_figure_20261010/browser`.

```bash
OPENNWM_CHECK_MODE=local OPENNWM_CHECK_URL=http://127.0.0.1:8765/ /file_system/vepfs/algorithm/dujun.nie/.npm-global/bin/node verify.cjs > local.log 2>&1
```

`local/report.json` and `local.log` record results. Log tail: `COMPLETE passed=true checks=19/19 exit=0`. Unrelated media are blocked for this targeted method check; no inference is requested.

Public commit: `1cfae647bd69ff34b494db141ab26bb3cc2b0df4`. [Pages run 38057216933](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38057216933) successfully deployed that exact SHA. The public checkout is clean.

Public browser check, same working directory, PID 681052:

```bash
OPENNWM_CHECK_MODE=public OPENNWM_CHECK_URL=https://b0b8k1ng.github.io/OpenNWM/ OPENNWM_DEPLOY_SHA=1cfae647bd69ff34b494db141ab26bb3cc2b0df4 /file_system/vepfs/algorithm/dujun.nie/.npm-global/bin/node verify.cjs > public.log 2>&1
```

The public page's original image loaded directly from GitHub Pages. The targeted check blocked unrelated media; other page modules and styles remained actual public requests. All **19/19 public checks passed**, PID 681052, exit **0**: desktop/mobile EN/ZH display, complete image ratio, opening the original-image link, absence of animation controls/module requests, and no JavaScript errors. The log ends `COMPLETE passed=true checks=19/19 exit=0`; screenshots and `public/report.json` preserve the result. Network transfer was repeated after language changes rebuilt the image; the process finished naturally before a proposed cutoff of the redundant popup request, and no process was stopped. Desktop and mobile screenshots were visually reviewed.

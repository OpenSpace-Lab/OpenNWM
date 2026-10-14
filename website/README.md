# OpenNWM 项目网站

原生 HTML、CSS、JavaScript 学术主页，无前端依赖或构建步骤。图片、视频、字体和论文均随网站保存；静态浏览及播放预设无需 GPU。自由绘制轨迹后的真实预测由可选的本机 GPU 服务提供。

## 当前页面（2026-10-10）

本轮改为固定全景视频墙停留 3 秒，随后仅播放月球与火星，进度由轨道上的小车表示。方法区恢复为论文架构主图，保留完整原图与一段概述，点击可查看全尺寸图片。Prediction 包含 9 个地球样本和 3 个月面微调样本，统一展示 Given path 与 Ground truth / OpenNWM 双视频滑块；月面使用灰色卡片、月球图标和“少量微调”标注。独立 Beyond Earth 区块已移除。

页面沿用炭黑、黄绿色与常规宽度 Arial 主题，顺序为方法 → Prediction → 导航可视化 → 预测与导航指标 → Demo → NavAnywhere 数据集，末尾保留论文、代码及引用。首屏提供 Paper、Code、Demo 和数据集入口。首屏及 Prediction 修订已部署为 `7d9582e`；根据最新要求，方法区已恢复为静态论文主图并部署为 `1cfae64`。本次恢复记录见 [content/MAIN_FIGURE_20261010_VALIDATION.md](content/MAIN_FIGURE_20261010_VALIDATION.md)。

- 方法区仅展示论文架构主图，保持原始内容与比例，可点击打开完整尺寸；动画与阶段控件已移除。
- Prediction 展示 12 个场景的真实条件轨迹及同步视频滑块，素材和溯源在 `content/demos.json`，只在卡片进入视野时播放。月面 7–9 复用既有直接预测结果：30 个训练窗口、1,000 步微调，测试场景未参与训练。每段包含共同初始帧及 16 个对应目标帧；真实位姿、采样时刻和源文件 SHA 保留在清单中。
- 导航区保留 8 个精选离线规划样例 RECON 059、088、046、025、085、074、061、078，展示起点、目标、预测视频及参考/规划轨迹。离线 ATE 为 0.263–0.807 m，终点误差为 0.498–1.327 m；所有预测帧已人工审阅。这些案例不代表实机闭环执行，筛选和推理记录见 `content/REFINEMENT_20261010_VALIDATION.md`。
- Results 的预测与导航基线对比使用相同图表及数据集/指标控件；导航支持 RECON、SCAND、Office-Go2、TartanDrive 的 ATE/RPE。论文基准数值未改动。
- 页面不显示模型参数数量、表格编号、FPS/尺寸及重复可视化解释；17.5M 数据集规模保留。采样协议和来源信息在 JSON 中维护。

## 本地预览

完整体验（网站 + 实时推理），在仓库根目录运行：

```bash
cd /path/to/OpenNWM
source .env
nvidia-smi
# 确认空闲 GPU 后选择编号；下面以 GPU 4 为例。
CUDA_VISIBLE_DEVICES=4 "$NWM_PYTHON" -u website/server/serve.py --port 8765
```

打开 <http://127.0.0.1:8765/>。远程机器可用 `ssh -L 8765:127.0.0.1:8765 your-host` 转发端口。服务默认只监听本机，首次生成时加载模型，后续复用；详细接口、输入范围及推理协议见 [server/README.md](server/README.md)。

需要在终端会话结束后继续运行时，使用独立后台会话（不要重复启动已运行的端口）：

```bash
source .env
preview_runtime_dir=/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/preview_runtime
mkdir -p "$preview_runtime_dir"
# 先检查显存与占用，再选择 GPU；模型在首次预测时才加载。
nohup setsid env CUDA_VISIBLE_DEVICES=1 "$NWM_PYTHON" -u website/server/serve.py --host 127.0.0.1 --port 8765 > "$preview_runtime_dir/server.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$preview_runtime_dir/server.pid"
```

用 `curl -f http://127.0.0.1:8765/api/health` 检查服务；启动后 `modelLoaded: false` 表示模型尚未加载。后台运行不依赖当前终端，但机器重启后需要重新启动。远程服务器上的 `127.0.0.1` 需经客户端端口转发访问；Codex 预览应打开该服务对应的浏览器端口代理。

只预览静态页面时，在仓库的 `website` 目录启动 Python 标准库服务器：

```bash
cd /path/to/OpenNWM/website
conda run --no-capture-output -n base python -m http.server 8000 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8000>，按 `Ctrl+C` 停止。也可用已有 Python 3 的绝对路径替换 `conda run --no-capture-output -n base python`，无需安装训练依赖。页面通过 HTTP 读取 JSON，请使用服务器预览。没有 API 时自动进入预计算模式，不会用预设结果冒充自绘轨迹的预测。

## GitHub Pages

2026-10-10 静态主图恢复提交 `1cfae647bd69ff34b494db141ab26bb3cc2b0df4` 已由 [Pages 工作流](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38057216933)成功部署。方法区恢复论文 Figure 2，移除动画与阶段控件；此前动画版本为历史记录。

2026-10-10 方法动画提交 `dfae157c0a209362ac5eaf2c134f2270400b248b` 已由 [Pages 工作流](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38055039324)成功部署。论文原始训练图配合原生网页阶段动画，详情见 `content/METHOD_ANIMATION_20261010_VALIDATION.md`。

2026-10-10 精简修订提交 `b3a99e28e5658618ddcb0897ddac190e39f4e829` 已由 [Pages 工作流](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38038052434)成功部署。28 个线上文件（8 个代码/JSON、5 个新视频和 15 张新图片）首次读取成功且 SHA256 一致，核对进程退出 0；公网文案、结构、图表与 HTTP/1.1 媒体播放验收已通过；传输差异及完整记录见 `content/REFINEMENT_20261010_VALIDATION.md`。以下为历史成功发布记录。

2026-10-10 结构修订提交 `44e3e5013643b65b971dd72c1fffe1f3d40cba76` 已由 [Pages 工作流](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38029698470) 成功发布；内容与验证记录见 `content/STRUCTURE_20261010_VALIDATION.md`。

2026-10-10 主题提交 `62012a11626a49bbd10836f841f2680c62536c38` 已由 [Pages 工作流](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38021431606) 成功发布；10 个线上文件哈希一致，正式网址桌面与手机配色、布局及滑块检查通过。静态站可选 API 的预期 404 单独记录，完整证据见 `content/REVISION_20261010_VALIDATION.md`。

2026-10-09 最终同步修复 `d82e5fe7f573a1df24835879fc2bb3eb41c0d059` 已经 [GitHub Pages 工作流](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/37883217041) 成功部署，12 个公网文件的哈希与公开仓库一致。04:17 UTC 的最终本地浏览器检查通过 9 个 motion 卡片、3 个新导航案例及 16 组中英文响应式布局，最大双视频漂移为 0.008309 秒。两路媒体分别中断 7 秒的实际断流检查也通过，共同暂停/恢复期间最大漂移不超过 0.000341 秒，缓冲时没有 seek 或时间倒退。

线上验收由两份报告共同覆盖：04:25 UTC 开始的检查通过全部 9 个 motion 卡片、3 个导航案例与 16 组布局，最大视频漂移 0.008296 秒，但最后刷新后的播放检查仍使用旧 20 秒等待，超时后整轮报告为 `passed: false`；04:34 UTC 开始的独立补验把媒体等待统一为 120 秒，自动离屏暂停/恢复、全局暂停及单卡暂停保持均通过，报告为 `passed: true`、退出状态 0。两次均无 JavaScript、HTTP 或请求失败，真实公网桌面导航和手机画廊截图已目测正常。此次测试的 motion 首播等待为 17.6–92.4 秒；网络诊断确认是媒体传输缓慢，补验期间未修改网站源码。所有验收项目已有通过证据，但不将超时的整轮报告标记为通过。

首轮修订 `7cdf56ca8ff09f8f3efd0e0f42c40d9417b122f8` 的 [工作流](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/37881633118) 部署成功，11 个线上文件哈希与公开仓库一致，但线上 TartanDrive 视频漂移 1.692 秒，未通过同步验收。后续中间版虽通过常规本地检查，断流测试仍发现缓冲时回退时间的问题；最终修复保留共同播放时间，每次等待屏障至多执行一次 seek。完整结果及历史失败保留在 [本次验证记录](content/REVISION_20261009_VALIDATION.md)。

2026-09-30 修订已同步并发布到公开仓库，部署提交为 `fa383fa3b4557a518925cc1d0ebb455258237df7`，[GitHub Pages 工作流](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/36725653537)成功。公开页面以同一段影片依次展示 12 个地球场景、2 个月球场景、2 个火星场景及 225 格无文字视频墙；19 个数据集卡片全部使用视频，交互 Demo 提供 3 个精选场景的 9 个真实预计算预测。静态 Pages 不提供 GPU 实时推理，自绘轨迹需要连接推理服务。

网站地址：<https://b0b8k1ng.github.io/OpenNWM/>。

网站通过独立公开仓库 [B0B8K1ng/b0b8k1ng.github.io](https://github.com/B0B8K1ng/b0b8k1ng.github.io) 免费部署；该仓库仅包含网站文件，原始 [OpenNWM](https://github.com/B0B8K1ng/OpenNWM) 代码仓库继续保持私有。

在 **b0b8k1ng.github.io** 仓库的 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。随后进入 **Actions → Deploy website to GitHub Pages → Run workflow**，选择 `main` 运行一次。以后向该公开仓库的 `main` 推送修改时会自动部署，无需构建或安装前端依赖；部署状态见工作流的 `github-pages` 环境。

工作流将网站发布到 `OpenNWM/` 子路径，并为用户站根地址和旧地址 `/OpenNWM-website/` 提供跳转，统一访问上方网站地址。

独立仓库的根目录就是网站目录，本地预览时将上方命令中的目录改为 `/path/to/OpenNWM-website`。在私有仓库修改 `website/` 后，将更新的网站文件同步到公开仓库并提交、推送，即可发布。

## 修改内容

| 文件 | 内容 |
| --- | --- |
| `index.html` | 页面结构、固定文案及 `data-en` / `data-zh` 双语文本 |
| `style.css`、`reference-theme.css`、`app.js` | 全局样式、数据集交互、语言切换与页面初始化 |
| `research.js`、`method.css` | 单张论文架构主图与视觉预测基线对比图 |
| `motion-gallery.js`、`motion-gallery.css` | 12 个场景（9 地球 + 3 月面）的轨迹及双视频滑块对照 |
| `navigation.js`、`navigation.css` | 8 个离线规划样例、同步轨迹与导航基线对比图 |
| `cinematic.js`、`playground.js` | 首屏固定视频墙/月球/火星播放、鼠标/触屏轨迹与推理状态 |
| `content/paper.json` | 论文信息、方法、结果、数据规模及原文出处 |
| `content/PAPER_SOURCES.md` | 论文页码、表格来源及统计口径 |
| `content/datasets.json` | 数据集卡片、环境标签、首页视频与真实素材来源 |
| `content/demos.json` | 模型对照视频、权重信息、预测协议与样例来源 |
| `content/navigation.json` | 离线规划样例、参考/规划轨迹、指标与素材校验信息 |
| `content/cinematic.json` | 第一人称开场影片与视频墙的场景名、片段时间、来源和剪辑窗口 |
| `content/playground.json` | 筛选后的交互场景、对应轨迹与逐条推理记录 |
| `content/space.json` | 月面测试及历史行星素材的来源档案；当前画廊入口由 demos.json 维护 |
| `server/` | 本地静态/API 服务、轨迹转动作、真实 GPU 推理和预设生成 |
| `assets/` | 论文、数据集素材、模型演示和本地字体；字体许可见 `assets/fonts/*-OFL.txt` |
| `../tools/build_website_motion_gallery.py` | 从完成的推理归档导出新增预测对照素材（CPU） |
| `../tools/build_website_lunar_gallery.py` | 从已完成月面直接预测归档导出滑块素材（CPU），并核对源帧、位姿和权重 SHA |
| `../tools/build_website_curved_navigation.py` | 新版曲折导航的候选筛选、独立 CEM 推理及已审阅案例导出 |
| `../tools/export_website_navigation.py` | 历史 2026-10-08 导航归档导出器；不用于新版曲折导航 |

替换媒体时，把文件放入对应 `assets/` 子目录，再修改 JSON 中的相对路径。保留稳定且唯一的 `id`；新增来源时保持 `paper.json` 与 `datasets.json` 的来源标识一致。修改论文指标或固定文案时，同时检查首页相关数字及中英文显示。

数据集区的 **15 个 NavAnywhere 视频预训练来源 + 4 个有动作标注来源**全部使用连续帧视频，卡片进入视野时播放，离开时暂停，也可打开弹窗播放。封面仅用于加载前占位；展示帧率不代表原始拍摄帧率。论文的 NavAnywhere 总规模仅对应其中 15 个来源。

Prediction 以紧凑网格展示 9 个地球与 3 个月面场景，每张卡片包含真实轨迹与两路同步视频。导航区通过独立清单管理 8 个局部规划案例，真实曲折路径、同步游标和预测视频均已通过浏览器检查。交互区提供筛选后的轨迹条件预计算结果，也支持连接本地服务后绘制合法平面轨迹。

交互演示使用 Nav15 Stage 2 joint100k 权重、seed 0、250 步 DDPM，单张初始图复制成四帧上下文，自回归预测 16 帧，以 4 FPS 播放 4 秒。VAE 使用后验均值，与基准评估的后验采样不同；这些是定性演示，不作为论文分数。Prediction 中的月面样本使用另经 30 个 LuSNAR 窗口、1,000 步微调的结果，在未参与微调的仿真月面 7–9 展示预测迁移；不代表实机导航验证。旧未微调行星压力测试及其漂移记录保留在 JSON 和素材中，当前页面不展示。行星素材的播放时间不代表真实任务时间。

开场为 25.25 秒的影片：固定完整的 225 格数据集视频墙（3 秒）→ LuSNAR 月球仿真 → SynMars-TW 火星仿真连续帧。影片内不叠加人物或机器人贴图，进度条上的小车随播放移动。片段名称同步显示，提供章节按钮和键盘跳转。O 字内部为 1/2 地球、1/4 月球、1/4 火星纹理。网站每次加载默认英语，可手动切换中文。

视频墙使用来自 NavAnywhere 全部 15 个来源的 225 条不同轨迹，不烧录任何文字。它位于同一个首屏 MP4 的开头（0–3 秒），直接展示完整视频墙，随后播放月球与火星。实际时长和章节边界以 `content/cinematic.json` 为准；来源追溯保留在清单中。

数据集弹窗视频会先加载为浏览器内存中的 Blob，再交给原生播放器，以支持普通 Python 静态服务器上的时间轴拖动。其他视频使用原生媒体加载；建议保持每段素材在几 MB 以内。

浏览器验收范围包括中英切换、预设播放、切换场景清除旧预测、画廊播放与响应式布局，具体结果见本次验证记录。系统开启“减少动态效果”时，背景和预测画廊不自动播放；月面预测卡片遵循相同的播放控制。数据筛选及论文展示保留。

## 重新制作素材

在仓库根目录运行，源数据和较大的中间文件保留在 NAS，网站只保存最终素材及可公开的溯源信息：

预测画廊从已有归档重新导出，使用 CPU，不运行模型；需要 `base` 环境中的 Pillow 和系统 `ffmpeg`：

```bash
conda run --no-capture-output -n base python tools/build_website_motion_gallery.py
```

新版曲折导航使用 `tools/build_website_curved_navigation.py`，分为 `prepare`（CPU 候选筛选）、`run`（GPU 推理）和 `publish`（已审阅素材导出，不运行模型）三步。`run` 与 `publish` 都要求通过 `--ids` 明确指定样例；当前采用 `recon-059`、`recon-088`、`recon-046`、`recon-025`、`recon-085`、`recon-074`、`recon-061`、`recon-078`。生成结果和审阅图保存在 NAS，`publish` 才会写入网站素材和 `content/navigation.json`。运行推理前检查 GPU 与磁盘空间，使用 `nwm` 环境并显式设置选定的 `CUDA_VISIBLE_DEVICES`；记录命令、PID 和日志，不替换正在运行的实验。

```bash
conda run --no-capture-output -n nwm python tools/build_website_curved_navigation.py --help
conda run --no-capture-output -n nwm python tools/build_website_curved_navigation.py publish --ids recon-059 recon-088 recon-046 recon-025 recon-085 recon-074 recon-061 recon-078
```

导出器要求至少两个转弯段各达到 6°、总转角至少 25°；终点误差不超过 1.5 m 且归一化值不超过 0.35，ATE 不超过 1 m 且归一化值不超过 0.25，并需人工审阅全部预测帧。2026-10-10 扩展推理分 4 批完成，全部退出 0；命令、PID、日志、来源脚本快照和筛选记录见本轮验收文档。论文基准分数与这些独立生成的离线定性样例分开维护。旧 `tools/export_website_navigation.py` 会重写导航清单为历史样例，仅在恢复旧版时使用。

其他素材与可选实时推理：

```bash
conda run --no-capture-output -n base python tools/build_website_cinematic.py sample
conda run --no-capture-output -n base python tools/build_website_cinematic.py prepare
conda run --no-capture-output -n base python tools/build_website_cinematic.py render
conda run --no-capture-output -n base python tools/build_website_dataset_videos.py
source .env
# 先检查 GPU；生成器跳过已完成的场景。
CUDA_VISIBLE_DEVICES=4 "$NWM_PYTHON" -u website/server/generate.py
conda run --no-capture-output -n base python tools/build_website_space.py
"$NWM_PYTHON" -m unittest discover -s website/server -p 'test_*.py'
```

本次修订验证记录见 [content/PREDICTION_20261010_VALIDATION.md](content/PREDICTION_20261010_VALIDATION.md)。上一轮导航扩展记录见 [content/REFINEMENT_20261010_VALIDATION.md](content/REFINEMENT_20261010_VALIDATION.md)。2026-10-10 的结构/主题记录与 [2026-10-09 同步验收](content/REVISION_20261009_VALIDATION.md)继续保留。此前发布记录保留在 [content/REVISION_20260930_VALIDATION.md](content/REVISION_20260930_VALIDATION.md)，早期改版记录见 [content/REDESIGN_VALIDATION.md](content/REDESIGN_VALIDATION.md)。增换样例时同步更新 checkpoint、轨迹、采样协议和帧数等来源记录；定性样例与论文汇总评测结果分别维护。

## 搬迁与状态

完整复制 `website/` 即可在其他机器或支持静态文件的服务器上使用；保持目录层级及相对路径，不依赖本机 NAS 或旧项目目录。JSON 中的源轨迹信息用于追溯，不是页面加载所需的本地路径。静态部署的根目录应设为 `website/`。

代码仓库目前为私人仓库，GitHub 链接需要协作者权限。论文当前标注为匿名稿件、ICLR 2027 审稿中，引用为临时版本；作者、录用状态和正式链接确认后，再同步更新 `paper.json`、页面与引用文本。

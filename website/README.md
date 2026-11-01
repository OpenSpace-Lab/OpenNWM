# OpenNWM 项目网站

原生 HTML、CSS 和 JavaScript 学术主页，无前端构建步骤。正式地址：
[OpenNWM](https://b0b8k1ng.github.io/OpenNWM/)。此目录也可独立作为静态站部署；
论文、字体、图片和视频均保留在 `assets/`，模型服务不属于静态站依赖。

## 页面结构

当前页面按以下顺序展示：

1. 首页背景影片、OpenSpace Lab 标识，以及论文、代码、模型、Demo 和数据集入口。
2. 作者与单位。
3. 方法图使用 2026-10-31 提供的 `method.pdf`，直接导出为矢量 SVG，网页使用 `assets/paper/method-overview-dark-20261031.svg`：透明底与页面 `--paper` 背景一致，文字、箭头和模块使用深色主题配色。保留原始结构、公式轮廓和照片字节；仅对公式位图、Logo 文字和阴影应用 SVG 颜色滤镜。文字矢量轮廓放大时保持清晰且不依赖字体；点击打开逐字节保留的原始 PDF。原始导出 `method-overview-20261031.svg` 一并保留，来源和配色映射记录在各自同名 JSON 中。绿色 N 的独立矢量文件为 `assets/opennwm-logo.svg` 和 `assets/opennwm-logo.pdf`，均为透明背景、无嵌入位图。
4. 论文 Table 1 的未来预测结果、Table 2 的导航结果，以纵向排列的 Prediction / Navigation 竖柱图展示精简指标，方法名置于横轴；预测和导航采用一致的双列绘图区，保留全部模型对比。
5. 真机导航：按完整记录分组，所有子片段直接平铺；每个场景至少 4 段，实际就绪内容及筛选记录以 `content/go2.json` 为准。
6. Visualization：地球、月面、火星的未来预测对照，以及离线导航可视化。
7. Demo：选择场景、绘制轨迹、提交真实模型推理并查看返回视频。
8. NavAnywhere Dataset：桌面端每行五个视频，全部来源直接排列。

论文结果取自 `content/paper.json`，保留原始数值、优劣方向、并列和缺失值；
精选视频不改变论文汇总分数。数据集的 17.5M visual observations、15 pretraining
sources、70.8K navigation sequences 放在数据集介绍中。数据集卡片包括 15 个
预训练来源和 4 个额外有动作标注的来源，后者不计入 NavAnywhere 的 15 个来源。

真机视频以第三人称为主体，同屏展示对齐的第一人称 GT、OpenNWM/NWM rollout 和参考/规划
轨迹，每条记录的全部片段直接展示，桌面端两列、手机端一列，无需切换。
视频使用清晰的 1080p 无音轨导出。右下角 ATE/RPE 使用该片段全部完整
一秒规划窗口的算术平均值，整段保持不变，数值旁的向下箭头表示越低越好。

当前画廊按已有评测结果选片，每条完整录制作为一个场景，各展示 4 段，共 11 个
场景、44 段。其中 41 段（93.2%）的 OpenNWM ATE 与 RPE 均低于 NWM。
部分片段根据连续规划窗口裁短，并重新计算该时段指标、渲染视频中的数值；未重新
运行或改动模型预测。原始记录、完整候选评测与选片依据继续保留。这是经成绩筛选的
定性展示，与论文汇总基准分别记录；完整协议见[真机画廊说明](content/GO2_MANIFEST.md)。

火星对照使用 MarsLab 的 main_crater 地形与 Perseverance 资产，通过 Isaac Sim 4.1
兼容录制器运行物理仿真，采用简化照明。RGB 与车体真值位姿以 4 Hz 同步记录；动作
由实际平面位移和转角计算，米制位姿及控制命令另行保留。使用 6 条训练轨迹的
30 个片段（480 个目标对）微调 OpenNWM 1,000 步，1 条轨迹用于验证，3 条用于测试。
网页固定展示三条测试轨迹各自的第一个片段，分别为左弯、右弯和 S 弯，按 1× 仿真时间播放。这是同一地形上的
留出轨迹测试，不代表跨地形泛化。协议、全部测试结果与限制见
[火星弯道验证记录](content/MARSLAB_CURVES_20261029_VALIDATION.md)。

Moon 7–9 是 LuSNAR 的仿真场景编号。月面模型使用 Moon 1–5 的 30 个片段微调
1,000 步，Moon 6 用于验证，Moon 7–9 仅用于测试。当前三个片段在推理前按真实转弯
幅度选定，沿用原微调权重；详情见[月球选样说明](content/LUNAR_CURVES_20261029_VALIDATION.md)。
画廊下方提供简短的英文训练与测试划分说明；完整协议保留在验证记录中。

## 媒体加载与播放

影片、真机和数据集视频静音自动播放，不提供开始/暂停按钮；首页保留进度展示。
浏览器仍决定自动播放资格。`media-scheduler.js` 统一调度可见视频及同步视频组，
优先加载可见画面；当前视频有约 2 秒缓冲后，桌面最多为两个邻近项目、手机最多为一个
邻近项目预缓冲视频帧。离屏或隐藏标签页时暂停并释放不需要的传输和解码资源，回到
可见区域恢复进度，包括同步视频组。它不降低视频分辨率、码率或帧率。

数据集预览保留现有视频与实际播放参数，页面不显示视频速度、帧率或时间依据标注。
各来源的采样时间和导出参数仍保留在 `content/datasets.json` 中用于溯源。

`content/media-delivery.json` 指定经过校验的 ModelScope 媒体镜像，当前固定版本为：

```text
https://modelscope.cn/models/LittleBoss/OpenNWM-Website/resolve/site-20261031-v7/
```

`index.html` 的 `media-delivery-config` 内联同一份配置，使首次加载无需额外网络请求。
更新镜像时需同步两处配置，发布流程会检查一致性；缺少有效内联配置时才读取 JSON 文件。

镜像保持 `assets/...` 路径，文件与网站副本逐字节相同，支持视频 Range 请求和
跨域加载。`media-delivery.js` 在配置缺失时使用本站路径，并在单个镜像资源失败时
回退到本站副本或固定提交的同字节源码归档；本次页面访问会记住失败 URL，滚回视频时不会重新请求同一失败镜像。
源素材和本站文件没有被替换为低清版本。方法图的 SVG 和原始 PDF 直接由本站提供，不依赖视频镜像。
公开校验清单为镜像根目录的 `media-manifest.json`。
该版本包含当前展示的 44 个真机片段、19 个重新定时的数据集视频以及月球/火星曲线素材，
各文件的远端大小与 SHA256 均已校验。小 MP4 也通过 LFS 分发，保持上传文件字节，
使数据集、真机和月球/火星视频都支持标准 HTTP 206 首段与尾段请求。
`localAssetPrefixes` 为空。Garden paths 和 Autumn grove 的 8 段视频使用原始 CRF18
母版，移除了 CRF23 网页二次压缩，保持 1080p、24 fps、时长及预测内容。
为控制 Pages 站点体积，这 8 段视频的同画质备用源由固定 Git 提交的源码归档提供
（`content/go2.json` 中的 `fallback_video`）；其他资源保留本站副本。
这 8 段母版以 4 秒为自动播放启动缓冲目标；浏览器停止预加载或准备超过 10 秒且
已有可播放帧时允许启动。`media-cache-worker.js` 在流式播放的同时保存完整原始字节，
后续循环的 Range 请求复用本地缓存。缓存不可用时继续普通网络播放；不重新编码视频。

真机画廊的 44 段视频均已进行人脸复核。有人脸的片段采用局部、随时间移动的马赛克，
第一视角 GT、OpenNWM 和 NWM 使用相同遮罩区域；预览图同步更新。
选段、时长、帧数、轨迹及评分不变，OpenNWM 仍在 41/44 段中同时取得更低的 ATE/RPE。
核验记录见 `assets/go2/privacy-review.json`。Pages 只部署当前清单选中的真机视频和
预览图，排除已被替换的版本。

## 本地预览与部署

进入包含 `index.html` 的目录。源代码仓库中是 `website/`，独立网站仓库中是根目录：

```bash
cd /path/to/site
conda run --no-capture-output -n base python -m http.server 8000 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8000/>；按 `Ctrl+C` 停止。可用 Python 3 的绝对路径替换上方
`conda run`。页面通过 HTTP 读取 JSON，请通过服务器预览。

公开站由 [网站仓库](https://github.com/B0B8K1ng/b0b8k1ng.github.io) 的 GitHub Pages
工作流部署到 `/OpenNWM/`。保持所有相对路径和目录层级即可迁移到其他静态服务器。
页面仅提供英文。推送前检查桌面/手机布局、媒体回退以及 Demo 的在线/离线状态。

## 自定义轨迹推理

网页读取 `content/live-service.json` 中的 HTTPS API 地址，提交绘制的轨迹、轮询状态，
完成后自动展示真实返回视频。当前服务使用单张本机 GPU、OpenNWM Stage 2 joint100k
原始 DDPM 模型和 50 个扩散步，不使用 DMD2。输出 16 帧，以 4 FPS 展示；返回的
manifest 记录轨迹、动作、权重校验值和实际采样配置。历史预计算场景使用 250 步
DDPM，其协议与实时请求分别保存。

当前公网连接使用 Cloudflare quick tunnel，GPU 服务和隧道进程必须持续运行。
重建隧道后地址可能改变，需要更新并发布 `content/live-service.json`。API 离线时
仍可查看真实预计算结果；自绘轨迹不会被替换成预设结果。静态 Pages 本身不执行模型。

部署、GPU 选择、队列与接口说明见源代码仓库的
[推理服务文档](https://github.com/B0B8K1ng/OpenNWM/blob/main/website/server/README.md)。
服务代码和运行环境由源代码仓库管理，独立静态网站无需包含 `server/`。

## 内容维护与溯源

| 文件 | 内容 |
| --- | --- |
| `index.html`、`reference-theme.css` | 页面结构、作者单位、英文固定文案与主题 |
| `research.js`、`results.css`、`content/paper.json` | 论文结果柱状图及原始数值 |
| `content/go2.json`、`go2.js` | 当前精选真机片段与显示顺序 |
| `content/demos.json`、`content/navigation.json` | 预测/离线规划视频、轨迹和推理来源 |
| `content/playground.json`、`content/live-service.json` | Demo 场景、预计算结果及实时 API 地址 |
| `content/datasets.json`、`content/cinematic.json` | 数据集视频及首页影片来源 |
| `media-scheduler.js`、`media-delivery.js` | 可见区域播放调度、镜像加载与本站回退 |
| `assets/` | 公开媒体、论文、本地字体和对应许可 |

新增或替换媒体时，同步更新清单中的路径、稳定 ID、实际时长、采样协议和校验值。
先检查 GPU 与磁盘容量，再运行需要推理或大规模输出的源仓库工具；源视频、完整对齐
视频、模型输出和内部审计保留在 NAS，网站只使用最终展示素材及可公开的溯源信息。

论文来源见 [PAPER_SOURCES.md](content/PAPER_SOURCES.md)，真机数据约定见
[GO2_MANIFEST.md](content/GO2_MANIFEST.md)。历史方法与媒体审阅分别保留在
[主图记录](content/MAIN_FIGURE_20261010_VALIDATION.md)、
[预测记录](content/PREDICTION_20261010_VALIDATION.md)、
[导航记录](content/REFINEMENT_20261010_VALIDATION.md)、
[火星记录](content/SYNMARS_20261029_VALIDATION.md) 和
[同步验证](content/REVISION_20261009_VALIDATION.md) 中；其中历史页面布局及部署状态
不是当前页面规范。

# OpenNWM 项目网站

原生 HTML、CSS、JavaScript 静态学术主页，无前端依赖、构建步骤或后端。图片、视频、字体和论文均随网站保存，浏览网站无需 GPU、模型权重或原始数据集。

## 本地预览

在仓库的 `website` 目录启动 Python 标准库服务器：

```bash
cd /path/to/OpenNWM/website
conda run --no-capture-output -n open-nwm python -m http.server 8000 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8000>，按 `Ctrl+C` 停止。也可用已有 Python 3 的绝对路径替换 `conda run --no-capture-output -n open-nwm python`，无需安装训练依赖。页面通过 HTTP 读取 JSON，请使用服务器预览。

## GitHub Pages

网站地址：<https://b0b8k1ng.github.io/OpenNWM-website/>（首次部署成功后可访问）。

网站通过独立公开仓库 [B0B8K1ng/OpenNWM-website](https://github.com/B0B8K1ng/OpenNWM-website) 免费部署；该仓库仅包含网站文件，原始 [OpenNWM](https://github.com/B0B8K1ng/OpenNWM) 代码仓库继续保持私有。

在 **OpenNWM-website** 仓库的 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。随后进入 **Actions → Deploy website to GitHub Pages → Run workflow**，选择 `main` 运行一次。以后向该公开仓库的 `main` 推送修改时会自动更新网站，无需构建或安装前端依赖；部署状态和实际地址见工作流的 `github-pages` 环境。

独立仓库的根目录就是网站目录，本地预览时将上方命令中的目录改为 `/path/to/OpenNWM-website`。在私有仓库修改 `website/` 后，将更新的网站文件同步到公开仓库并提交、推送，即可发布。

## 修改内容

| 文件 | 内容 |
| --- | --- |
| `index.html` | 页面结构、固定文案及 `data-en` / `data-zh` 双语文本 |
| `style.css`、`app.js`、`research.js` | 样式、视频/数据集交互、方法/结果展示 |
| `content/paper.json` | 论文信息、方法、结果、数据规模及原文出处 |
| `content/PAPER_SOURCES.md` | 论文页码、表格来源及统计口径 |
| `content/datasets.json` | 数据集卡片、环境标签、首页视频与真实素材来源 |
| `content/demos.json` | 模型对照视频、权重信息、预测协议与样例来源 |
| `assets/` | 论文、数据集素材、模型演示和本地字体；字体许可见 `assets/fonts/*-OFL.txt` |

替换媒体时，把文件放入对应 `assets/` 子目录，再修改 JSON 中的相对路径。保留稳定且唯一的 `id`；新增来源时保持 `paper.json` 与 `datasets.json` 的来源标识一致。修改论文指标或固定文案时，同时检查首页相关数字及中英文显示。

数据集区展示 **15 个 NavAnywhere 视频预训练来源 + 4 个有动作标注来源**的真实画面。图片仅裁剪、缩放；视频来自连续存储帧，展示帧率不代表原始拍摄帧率。论文的 NavAnywhere 总规模仅对应其中 15 个来源。

模型演示区播放已有推理归档中的 Ground Truth、OpenNWM 和 NWM 对照视频，**不运行实时推理**。增换样例时同步更新其 checkpoint、轨迹、采样协议和帧数等来源记录；精选定性样例与论文的汇总评测结果分别维护。

小视频会先加载为浏览器内存中的 Blob，再交给原生播放器，保证普通 Python 静态服务器也支持拖动时间轴。建议保持每段素材在几 MB 以内。

已在 Chrome 133 检查同步播放、拖动/倍速/切换模型、数据筛选与弹窗、结果数值、中英切换、键盘操作，以及 320–1440 px 布局。系统开启“减少动态效果”时，背景与弹窗视频不自动播放。

## 搬迁与状态

完整复制 `website/` 即可在其他机器或支持静态文件的服务器上使用；保持目录层级及相对路径，不依赖本机 NAS 或旧项目目录。JSON 中的源轨迹信息用于追溯，不是页面加载所需的本地路径。静态部署的根目录应设为 `website/`。

代码仓库目前为私人仓库，GitHub 链接需要协作者权限。论文当前标注为匿名稿件、ICLR 2027 审稿中，引用为临时版本；作者、录用状态和正式链接确认后，再同步更新 `paper.json`、页面与引用文本。

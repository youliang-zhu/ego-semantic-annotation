# Ego Semantic Annotation

基于 [Macrodata Refiner](https://github.com/macrodata-labs/refiner) 的第一人称视频语义标注实验。保留上游源码、来源说明和 Apache-2.0 许可证；本项目增加了 Windows 本地 WGO-Bench demo，以及 Gemini API 完整输入输出记录。公开仓库从干净的初始提交开始，上游基线为 `5f76982`。

## 本项目入口

- `run_wgo_demo.ps1`：Windows 启动脚本，默认模型为 `gemini-3.8-flash`。
- `examples/wgo_ego/run_demo.py`：两轮标注、GT 展示和结果保存。
- `examples/wgo_ego/api_recording.py`：实际 prompt、图片、输出 Schema、原始响应记录。
- [Windows 使用说明](docs/examples/annotations/wgo-ego-windows.md)：安装、目录结构、API Key 配置及运行方法。

在项目根目录运行已验证成功的叠毛巾片段：

```powershell
.\run_wgo_demo.ps1 -Prepare -DataRoot '..\datasets\wgo-bench-ego-homer_56' -OutputRoot '.\artifacts\wgo-ego-homer_56'
.\run_wgo_demo.ps1 -RecordApi -DataRoot '..\datasets\wgo-bench-ego-homer_56' -OutputRoot '.\artifacts\wgo-ego-homer_56'
```

API Key 填在项目根目录 `.env`（参考 `.env.example`）。本地环境使用 `.venv/`。上述命令的输出写入 `artifacts/wgo-ego-homer_56/`，数据集位于项目上一级的 `datasets/wgo-bench-ego-homer_56/`。不指定目录时仍使用原来的 `wgo-bench-ego` 数据和 `wgo-ego-demo` 输出。这些本地文件不提交到 Git；新克隆需要安装依赖并准备数据，详见使用说明。

实验状态（2026-10-10）：`homer_56` 的前 16.67 秒已完成两轮标注，4 次 API 请求全部成功，得到 3 段预测；源 GT 有 4 段。完整 prompt、11 张实际输入图片、Schema 和原始响应均已保存并校验。打开输出目录的 `api-recording.html` 查看请求，`comparison.html` 查看原始 GT 对照。局部时间分段 F1 为 0.571（IoU ≥ 0.75、评分时对齐首尾；未运行语义评分模型，不是完整 benchmark 得分）。原先 `homer_1` 倒水片段仍返回 `PROHIBITED_CONTENT`，记录保留。第二轮为标签修正，不是 benchmark 评分；仓库不包含博客的完整评估器。

以下为上游项目介绍。

---

<p align="center">
  <img src="https://macrodata.co/logo.svg" alt="Macrodata" width="180">
</p>

<h1 align="center">Macrodata Refiner</h1>

Refiner is Macrodata's open-source Python library for reading, transforming, and
writing robotics datasets.

It provides one pipeline model for working with robot episodes, frames, videos,
metadata, and model-based processing. Use it to convert formats, transform data,
run inference, and write structured outputs on your own infrastructure.

This repository includes reference implementations from our published research.
At Macrodata, we develop proprietary models and pipelines for robotics data
processing. [Contact us directly](https://macrodata.co/contact) for access.

## Quickstart

Install:

```bash
pip install macrodata-refiner
```

This gives you:

- the Python package as `refiner`
- the CLI as `macrodata`

Launch a local pipeline:

```python
import refiner as mdr

def add_preview(row):
    return row.update(
        preview=" ".join(row["text"].split()[:20]),
    )

(
    mdr.read_jsonl("input/*.jsonl")
    .filter(mdr.col("lang") == "en")
    .with_columns(
        text=mdr.col("text").str.strip(),
        text_len=mdr.col("text").str.len(),
    )
    .map(add_preview)
    .write_parquet("s3://my-bucket/english-cleanup/")
    .launch_local(
        name="english-cleanup",
        num_workers=2,
    )
)
```

## Batteries included

- a consistent row and episode model for robot trajectories, frames, videos,
  metadata, tasks, and statistics
- readers and writers for LeRobot, HDF5, Zarr, MCAP, Parquet, JSONL, and other
  common data formats
- composable transforms and model inference for converting and enriching data
- open-source reference operations for motion trimming, subtask annotation,
  reward scoring, and hand tracking
- access to storage backends supported by `fsspec`, including S3, GCP, and
  Hugging Face
- in-process debugging and local multi-worker execution

## Docs

Read the [Refiner open-source documentation](https://macrodata.co/docs).

Start here:

- [Docs index](docs/index.md)
- [Quickstart](docs/quickstart.md)
- [Running pipelines](docs/running-pipelines/index.md)

Build a dataset:

- [Reading data](docs/reading-data/index.md)
- [Episode data](docs/episode-data/index.md)
- [Transforms](docs/transforms/index.md)
- [Episode operations](docs/episode-operations/index.md)
- [Writing data](docs/writing-data/index.md)
- [Examples](docs/examples/index.md)
- [Reference](docs/reference/index.md)

## Community

- join the Macrodata Discord: https://discord.gg/S8kZtmBR2x

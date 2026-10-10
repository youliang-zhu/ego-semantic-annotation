"""Run the repository's two-pass annotation on one local WGO-Bench ego clip."""

from __future__ import annotations

import argparse
import getpass
import html
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT.parent / "datasets" / "wgo-bench-ego"
ROOT = PROJECT_ROOT / "artifacts" / "wgo-ego-demo"


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def render_report(result=None):
    sample = json.loads((DATA_ROOT / "sample.json").read_text(encoding="utf-8"))
    video_url = Path(os.path.relpath(DATA_ROOT / "clip.mp4", ROOT)).as_posix()

    def segment_table(title, rows):
        cells = "".join(
            f"<tr><td>{s['start_sec']:.2f}–{s['end_sec']:.2f}s</td><td>{html.escape(s['subtask'])}</td></tr>"
            for s in rows
        )
        return f"<h2>{title}</h2><table>{cells}</table>"

    sheets = "".join(
        f'<a href="{p.name}"><img src="{p.name}" alt="Timestamped contact sheet"></a>'
        for p in sorted(ROOT.glob("contact-sheet-*.jpg"))
    )
    status = "模型标注已完成" if result else "预处理已完成；模型预测尚未运行"
    status_path = ROOT / "run-status.json"
    if result is None and status_path.exists():
        run_status = json.loads(status_path.read_text(encoding="utf-8"))
        status = run_status["message"]
    original_gt = ""
    original_path = DATA_ROOT / "clip.gt.original.json"
    if original_path.exists():
        original_text = original_path.read_text(encoding="utf-8")
        source_id = sample.get("source_id", sample["id"].split("_0-")[0])
        links = " · ".join(
            f'<a href="{html.escape(Path(os.path.relpath(DATA_ROOT / name, ROOT)).as_posix(), quote=True)}">{name}</a>'
            for name in (
                "clip.gt.original.json",
                f"{source_id}.gt.original.json",
                f"{source_id}.original.parquet",
                "gt.provenance.json",
            )
        )
        original_gt = (
            "<h2>HF 原始 GT</h2><p>下面保留本片段 GT 的原字段、英文标签和时间值。"
            "完整样本见下方文件；Parquet 保留整行数据（含视频），JSON 仅省略二进制视频。"
            "单行 Parquet 经重新序列化，字段和值已校验一致。</p>"
            f"<p>{links}</p><pre>{html.escape(original_text)}</pre>"
            f"<p>GT 截取终点为 {sample.get('clip_requested_end_sec', sample['clip_duration_sec'])} 秒，"
            f"编码后视频时长为 {sample['clip_duration_sec']} 秒。GT 时间值未改写。</p>"
        )
    predicted = (
        ""
        if result is None
        else segment_table("第一轮：分段与初始标签", result["predicted_subtasks"])
        + segment_table("第二轮：固定边界后的标签", result["labeled_subtasks"])
    )
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>WGO-Bench ego demo</title>
<style>body{{font:16px/1.6 system-ui;max-width:1120px;margin:32px auto;padding:0 24px;background:#f4f6f8;color:#17212c}}video{{width:100%;max-height:540px;background:#111}}img{{max-width:100%;margin:12px 0}}table{{width:100%;border-collapse:collapse;background:white}}td{{padding:9px;border-bottom:1px solid #ddd}}td:first-child{{width:160px}}.status{{padding:12px;background:#e2ebf8}}</style>
<h1>WGO-Bench · HomER 第一人称片段</h1><p class="status">{html.escape(status)}</p>
<p>{html.escape(sample["id"])} · {sample["clip_duration_sec"]:.2f} 秒 · {html.escape(sample["instruction"])}</p>
<video controls preload="metadata" src="{html.escape(video_url, quote=True)}"></video>
<p>来源：<a href="https://huggingface.co/datasets/macrodata/WGO-Bench">macrodata/WGO-Bench</a>，HomER 子集。人工标签仅用于对照，不传入模型。裁剪与完整 benchmark 的评测条件不同。</p>
<h2>第一轮实际使用的输入拼图</h2><p>每 0.5 秒取一帧，224px 宽，每张最多 20 帧，5 列；所有拼图一起发送。</p>{sheets}
{original_gt}
{segment_table("GT 时间概览（显示到两位小数；精确值见上方原始 JSON）", sample["gold_segments"])}{predicted}
<p>样本信息：{html.escape(str(DATA_ROOT / "sample.json"))}；处理信息见 preparation.json，完成推理后见 result.json。</p></html>'''
    (ROOT / "preview.html").write_text(page, encoding="utf-8")


def make_source():
    import refiner as mdr

    sample = json.loads((DATA_ROOT / "sample.json").read_text(encoding="utf-8"))
    # Deliberately omit gold segments from the model's input row.
    return mdr.from_items(
        [
            {
                "id": sample["id"],
                "instruction": sample["instruction"],
                "video_path": (DATA_ROOT / "clip.mp4").as_posix(),
            }
        ]
    ).to_robot_rows(
        episode_id_key="id", task_key="instruction", video_keys={"ego": "video_path"}
    )


async def save_contact_sheets(row):
    from refiner.robotics.subtask_annotation.utils import timestamped_contact_sheets

    records = []
    async for sheet in timestamped_contact_sheets(row.videos["ego"]):
        name = f"contact-sheet-{sheet.index:02d}.jpg"
        (ROOT / name).write_bytes(sheet.data)
        records.append(
            {
                "file": name,
                "timestamps": sheet.timestamps,
                "width": sheet.width,
                "height": sheet.height,
                "bytes": len(sheet.data),
            }
        )
    write_json(
        ROOT / "preparation.json",
        {"status": "prepared_no_model_predictions", "sheets": records},
    )
    return row.update({"sheet_count": len(records)})


def save_first_pass(row):
    write_json(
        ROOT / "segmentation.json",
        {"id": row["id"], "predicted_subtasks": row["predicted_subtasks"]},
    )
    return row


def main():
    global DATA_ROOT, ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Decode video and save exact model input sheets; no API calls.",
    )
    parser.add_argument("--provider", choices=["google", "openai"], default="google")
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--base-url")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument(
        "--record-api",
        action="store_true",
        help="Save actual API prompts, image bytes, schemas and raw responses.",
    )
    args = parser.parse_args()
    DATA_ROOT = args.data_root.resolve()
    ROOT = args.output_root.resolve()
    if not DATA_ROOT.is_relative_to(PROJECT_ROOT.parent):
        parser.error("--data-root must be inside the semantic annotation directory")
    if not ROOT.is_relative_to(PROJECT_ROOT / "artifacts"):
        parser.error("--output-root must be inside the project's artifacts directory")
    # Resolve the project root regardless of the terminal's current directory.
    # Existing process environment variables take priority over .env values.
    load_dotenv(PROJECT_ROOT / ".env", override=False, encoding="utf-8-sig")
    ROOT.mkdir(parents=True, exist_ok=True)
    scratch = PROJECT_ROOT / ".work"
    temporary = scratch / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    for name in ("TEMP", "TMP", "TMPDIR"):
        os.environ[name] = str(temporary)
    tempfile.tempdir = str(temporary)
    os.environ["REFINER_WORKDIR"] = str(ROOT / "runtime")
    os.environ["XDG_CACHE_HOME"] = str(scratch / "cache")
    os.environ["HF_HOME"] = str(scratch / "cache" / "huggingface")
    pipeline = make_source()
    if args.prepare:
        rows = pipeline.map_async(save_contact_sheets, max_in_flight=1).materialize()
        render_report()
        print(
            f"Prepared {rows[0]['sheet_count']} contact sheets. Open {ROOT / 'preview.html'}"
        )
        return
    import refiner as mdr

    if args.provider == "google":
        from refiner.inference.providers import GoogleEndpointProvider

        key = next(
            (
                os.environ[n]
                for n in [
                    "GOOGLE_GENERATIVE_AI_API_KEY",
                    "GEMINI_API_KEY",
                    "GOOGLE_API_KEY",
                ]
                if os.environ.get(n)
            ),
            None,
        )
        if not key and sys.stdin.isatty():
            key = getpass.getpass("Gemini API key (hidden): ").strip()
        if not key:
            raise SystemExit(
                f"Missing Gemini API key. Fill GOOGLE_GENERATIVE_AI_API_KEY in {PROJECT_ROOT / '.env'}, then run again."
            )
        os.environ["GOOGLE_GENERATIVE_AI_API_KEY"] = key
        provider = GoogleEndpointProvider(
            model=args.model, **({"base_url": args.base_url} if args.base_url else {})
        )
    else:
        from refiner.inference.providers import OpenAIEndpointProvider

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "Set OPENAI_API_KEY for your chosen compatible endpoint locally."
            )
        url = args.base_url or os.environ.get("OPENAI_BASE_URL")
        provider = OpenAIEndpointProvider(
            model=args.model, **({"base_url": url} if url else {})
        )
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    output = ROOT / "runs" / stamp
    if args.record_api:
        from api_recording import install_recorder

        # Local workers must be able to import the example's recorder module.
        module_path = str(Path(__file__).resolve().parent)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            filter(None, [module_path, os.environ.get("PYTHONPATH")])
        )
        pipeline = pipeline.map(
            partial(install_recorder, directory=str(output / "api"))
        )
    pipeline = (
        pipeline.map_async(
            mdr.robotics.subtask_annotation(
                provider=provider,
                video_key="ego",
                max_concurrent_requests=1,
                on_blocked_prompt="raise",
            ),
            max_in_flight=1,
        )
        .map(save_first_pass)
        .map_async(
            mdr.robotics.subtask_labeling(
                provider=provider,
                video_key="ego",
                max_concurrent_requests=1,
                on_blocked_prompt="raise",
            ),
            max_in_flight=1,
        )
        .map(
            lambda row: {
                "id": row["id"],
                "instruction": row["instruction"],
                "predicted_subtasks": row["predicted_subtasks"],
                "labeled_subtasks": row["labeled_subtasks"],
            }
        )
    )
    write_json(
        ROOT / "run-status.json",
        {
            "status": "running",
            "model": args.model,
            "message": "API 标注正在运行，尚无完整结果。",
            "run_directory": str(output),
        },
    )
    try:
        stats = pipeline.write_jsonl(str(output / "predictions")).launch_local(
            name="wgo-ego-small-demo", num_workers=1, rundir=str(output / "runtime")
        )
    except RuntimeError:
        write_json(
            ROOT / "run-status.json",
            {
                "status": "failed",
                "model": args.model,
                "message": "本次管线运行失败；没有完整预测，无法评分。详情见本次运行日志。",
                "run_directory": str(output),
            },
        )
        render_report()
        raise
    finally:
        if args.record_api:
            from api_recording import render_recording

            render_recording(output / "api")
            relative_report = (
                (output / "api" / "index.html").relative_to(ROOT).as_posix()
            )
            (ROOT / "api-recording.html").write_text(
                f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={relative_report}"><a href="{relative_report}">打开本次 API 输入输出完整记录</a>',
                encoding="utf-8",
            )
            print(
                f"API input/output record: {output / 'api' / 'index.html'}", flush=True
            )
    if stats.failed:
        raise SystemExit(f"Pipeline failed; inspect {output}")
    files = list((output / "predictions").glob("*.jsonl"))
    rows = [
        json.loads(line)
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one output episode, got {len(rows)}")
    result = {
        **rows[0],
        "provider": args.provider,
        "model": args.model,
        "run_directory": str(output),
    }
    write_json(ROOT / "result.json", result)
    write_json(output / "result.json", result)
    write_json(
        ROOT / "run-status.json",
        {
            "status": "completed",
            "model": args.model,
            "message": "两轮 API 标注已完成。",
            "run_directory": str(output),
        },
    )
    render_report(result)
    print(f"Completed. Open {ROOT / 'preview.html'}")


if __name__ == "__main__":
    main()

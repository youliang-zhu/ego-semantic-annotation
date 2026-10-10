"""Record actual HTTP payloads without changing inference inputs or responses."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import html
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
from PIL import Image


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def install_recorder(row, *, directory):
    """Install once in the worker before its inference steps execute."""
    from refiner.inference.internal.transport import AiohttpAPIClient

    if getattr(AiohttpAPIClient.post, "_wgo_recording", False):
        return row
    original_post = AiohttpAPIClient.post
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    counter = 0

    async def recorded_post(client, endpoint_path, **kwargs):
        nonlocal counter
        counter += 1
        payload = kwargs["json"]
        texts = [
            part["text"]
            for content in payload.get("contents", [])
            for part in content.get("parts", [])
            if "text" in part
        ]
        stage = (
            "segmentation"
            if texts and texts[0].startswith("Reconstruct")
            else "labeling"
        )
        folder = root / f"{counter:03d}-{stage}"
        folder.mkdir()
        write_json(folder / "request.json", payload)
        (folder / "prompt.txt").write_text("\n\n".join(texts), encoding="utf-8")
        config = payload.get("generationConfig", {})
        write_json(folder / "generation-config.json", config)
        write_json(folder / "output-schema.json", config.get("responseSchema"))
        readable = copy.deepcopy(payload)
        images = []
        for content_index, content in enumerate(readable.get("contents", [])):
            for part_index, part in enumerate(content.get("parts", [])):
                media = part.get("inlineData")
                if not media:
                    continue
                data = base64.b64decode(media["data"], validate=True)
                extension = {"image/jpeg": ".jpg", "image/png": ".png"}.get(
                    media["mimeType"], ".bin"
                )
                name = f"image-{len(images) + 1:02d}{extension}"
                (folder / name).write_bytes(data)
                with Image.open(io.BytesIO(data)) as image:
                    size = list(image.size)
                images.append(
                    {
                        "file": name,
                        "mime_type": media["mimeType"],
                        "bytes": len(data),
                        "size": size,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "content_index": content_index,
                        "part_index": part_index,
                    }
                )
                media["data"] = (
                    f"[base64 bytes extracted to {name}; full data in request.json]"
                )
        write_json(folder / "request.readable.json", readable)
        metadata = {
            "stage": stage,
            "method": "POST",
            "url": f"{client.base_url.rstrip('/')}/{endpoint_path.lstrip('/')}",
            "request_headers": {
                "Content-Type": "application/json",
                "x-goog-api-key": "[REDACTED]",
            },
            "started_at": datetime.now(UTC).isoformat(),
            "images": images,
            "status": "pending",
        }
        write_json(folder / "metadata.json", metadata)
        started = time.monotonic()
        try:
            response = await original_post(client, endpoint_path, **kwargs)
            raw = await response.read()
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
            metadata.update(
                status="transport_error",
                error_type=type(exc).__name__,
                elapsed_sec=time.monotonic() - started,
            )
            write_json(folder / "metadata.json", metadata)
            raise
        (folder / "response.raw.json").write_bytes(raw)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body = {}
        write_json(folder / "response.json", body)
        block = body.get("promptFeedback", {}).get("blockReason")
        candidates = body.get("candidates", [])
        output = "".join(
            part.get("text", "")
            for candidate in candidates
            for part in candidate.get("content", {}).get("parts", [])
            if not part.get("thought")
        )
        (folder / "model-output.txt").write_text(output, encoding="utf-8")
        if output:
            try:
                structured = json.loads(output)
            except ValueError:
                structured = None
            if structured is not None:
                write_json(folder / "model-output.json", structured)
        metadata.update(
            status="blocked"
            if block
            else "success"
            if response.status < 400 and candidates and output
            else "error",
            http_status=response.status,
            block_reason=block,
            elapsed_sec=round(time.monotonic() - started, 3),
            response_id=body.get("responseId"),
            model_version=body.get("modelVersion"),
            usage=body.get("usageMetadata"),
            response_headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() not in {"set-cookie", "authorization", "x-goog-api-key"}
            },
        )
        write_json(folder / "metadata.json", metadata)
        print(
            f"API capture {folder.name}: {metadata['status']} (HTTP {response.status})",
            flush=True,
        )
        return response

    recorded_post._wgo_recording = True
    AiohttpAPIClient.post = recorded_post
    return row


def render_recording(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    sections = []
    index = []
    for path in sorted(root.glob("*/metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        folder = path.parent
        prefix = folder.name
        index.append({"directory": prefix, **meta})
        prompt = (folder / "prompt.txt").read_text(encoding="utf-8")
        schema = (folder / "output-schema.json").read_text(encoding="utf-8")
        response = (
            (folder / "response.json").read_text(encoding="utf-8")
            if (folder / "response.json").exists()
            else "No response received"
        )
        images = "".join(
            f'<figure><img src="{prefix}/{im["file"]}"><figcaption>{im["file"]} · {im["size"]} · {im["bytes"]} bytes</figcaption></figure>'
            for im in meta["images"]
        )
        links = " · ".join(
            f'<a href="{prefix}/{p.name}">{p.name}</a>'
            for p in sorted(folder.glob("*"))
            if p.suffix in {".json", ".txt"}
        )
        sections.append(
            f"<section><h2>{prefix} · {meta['status']}</h2><p>{html.escape(meta['url'])}</p><p>{links}</p><h3>实际 prompt</h3><pre>{html.escape(prompt)}</pre><h3>实际输入图片（按请求顺序）</h3>{images}<h3>要求的输出格式（Schema，不是实际预测）</h3><pre>{html.escape(schema)}</pre><h3>API 完整响应</h3><pre>{html.escape(response)}</pre></section>"
        )
    write_json(root / "index.json", index)
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Gemini API 输入输出记录</title>
<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;background:#f5f7fa;color:#16212d}section{background:white;padding:24px;margin:24px 0;border:1px solid #ccd5df}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f6;padding:16px;font-size:13px}img{max-width:100%}figure{margin:16px 0}a{color:#1758b0}</style>
<h1>Gemini API 输入输出记录</h1><p>以下为这次运行实际发出的每一个 HTTP 请求。success、blocked、error 分开记录，HTTP 200 本身不代表标注成功。</p>
<p>request.json 保留完整图片 base64；request.readable.json 将 base64 替换为图片文件说明，便于阅读。图片文件由实际请求解码保存，未经再次压缩。prompt.txt 是实际文本；output-schema.json 是要求的输出结构；response.raw.json 是原始响应体；model-output.json 是模型文本中的 JSON。API Key 不保存。</p>
<p>第一轮一次提交全部拼图，输出 segments 数组。第二轮按预测片段分别请求，输入前一段、当前段、后一段三张图，输出单个 subtask 标签。若第一轮失败，第二轮不会发生，也不会生成虚构示例。</p>"""
    (root / "index.html").write_text(
        page + "".join(sections) + "</html>", encoding="utf-8"
    )

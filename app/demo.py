"""Gradio demo UI, mounted into the FastAPI app at /demo.

A self-contained demo: upload a chest X-ray, see the top findings with their
GradCAM heatmaps rendered as images and the cited literature. Calls the SAME
Orchestrator the /analyze API uses, so the demo and the API can never diverge.

This is a demo surface for this backend repo — the production UI is the separate
Next.js frontend. Gradio is chosen for near-zero UI code and native image
rendering (Swagger can only show the heatmap as a base64 string).
"""
from __future__ import annotations

import base64
import io

import gradio as gr
from PIL import Image

from app.api.analyze import get_orchestrator

DISCLAIMER_MD = (
    "**MedVision AI is a research/educational prototype and is NOT a diagnostic "
    "tool.** Outputs are not a substitute for evaluation by a qualified clinician."
)


def _b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def analyze_image(image_path: str):
    """Run the orchestrator and return (gallery, markdown) for the UI."""
    if not image_path:
        return [], "Upload a chest X-ray to analyze."

    import os
    result = get_orchestrator().analyze(
        image_path, image_id=os.path.basename(image_path)
    )

    gallery = []      # (heatmap image, caption) for findings that have one
    lines = [f"### Findings ({result.num_present} present)\n"]

    for f in result.findings:
        marker = f"**{f.name}** — probability {f.probability:.2f} "
        marker += f"(threshold {f.threshold:.3f})"
        lines.append(marker)

        if f.heatmap_base64:
            img = _b64_to_image(f.heatmap_base64)
            gallery.append((img, f"{f.name} (p={f.probability:.2f})"))

        if f.articles:
            for a in f.articles:
                lines.append(f"- [{a.title}]({a.url}) — *{a.journal}* ({a.year})")
        lines.append("")

    return gallery, "\n".join(lines)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MedVision AI — Demo") as demo:
        gr.Markdown("# MedVision AI — Chest X-ray Analysis (Demo)")
        gr.Markdown(DISCLAIMER_MD)

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="filepath", label="Chest X-ray (PNG/JPG)")
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                gallery_out = gr.Gallery(label="GradCAM heatmaps (top findings)")

        findings_out = gr.Markdown()

        analyze_btn.click(
            fn=analyze_image,
            inputs=image_in,
            outputs=[gallery_out, findings_out],
        )

    return demo

from __future__ import annotations

import argparse
import sys
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aip_predictor.config import configured_path, load_config
from aip_predictor.fasta import records_from_text
from aip_predictor.pipeline import AIPPredictor


RESULT_COLUMNS = [
    "ID",
    "Sequence",
    "Length",
    "PLM_Probability",
    "DDE_Probability",
    "Final_Probability",
    "Prediction_Label",
    "Length_Note",
]

DISPLAY_COLUMNS = {
    "PLM_Probability": "PLM P",
    "DDE_Probability": "DDE P",
    "Final_Probability": "Final P",
    "Prediction_Label": "Label",
    "Length_Note": "Note",
}
DISPLAY_RESULT_COLUMNS = [DISPLAY_COLUMNS.get(column, column) for column in RESULT_COLUMNS]

CSS = """
:root {
  --aip-ink: #17211d;
  --aip-muted: #5f6d67;
  --aip-line: #d5ddd9;
  --aip-surface: #ffffff;
  --aip-green: #13745a;
  --aip-amber: #b56d16;
}
body, .gradio-container {
  background: #f4f6f5 !important;
  color: var(--aip-ink) !important;
  font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}
.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding: 22px !important;
}
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  border-bottom: 1px solid var(--aip-line);
  padding: 2px 0 18px;
  margin-bottom: 18px;
}
.app-title {
  margin: 0;
  font-size: 25px;
  line-height: 1.2;
  font-weight: 720;
  letter-spacing: 0;
}
.app-subtitle {
  margin: 5px 0 0;
  color: var(--aip-muted);
  font-size: 14px;
  letter-spacing: 0;
}
.local-badge {
  border: 1px solid #9fc6b9;
  color: #0d6048;
  background: #edf7f3;
  border-radius: 6px;
  padding: 7px 10px;
  font-size: 13px;
  white-space: nowrap;
}
.panel {
  background: var(--aip-surface);
  border: 1px solid var(--aip-line);
  border-radius: 7px;
  padding: 16px;
}
.section-title {
  font-size: 15px;
  font-weight: 680;
  margin: 0 0 10px;
  letter-spacing: 0;
}
.summary-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1px;
  border: 1px solid var(--aip-line);
  background: var(--aip-line);
  border-radius: 7px;
  overflow: hidden;
  margin: 16px 0;
}
.summary-item {
  background: #fff;
  padding: 12px 14px;
}
.summary-label {
  color: var(--aip-muted);
  font-size: 12px;
}
.summary-value {
  color: var(--aip-ink);
  font-size: 19px;
  font-weight: 700;
  margin-top: 2px;
}
#run-button {
  background: var(--aip-green) !important;
  border-color: var(--aip-green) !important;
}
#status-line {
  min-height: 42px;
}
footer { display: none !important; }
@media (max-width: 720px) {
  .gradio-container { padding: 12px !important; }
  .app-header { align-items: flex-start; flex-direction: column; }
  .summary-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""


def _summary_html(total: int = 0, aip: int = 0, non_aip: int = 0, elapsed: str = "-") -> str:
    return f"""
    <div class="summary-strip">
      <div class="summary-item"><div class="summary-label">Sequences</div><div class="summary-value">{total}</div></div>
      <div class="summary-item"><div class="summary-label">AIP</div><div class="summary-value">{aip}</div></div>
      <div class="summary-item"><div class="summary-label">Non-AIP</div><div class="summary-value">{non_aip}</div></div>
      <div class="summary-item"><div class="summary-label">Elapsed</div><div class="summary-value">{elapsed}</div></div>
    </div>
    """


def _progress_value(message: str) -> float:
    if message.startswith("Extracting 400"):
        return 0.03
    if message.startswith("Extracting five"):
        return 0.08
    if message.startswith("PLM cache"):
        return 0.06
    if message.startswith("Prediction cache"):
        return 0.02
    if message.startswith("PLM fold"):
        try:
            fold = int(message.split()[2].split("/")[0])
            return min(0.12 + fold * 0.12, 0.72)
        except (IndexError, ValueError):
            return 0.12
    if message.startswith("Running PLM"):
        return 0.78
    if message.startswith("Combining"):
        return 0.94
    if message.startswith("Prediction complete"):
        return 1.0
    return 0.01


class PredictorService:
    """Own one predictor instance and serialize access to GPU-backed inference."""

    def __init__(self, config_path: str | None, device: str | None):
        self.config_path = config_path
        self.device = device
        self.predictor: AIPPredictor | None = None
        self.lock = threading.Lock()

    def predict(self, records, progress: gr.Progress) -> pd.DataFrame:
        with self.lock:
            if self.predictor is None:
                progress(0.01, desc="Loading fitted classifiers")
                self.predictor = AIPPredictor(
                    config_path=self.config_path,
                    device=self.device,
                )
            self.predictor.progress = lambda message: progress(
                _progress_value(message), desc=message
            )
            return self.predictor.predict(records)


def _records_from_inputs(sequence_text: str, upload_path: str | None):
    if upload_path:
        text = Path(upload_path).read_text(encoding="utf-8-sig")
        return records_from_text(text)
    return records_from_text(sequence_text or "")


def build_app(config_path: str | None = None, device: str | None = None) -> gr.Blocks:
    config = load_config(config_path)
    results_dir = configured_path(config, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    service = PredictorService(config_path, device)

    def run_prediction(
        sequence_text: str,
        upload_path: str | None,
        progress: gr.Progress = gr.Progress(),
    ):
        started = time.perf_counter()
        try:
            records = _records_from_inputs(sequence_text, upload_path)
            result = service.predict(records, progress)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        result = result.loc[:, RESULT_COLUMNS]
        token = uuid.uuid4().hex[:8]
        output_path = results_dir / f"aip_predictions_{token}.csv"
        result.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.8f")
        elapsed_seconds = time.perf_counter() - started
        aip_count = int((result["Prediction_Label"] == "AIP").sum())
        non_aip_count = len(result) - aip_count
        summary = _summary_html(
            total=len(result),
            aip=aip_count,
            non_aip=non_aip_count,
            elapsed=f"{elapsed_seconds / 60:.1f} min" if elapsed_seconds >= 60 else f"{elapsed_seconds:.1f} s",
        )
        status = f"Completed {len(result)} sequence(s). Results saved locally."
        display_result = result.rename(columns=DISPLAY_COLUMNS)
        return display_result, str(output_path), summary, status

    def load_example() -> tuple[str, None]:
        text = (ROOT / "examples" / "example_sequences.fasta").read_text(encoding="utf-8")
        return text, None

    def clear_all():
        return "", None, pd.DataFrame(columns=DISPLAY_RESULT_COLUMNS), None, _summary_html(), "Ready"

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.gray,
        radius_size=gr.themes.sizes.radius_sm,
    )
    with gr.Blocks(theme=theme, css=CSS, title="AIP Dual-Branch Predictor") as demo:
        gr.HTML(
            """
            <header class="app-header">
              <div>
                <h1 class="app-title">AIP Dual-Branch Predictor</h1>
                <p class="app-subtitle">PLM-GRANDE and DDE-TabICLv2 probability fusion</p>
              </div>
              <div class="local-badge">Local inference</div>
            </header>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, elem_classes="panel"):
                gr.HTML('<div class="section-title">Sequence input</div>')
                sequence_text = gr.Textbox(
                    label="FASTA or one sequence per line",
                    lines=11,
                    max_lines=20,
                    placeholder=">peptide_1\nGIKKFLHSAKKFGKAFVGEIMKS",
                )
                with gr.Row():
                    load_button = gr.Button("Load example", variant="secondary")
                    clear_button = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=2, elem_classes="panel"):
                gr.HTML('<div class="section-title">FASTA file</div>')
                upload = gr.UploadButton(
                    "Upload FASTA",
                    file_types=[".fasta", ".fa", ".faa", ".txt"],
                    type="filepath",
                    file_count="single",
                    variant="secondary",
                    size="lg",
                )
                run_button = gr.Button(
                    "Run prediction",
                    variant="primary",
                    size="lg",
                    elem_id="run-button",
                )
                status = gr.Textbox(
                    value="Ready",
                    label="Status",
                    interactive=False,
                    elem_id="status-line",
                )

        summary = gr.HTML(_summary_html())
        gr.HTML('<div class="section-title">Prediction results</div>')
        results = gr.Dataframe(
            value=pd.DataFrame(columns=DISPLAY_RESULT_COLUMNS),
            headers=DISPLAY_RESULT_COLUMNS,
            datatype=["str", "str", "number", "number", "number", "number", "str", "str"],
            interactive=False,
            wrap=True,
            max_height=430,
        )
        download = gr.File(label="Download CSV", interactive=False)

        run_button.click(
            fn=run_prediction,
            inputs=[sequence_text, upload],
            outputs=[results, download, summary, status],
            show_progress="full",
        )
        load_button.click(
            fn=load_example,
            inputs=None,
            outputs=[sequence_text, upload],
            show_progress="hidden",
        )
        clear_button.click(
            fn=clear_all,
            inputs=None,
            outputs=[sequence_text, upload, results, download, summary, status],
            show_progress="hidden",
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local AIP web predictor")
    parser.add_argument("--config", default=None, help="Inference configuration JSON")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    demo = build_app(config_path=args.config, device=args.device)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        show_error=True,
        allowed_paths=[str(ROOT / "results")],
    )


if __name__ == "__main__":
    main()

"""Optional Gradio demo running the full SIMPLE-large on a server (not used by default).

The Space is published as a *static* Space (index.html + demo.js run SIMPLE-small in the
visitor's browser), because hosting Gradio Spaces requires a Hugging Face PRO plan. To use this
app instead: set `sdk: gradio` and `app_file: gradio_app.py` in README.md, add a
requirements.txt with `simple-metric` and `evaluate`, and give the Space a read token as the
secret HF_TOKEN while the model repos are private.
"""

from pathlib import Path

import evaluate
import gradio as gr

# Load the metric script that sits next to this file (no round-trip to the Hub), and run
# one example at start-up so the model is downloaded and any problem shows in the logs.
simple = evaluate.load(str(Path(__file__).with_name("simple.py")), "large")
simple.compute(sources=["The cat sat on the mat."], predictions=["The cat sat."], batch_size=1)


def score(source: str, simplification: str):
    out = simple.compute(sources=[source], predictions=[simplification], batch_size=1)
    mu, sigma = out["scores"][0], out["sigmas"][0]
    verdict = ("successful simplification" if mu > 0.3 else
               "weak simplification" if mu > 0 else
               "unsuccessful (copy, complexification or degenerate)")
    return {"score (mu)": round(mu, 4), "uncertainty (sigma)": round(sigma, 4), "reading": verdict}


demo = gr.Interface(
    fn=score,
    inputs=[gr.Textbox(label="Source sentence", lines=3),
            gr.Textbox(label="Candidate simplification", lines=3)],
    outputs=gr.JSON(label="SIMPLE"),
    title="SIMPLE: reference-free text simplification metric",
    description=("Scores in (-1, 1): +1 = very successful simplification, 0 = copy of the "
                 "source, -1 = unrelated or degenerate text. Sigma is the model's uncertainty."),
    examples=[
        ["The feline perched upon the mat in a state of repose.", "The cat sat on the mat."],
        ["The cat sat on the mat.", "The feline perched upon the mat in a state of repose."],
    ],
)

if __name__ == "__main__":
    demo.launch()

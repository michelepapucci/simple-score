"""SIMPLE as a Hugging Face `evaluate` metric.

    import evaluate
    simple = evaluate.load("mpapucci/simple")             # SIMPLE-large
    simple = evaluate.load("mpapucci/simple", "base")     # or "small"
    out = simple.compute(sources=[...], predictions=[...])

This module is a thin wrapper around the `simple-metric` package
(`pip install simple-metric`), which can also be used directly.
"""

import datasets
import evaluate

_CITATION = """\
@misc{simple-metric,
  title  = {Rethinking Text Simplification Evaluation as Preference Learning},
  note   = {Citation will be added upon publication},
}
"""

_DESCRIPTION = """\
SIMPLE (SImplicity Metric via Preference LEarning) is a reference-free learned metric for
sentence simplification. Given a source sentence and a candidate simplification it returns a
quality score in (-1, 1): +1 for a highly successful simplification, 0 for a verbatim copy of
the source, and -1 for unrelated or degenerate text; negative scores flag unsuccessful
simplifications, including complexifications. With every score it also reports sigma, the
model's own uncertainty (lower = more confident). SIMPLE is a DeBERTa-v3 cross-encoder trained
on pairwise preferences between simplifications (Thurstone model) and calibrated on human
quality ratings. No reference simplifications are needed.
"""

_KWARGS_DESCRIPTION = """
Args:
    sources (list of str): the original (complex) sentences.
    predictions (list of str): the candidate simplifications, aligned with `sources`.
    batch_size (int, optional): batch size for scoring. Defaults to 32.
    device (str, optional): e.g. "cuda", "cuda:1" or "cpu". Defaults to CUDA when available.
Returns:
    simple (float): mean SIMPLE score over all examples, in (-1, 1).
    scores (list of float): per-example scores (mu).
    sigmas (list of float): per-example uncertainty (sigma > 0.05; lower = more confident).
Examples:
    >>> simple = evaluate.load("mpapucci/simple")
    >>> out = simple.compute(
    ...     sources=["The feline perched upon the mat in a state of repose."],
    ...     predictions=["The cat sat on the mat."])
    >>> sorted(out)
    ['scores', 'sigmas', 'simple']
"""

_SIZES = ("small", "base", "large")


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Simple(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {"sources": datasets.Value("string"), "predictions": datasets.Value("string")}
            ),
            codebase_urls=["https://github.com/michelepapucci/simple-metric"],
            reference_urls=["https://huggingface.co/mpapucci/simple-metric-large"],
        )

    def _download_and_prepare(self, dl_manager):
        name = (self.config_name or "default").lower()
        if name not in (*_SIZES, "default"):
            raise KeyError(f"Unknown SIMPLE size {self.config_name!r}; use one of {_SIZES}.")
        self._model_name = "large" if name == "default" else name
        self._scorers = {}

    def _scorer(self, device):
        key = str(device)
        if key not in self._scorers:
            from simple_metric import SIMPLE

            self._scorers[key] = SIMPLE.from_pretrained(self._model_name, device=device)
        return self._scorers[key]

    def _compute(self, sources, predictions, batch_size=32, device=None):
        out = self._scorer(device).score(sources, predictions, batch_size=batch_size)
        return {"simple": out.mean, "scores": out.mu, "sigmas": out.sigma}

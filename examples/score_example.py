"""Score a few simplifications with a released SIMPLE model."""

from simple_score import SIMPLE

sources = [
    "The physician recommended that the patient refrain from strenuous physical activity.",
    "The physician recommended that the patient refrain from strenuous physical activity.",
    "The doctor told the patient not to exercise hard.",
]
candidates = [
    "The doctor told the patient not to exercise hard.",  # good simplification
    "The physician recommended that the patient refrain from strenuous physical activity.",  # copy
    "The physician recommended that the patient refrain from strenuous physical activity.",  # complexification
]

metric = SIMPLE.from_pretrained("large")
out = metric.score(sources, candidates)
for c, mu, sigma in zip(candidates, out.mu, out.sigma):
    print(f"mu={mu:+.3f}  sigma={sigma:.3f}  {c}")

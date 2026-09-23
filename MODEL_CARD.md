# Model Card — MedVision AI Chest X-ray Classifier

This card documents the classification model at the core of MedVision AI, its
measured performance, and its limitations. It follows the model-card convention
for responsible ML documentation.

## Model details

- **Model:** TorchXRayVision DenseNet-121 (`densenet121-res224-all`), used as a
  pretrained model — MedVision builds *around* it and does not train it.
- **Task:** multi-label classification of frontal chest radiographs across 18
  pathologies (each finding scored independently, 0-1).
- **Input:** a single frontal chest X-ray (DICOM or PNG/JPG), normalized to the
  model's expected 224x224 tensor.
- **Output:** a probability per pathology, thresholded present/absent using the
  model's *published* operating points (not tuned by MedVision — see DECISIONS
  D-07).
- **Role in MedVision:** the deterministic core. Its outputs feed explainability
  (GradCAM heatmaps) and literature grounding (PubMed), and — in the clinical
  workflow — are persisted as cases. Generated narrative (a possible VLM layer)
  is kept strictly separate (D-04).

## Intended use

- **Intended:** decision *support* — a fast, transparent, evidence-linked
  first-pass reading for a clinician, especially where radiologist access is
  scarce. The clinician is always in the loop and makes the decision.
- **NOT intended:** autonomous diagnosis; a substitute for a qualified
  radiologist; clinical deployment without validation; any use on non-frontal or
  non-chest images. This is a research/educational prototype.

## Evaluation

Evaluated on the **NIH ChestX-ray14 public sample** (5,606 labeled frontal
radiographs, CC0-licensed), comparing the model's per-pathology probabilities
against the dataset's ground-truth labels.

**How to read these numbers.** The metric is AUC (Area Under the ROC Curve). It
measures how well the model separates a finding being *present* from *absent*,
regardless of where you set the present/absent cutoff — specifically, it is the
probability the model gives a higher score to an X-ray that truly has the finding
than to one that doesn't. **0.5 means random guessing; 1.0 means perfect; roughly
0.7-0.9 is the typical useful range for chest X-ray models.** AUC is chosen
because it is threshold-independent (it measures the model's ranking ability, not
a particular cutoff) and robust to class imbalance (most images don't have any
given finding). "Positive examples" is how many images in the sample actually had
that finding — more positives means a more statistically reliable AUC. AUC is
reported only for findings with at least 30 positive examples; findings below
that, or absent from NIH-14's label set, are not evaluated here.

| Pathology | AUC | Positive examples |
|---|---|---|
| Cardiomegaly | 0.871 | 141 |
| Edema | 0.854 | 118 |
| Effusion | 0.847 | 644 |
| Mass | 0.802 | 284 |
| Consolidation | 0.793 | 226 |
| Atelectasis | 0.776 | 508 |
| Pleural Thickening | 0.755 | 176 |
| Pneumothorax | 0.745 | 271 |
| Fibrosis | 0.733 | 84 |
| Emphysema | 0.705 | 127 |
| Pneumonia | 0.683 | 62 |
| Nodule | 0.674 | 313 |
| Infiltration | 0.672 | 967 |

**Mean AUC (13 reported findings): 0.762.**

These results are consistent with TorchXRayVision's published NIH performance,
which independently validates the MedVision pipeline (preprocessing, inference,
scoring, and label alignment) end to end.

## Limitations

- **Four of the model's 18 findings are not evaluated here** (Lung Lesion,
  Fracture, Lung Opacity, Enlarged Cardiomediastinum): NIH ChestX-ray14 does not
  label them, so no ground truth exists in this dataset. Their real-world
  performance is unknown from this evaluation.
- **Hernia** had too few positive examples (13 < 30) to report a stable AUC.
- **Performance varies by finding.** Effusion, cardiomegaly, and edema are
  detected well (AUC ~0.85+); infiltration, nodule, and pneumonia are
  substantially weaker (~0.67-0.68). Users should weight findings accordingly.
- **Conservative probabilities.** The model expresses calibrated, compressed
  probabilities (rarely near 1.0); its published thresholds are low to match.
  Read a probability relative to its per-finding threshold, not on an absolute
  scale.
- **Dataset scope.** NIH ChestX-ray14 labels are NLP-extracted from radiology
  reports (weak labels, not expert per-image review), and the 5,606-image sample
  is a subset of the full 112,000-image set. Numbers are solid and citable but
  not a substitute for prospective clinical validation.
- **Not validated clinically.** No claim of clinical-grade accuracy or safety.

## Ethical considerations

- **Non-diagnostic, clinician-in-the-loop by design.** The system surfaces
  findings, shows *where* it looked (GradCAM), and links evidence; the clinician
  decides. This bounds the risk of an unvalidated model influencing care.
- **Explainability and grounding** let a clinician audit each prediction rather
  than trust a black box.
- **Data governance.** Development uses public/synthetic data only (D-03);
  patient data in the clinical workflow is synthetic, access-scoped per
  clinician, and minimized.
- **Equity intent.** Designed with low-resource settings in mind, but as design
  intent — not a claim of validated deployment in any setting.

## Reproducing the evaluation

    bash scripts/fetch_sample_data.sh      # dev images (separate from eval data)
    # download the NIH sample (Kaggle: nih-chest-xrays/sample) into data/nih-eval/
    python -m scripts.evaluate --n 5606 --min-pos 30

Results are written to data/nih-eval/evaluation_results.json.

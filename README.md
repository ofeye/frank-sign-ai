# Frank Sign AI Project

> 🇹🇷 **Türkçe**: Frank İşaretinin Kardiyovasküler Hastalıklar ile İlişkisinin Yapay Zeka Araçları Yardımıyla Non-İnvazif Değerlendirilmesi

## 📌 Project Overview

This project develops an AI system to analyze **Frank's Sign** (diagonal earlobe crease) from ear images and predict cardiovascular disease risk. Unlike traditional binary (present/absent) assessment, we quantify **geometric features** (length, curvature, depth, thickness, localization) using deep learning segmentation and combine them with clinical data for risk prediction.

### Key Innovation
- **Quantitative Analysis**: Moving from subjective "yes/no" to measurable geometric features
- **Explainable AI (XAI)**: SHAP + Grad-CAM for transparent clinical decisions
- **Non-invasive & Low-cost**: Using standard camera images

## 🚀 Quick Start

```bash
# Clone and setup
cd /Applications/Codes/84. AI_FrankSign
pip install -e .

# Parse CVAT annotations (package entrypoint)
franksign-parse --input data/annotations/annotations.xml

# Validate clinical CSV (sample or production)
python scripts/validate_data.py --clinical "FS - AI - Sayfa1.csv"
# Optional: include CVAT structural checks
python scripts/validate_data.py --clinical "FS - AI - Sayfa1.csv" \
  --annotations data/annotations/annotations.xml

# Train/Evaluate (placeholders for now)
franksign-train --config configs/default.yaml
franksign-eval --config configs/default.yaml
```


## 📁 Project Structure

```
├── src/franksign/      # Main Python package
│   ├── data/           # Data loading, parsing, preprocessing
│   ├── models/         # Neural network architectures
│   ├── training/       # Training loops
│   ├── evaluation/     # Metrics (Dice, IoU, etc.)
│   └── utils/          # Visualization, helpers
├── configs/            # YAML configuration files
├── data/               # Datasets (raw, processed, splits)
├── models/             # Saved model checkpoints
├── experiments/        # Experiment logs and results
├── scripts/            # CLI scripts
└── docs/               # Additional documentation
```

## 📊 Current Status

| Phase | Status | Timeline |
|-------|--------|----------|
| Data Collection | 🟡 In Progress | Month 1-12 |
| Annotation (CVAT) | 🟡 Pilot (121 images) | Month 1-12 |
| Feature Extraction | 🟢 Basic skeleton ready | Month 6-15 |
| Model Development | ⚪ Planned | Month 9-16 |
| Clinical Validation | ⚪ Planned | Month 14-18 |

## 🔧 Technology Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch
- **Image Processing**: OpenCV, scikit-image
- **Annotation**: CVAT (app.cvat.ai)
- **XAI**: SHAP, Grad-CAM++

## 📚 References

- TÜBİTAK 1005 Project Proposal: `Project_Main.md`
- Annotation Schema: `docs/data_schema.md`
- AI Agent Guidelines: `AGENTS.md`

## 👥 Team

- Dr. Reha TÜRK (Project Lead)
- Dr. Eda AKSOY
- Dr. Muhsin SARIHAN
- Dr. Öğr. Üyesi Tolga BERBER
- Osman Furkan YILMAZ

---
*Karadeniz Teknik Üniversitesi - 2025-2026*

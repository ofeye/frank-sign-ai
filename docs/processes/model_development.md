# Process: Model Development

## Status: ⚪ Planned
## Last Updated: 2026-01-13
## Owner: TBD

---

## Current Objectives

- [ ] Baseline models (classical CV)
- [ ] U-Net implementation
- [ ] Attention U-Net implementation
- [ ] XAI integration (Grad-CAM, SHAP)
- [ ] Model comparison framework

---

## Progress Log

### 2026-01-13
**Status:** Waiting for Data Pipeline completion

**Prerequisites:**
- [x] Feature extraction working
- [x] Clinical data loader working
- [ ] PyTorch Dataset class (blocking)
- [ ] Image preprocessing pipeline

---

## Dependencies

**Depends on:**
- Data Pipeline (Dataset class required)
- Environment setup (PyTorch installation)

**Blocking:**
- Evaluation & Validation
- Clinical Integration

---

## Planned Architecture

```
Input Image (256x256x3)
    ↓
Encoder (ResNet34, pretrained)
    ↓
Attention Gates
    ↓
Decoder
    ↓
Output Mask (256x256x3)
    - Background
    - Frank Sign Line
    - Frank Sign Region
```

---

## Key Decisions (Pending)

1. ❓ Start with U-Net or Attention U-Net?
2. ❓ Multi-task learning (segmentation + classification)?
3. ❓ Transfer learning from medical imaging datasets?

---

## Files Involved (Planned)

| File | Purpose |
|------|---------|
| `src/franksign/models/unet.py` | U-Net architecture |
| `src/franksign/models/attention_unet.py` | Attention U-Net |
| `src/franksign/training/trainer.py` | Training loop |
| `src/franksign/evaluation/metrics.py` | Dice, IoU, Hausdorff |

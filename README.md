# FracAtlas Bone Fracture Segmentation - Reproducibility Study


## 🔍 Overview

This repository contains a **comprehensive reproducibility study** of the FracAtlas bone fracture segmentation paper. We attempted to replicate published results and conducted systematic experiments with data quality improvements and hyperparameter optimization.

### Key Findings

- 🚨 **Reproducibility Crisis**: Achieved only **60-65% of published performance** (38.7% gap)
- ✅ **Dataset Contribution**: Discovered and fixed annotation errors in public FracAtlas dataset
- 📊 **Systematic Analysis**: Tested multiple configurations (image sizes, training durations)
- 📈 **Transparent Reporting**: Complete methodology, metrics, and negative results documented

---

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Performance Comparison](#performance-comparison)
- [Models](#models)
- [Dataset Notes](#dataset-notes)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis & Results](#analysis--results)
- [Key Contributions](#key-contributions)
- [Citation](#citation)

---

## 📁 Project Structure

```
data_segm/
│
├── analysis/                          # Comprehensive reproducibility analysis
│   ├── plots/                         # 6 publication-quality visualizations
│   │   ├── 01_metrics_comparison_bars.png
│   │   ├── 02_performance_gap_vs_baseline.png
│   │   ├── 03_radar_chart.png
│   │   ├── 04_metrics_heatmap.png
│   │   ├── 05_v7_vs_v8_detailed.png
│   │   └── 06_suspicious_performance_gap.png
│   │
│   ├── results/                       # CSV files with all metrics
│   │   ├── all_models_combined.csv    # All 4 models comparison
│   │   ├── performance_gaps_analysis.csv
│   │   ├── v7_vs_v8_comparison.csv
│   │   ├── statistical_analysis.csv
│   │   └── executive_summary.txt
│   │
│   ├── report/                        # Final technical report
│   │   └── COMPREHENSIVE_TECHNICAL_REPORT.txt  # 300+ lines
│   │
│   ├── scripts/                       # Analysis scripts
│   │   ├── setup.py                   # Setup directories
│   │   ├── eval_1st&2nd.py           # Evaluate original + reproduction
│   │   ├── eval_3rd_4th.py           # Load v7 + v8 metrics
│   │   ├── combine_analyse.py        # Statistical analysis
│   │   ├── generate_vis.py           # Create visualizations
│   │   └── generate_report.py        # Generate final report
│   │
│   └── config_comparison/             # Model configurations
│       ├── v7_config_summary.txt
│       └── v8_config_summary.txt
│
├── models/                            # Model weights
│   ├── yolov8_segmentation_fractureAtlas.pt  # Original paper model
│   └── v7_best_mine.pt               # Our best model (v7)
│
├── reproduce/                         # Reproduction attempt
│   └── runs/segment/train/
│       ├── weights/
│       │   ├── best.pt               # Reproduction model weights
│       │   └── last.pt
│       ├── results.csv               # Training metrics
│       ├── args.yaml                 # Training arguments
│       ├── confusion_matrix.png
│       ├── BoxF1_curve.png
│       ├── MaskF1_curve.png
│       └── (various training plots and validation images)
│
├── v7/                                # Our v7 model (imgsz=800, epochs=100)
│   ├── bone_fracture_model/
│   │   ├── weights/
│   │   │   ├── best.pt              # v7 model weights
│   │   │   └── last.pt
│   │   ├── results.csv              # Training history
│   │   ├── args.yaml                # v7 configuration
│   │   └── (performance curves, confusion matrices)
│   │
│   ├── results/                      # v7 evaluation results
│   │   ├── comprehensive_metrics.csv # All v7 metrics
│   │   ├── predictions/             # Test predictions
│   │   ├── comparisons/             # Ground truth vs predictions
│   │   └── (various plots)
│   │
│   └── data.yaml                     # Dataset configuration
│
├── v8/                                # Our v8 model (imgsz=640, epochs=120)
│   ├── bone_fracture_model/
│   │   ├── weights/
│   │   │   ├── best.pt              # v8 model weights
│   │   │   └── last.pt
│   │   ├── results.csv              # Training history
│   │   ├── args.yaml                # v8 configuration
│   │   └── (performance curves, confusion matrices)
│   │
│   └── results/                      # v8 evaluation results
│       ├── comprehensive_metrics.csv # All v8 metrics
│       ├── predictions/             # Test predictions
│       ├── comparisons/             # Ground truth vs predictions
│       └── (various plots)
│
├── data.yaml                          # Main dataset configuration
├── frac-atlas.ipynb                  # Main training notebook
├── Train_8s_ta3hm_homa.ipynb        # Additional training experiments
└── .gitignore                        # Git ignore file

```

---

## 📊 Performance Comparison

### Results Summary

| Model | Architecture | ImgSz | Epochs | mAP50 | F1-Score | Dice | Status |
|-------|-------------|-------|--------|-------|----------|------|--------|
| **Original Paper** | YOLOv8 | ? | ? | **0.8718** | **0.8281** | **0.8281** | ⚠️ Suspicious |
| **Reproduction** | YOLOv8/11 | 640 | 100 | 0.5278 | 0.5436 | 0.5436 | 39.5% gap |
| **Our v7** | YOLO11m-seg | 800 | 100 | **0.5343** | **0.6083** | **0.6083** | **Our Best** |
| **Our v8** | YOLO11m-seg | 640 | 120 | 0.5296 | 0.5735 | 0.5735 | Faster |

### Key Observations

- ❌ **Reproducibility Failure**: 38.7% performance gap vs original paper
- ✅ **Our Best Model**: v7 achieves mAP50=0.5343 (best we could achieve)
- 🚀 **Speed vs Quality**: v8 is 33% faster inference (15.5ms vs 23.1ms)
- 📉 **Extended Training**: 120 epochs (v8) didn't significantly improve over 100 (v7)

---

## 🤖 Models

### 1. Original Paper Model (Suspicious)
- **File**: `models/yolov8_segmentation_fractureAtlas.pt`
- **Source**: FracAtlas paper authors
- **Performance**: mAP50=0.8718, F1=0.8281
- **Status**: ⚠️ We couldn't reproduce these results

### 2. Reproduction Attempt
- **File**: `reproduce/runs/segment/train/weights/best.pt`
- **Goal**: Replicate their methodology
- **Performance**: mAP50=0.5278, F1=0.5436
- **Gap**: 39.5% below original

### 3. Our v7 Model (Best Performance)
- **File**: `v7/bone_fracture_model/weights/best.pt` or `models/v7_best_mine.pt`
- **Configuration**: YOLO11m-seg, 800px, 100 epochs
- **Performance**: mAP50=0.5343, F1=0.6083
- **Features**: Corrected annotations + augmentation

### 4. Our v8 Model (Optimized Speed)
- **File**: `v8/bone_fracture_model/weights/best.pt`
- **Configuration**: YOLO11m-seg, 640px, 120 epochs
- **Performance**: mAP50=0.5296, F1=0.5735
- **Features**: Faster inference, slightly lower quality

---

## 📦 Dataset Notes

### ⚠️ Important: Data Not Included

The following directories are **gitignored** and must be downloaded separately:

```
data/          # Your processed dataset
FracAtlas/     # Original FracAtlas dataset
venv/          # Python virtual environment
*.zip          # Compressed archives
```

### Download FracAtlas Dataset

1. **Original Dataset**: [FracAtlas Official Repository](https://figshare.com/articles/dataset/The_dataset/22363012)
2. **Dataset Contents**:
   - `FracAtlas/images/Fractured/` - X-ray images
   - `FracAtlas/Annotations/COCO JSON/` - COCO format annotations
   - `FracAtlas/Annotations/PASCAL VOC/` - VOC format annotations

### ⚠️ Annotation Quality Issues

**CRITICAL FINDING**: The public YOLO format annotations contain errors:
- Missing polygon points in segmentation masks
- Incomplete coordinate sequences
- Format inconsistencies

**Our Solution**: We corrected annotations by converting from COCO JSON format. See `frac-atlas.ipynb` for our correction pipeline.

### Dataset Structure After Download

```
FracAtlas/
├── images/
│   └── Fractured/          # X-ray images
├── Annotations/
│   ├── COCO JSON/
│   │   └── COCO_fracture_masks.json
│   └── PASCAL VOC/
│       └── *.xml files
└── Utilities/
    └── Fracture Split/
        ├── train.csv
        ├── test.csv
        └── valid.csv
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd data_segm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics pandas numpy matplotlib seaborn opencv-python albumentations

# Download FracAtlas dataset (place in ./FracAtlas/)
# See Dataset Notes section above
```

---

## 💻 Usage

### 1. Train a Model

```bash
# Train with v7 configuration (800px, 100 epochs)
python -c "from ultralytics import YOLO; model = YOLO('yolo11m-seg.pt'); model.train(data='data.yaml', imgsz=800, epochs=100, batch=16)"

# Train with v8 configuration (640px, 120 epochs)
python -c "from ultralytics import YOLO; model = YOLO('yolo11m-seg.pt'); model.train(data='data.yaml', imgsz=640, epochs=120, batch=16)"
```

Or use the notebook:
```bash
jupyter notebook frac-atlas.ipynb
```

### 2. Evaluate Models

```bash
# Evaluate original paper model
cd analysis/scripts
python eval_1st&2nd.py

# Load v7 and v8 metrics
python eval_3rd_4th.py
```

### 3. Generate Analysis

```bash
# Combine all metrics and perform statistical analysis
python combine_analyse.py

# Generate visualizations
python generate_vis.py

# Create comprehensive report
python generate_report.py
```

### 4. View Results

```bash
# Check analysis outputs
ls -lh ../results/
ls -lh ../plots/
cat ../report/COMPREHENSIVE_TECHNICAL_REPORT.txt
```

---

## 📈 Analysis & Results

### Generated Outputs

After running the analysis scripts, you'll have:

#### 1. **Metrics & Statistics** (`analysis/results/`)
- `all_models_combined.csv` - All 4 models comparison
- `performance_gaps_analysis.csv` - Gap analysis vs original paper
- `v7_vs_v8_comparison.csv` - Our models comparison
- `statistical_analysis.csv` - Cohen's d effect sizes
- `executive_summary.txt` - Key findings summary

#### 2. **Visualizations** (`analysis/plots/`)
- `01_metrics_comparison_bars.png` - Bar charts across models
- `02_performance_gap_vs_baseline.png` - Gap visualization
- `03_radar_chart.png` - Performance profile
- `04_metrics_heatmap.png` - All metrics heatmap
- `05_v7_vs_v8_detailed.png` - v7 vs v8 analysis
- `06_suspicious_performance_gap.png` - Reproducibility gap

#### 3. **Comprehensive Report** (`analysis/report/`)
- `COMPREHENSIVE_TECHNICAL_REPORT.txt` - 300+ line detailed analysis

### Key Metrics Explained

- **mAP50**: Mean Average Precision at IoU=0.5 (primary metric)
- **mAP50-95**: mAP averaged over IoU thresholds 0.5 to 0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Dice Coefficient**: 2×(Intersection) / (|A| + |B|), same as F1 for segmentation

---

## 🎯 Key Contributions

### 1. Annotation Error Discovery & Correction
- ✅ Found missing polygon points in public YOLO annotations
- ✅ Implemented conversion pipeline from COCO JSON
- ✅ Validated all annotation files
- ✅ Documented correction process for community

### 2. Systematic Hyperparameter Optimization
- ✅ Tested image sizes: 640px vs 800px
- ✅ Tested training durations: 100 vs 120 epochs
- ✅ Documented speed/quality tradeoffs
- ✅ All configurations saved for reproducibility

### 3. Comprehensive Data Augmentation
- ✅ Geometric: Flips, rotations
- ✅ Photometric: HSV, brightness, contrast
- ✅ Advanced: RandAugment, mosaic, erasing
- ✅ ~5x dataset size increase

### 4. Transparent Methodology
- ✅ All hyperparameters documented
- ✅ Complete training configurations saved
- ✅ Evaluation protocol clearly specified
- ✅ Results in standard formats (CSV, JSON)

### 5. Reproducibility Analysis
- ✅ Critical evaluation of original paper
- ✅ Statistical significance testing
- ✅ Performance gap quantification
- ✅ Honest negative results reporting

---

## 🔬 Research Findings

### Critical Observations

1. **Severe Reproducibility Failure**
   - 38.7% performance gap vs published results
   - Effect size: LARGE (Cohen's d > 6.0)
   - Not due to random variation

2. **Suspicious Aspects of Original Paper**
   - Incomplete methodology disclosure
   - No code or detailed hyperparameters released
   - Unrealistic performance (mAP50=0.87 very high for medical segmentation)
   - No ablation studies or error analysis

3. **Dataset Quality Issues**
   - Public YOLO annotations contain errors
   - Missing annotation points affect training
   - Correction necessary for proper training

4. **Our Model Performance**
   - Best achievable: mAP50=0.5343 (v7)
   - Consistent across multiple configurations
   - Systematic optimization didn't close gap

---

## 📚 Citation

If you use this work or find our annotation corrections useful, please cite:

```bibtex
@misc{fracatlas_reproducibility,
  author = {Your Name},
  title = {FracAtlas Bone Fracture Segmentation: A Reproducibility Study},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/data_segm}
}
```

Original FracAtlas paper:
```bibtex
@article{fracatlas2022,
  title={FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs},
  author={[Original Authors]},
  journal={[Journal Name]},
  year={2022}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

We welcome contributions, especially:
- Alternative reproduction attempts
- Different model architectures
- Additional data quality improvements
- Hyperparameter optimization strategies

Please open an issue or pull request!

---

## ⚠️ Disclaimer

This is an independent reproducibility study. The findings represent our best efforts to replicate published results. The performance gap may be due to:
- Unreported methodology details
- Different dataset versions
- Annotation quality differences
- Hardware/software environment variations

We encourage others to attempt reproduction and share their findings.

---

## 📧 Contact

For questions or collaborations:
- Open an issue on GitHub
- Email: [your.email@example.com]

---

## 🙏 Acknowledgments

- FracAtlas dataset creators
- Ultralytics YOLO team
- Medical imaging research community

---

## 📖 Additional Resources

- [FracAtlas Dataset](https://figshare.com/articles/dataset/The_dataset/22363012)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Our Comprehensive Report](analysis/report/COMPREHENSIVE_TECHNICAL_REPORT.txt)

---

**Note**: This repository demonstrates the importance of reproducibility in medical AI research. While we could not replicate the original paper's results, our systematic approach and dataset corrections provide value to the community.
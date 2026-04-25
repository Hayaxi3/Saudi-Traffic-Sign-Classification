# Saudi-Traffic-Sign-Classification
A computer vision project that trains and compares **four vision transformer architectures** on a Saudi traffic sign dataset, then deploys an interactive Gradio demo for real-time inference.

| Model | Architecture |
|:---|:---|
| ViT | `vit_base_patch16_224` |
| Swin | `swin_base_patch4_window7_224` |
| MobileViT | `mobilevit_s` |
| EfficientViT | `efficientvit_b1` |

All models are loaded with pretrained ImageNet weights via the `timm` library, with their classification heads replaced for 24-class output.

## Dataset
- **Source:** [Roboflow — Saudi Traffic Sign Dataset](https://app.roboflow.com/hayas-workspace-vivrw/saudi-traffic-sign-dataset/2)
- **Classes:** 24 traffic sign categories
- **Splits:** Train / Validation / Test
- **Preprocessing:** Auto-Orient, Resized to 224×224, normalized with ImageNet mean and std

## Notebook Sections
| # | Section | Description |
|:--|:--------|:------------|
| 1 | Environment Setup | Install dependencies and import all libraries |
| 2 | Dataset | Download dataset from Roboflow |
| 3 | Configuration & Reproducibility | Hyperparameters and random seeds |
| 4 | Data Preprocessing & DataLoaders | Transforms, datasets, and DataLoaders |
| 5 | Model Definitions | Build all five transformer models |
| 6 | Model Training | Training loop with early stopping and LR scheduling |
| 7 | Model Evaluation | Test-set classification reports |
| 8 | Model Visualization | Accuracy curves, confusion matrices, bar charts |
| 9 | Gradio UI | Interactive inference demo |

## Training Details
- **Optimizer:** AdamW (weight decay = 1e-4)
- **LR Scheduler:** ReduceLROnPlateau (patience = 3, factor = 0.5)
- **Loss:** CrossEntropyLoss
- **Early Stopping:** 7 epochs patience
- **Epochs:** 25 max
- **Batch size:** 32
- **Image size:** 224 × 224
- **Seed:** 42

## Gradio Demo
After training and evaluation, Section 9 launches an interactive UI with four tabs:
- **Single Model** — Upload an image, pick a model, and get top-K predictions with a confidence chart
- **Compare All Models** — Run all five models on the same image side by side
- **Model Info** — Parameter counts and class label reference table
- **Performance Results** — Test-set accuracy, precision, recall, and F1-score for all models

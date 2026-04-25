# Saudi-Traffic-Sign-Classification
A computer vision project that trains and compares **four vision transformer architectures** on a Saudi traffic sign dataset, then deploys an interactive Gradio demo for real-time inference.

| Model | Architecture |
|:---|:---|
| ViT | `vit_base_patch16_224` |
| Swin | `swin_base_patch4_window7_224` |
| MobileViT | `mobilevit_s` |
| EfficientViT | `efficientvit_b1` |

All models are loaded with pretrained ImageNet weights via the `timm` library, with their classification heads replaced for 24-class output.

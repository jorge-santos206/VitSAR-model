# Repository Structure

```text
attentionSarModel/
│
├── architectures/
│   └── deeplabv3.py
│       # DeepLabV3+ + Channel Attention model and training
│
├── data/
│   ├── dataset/
│   │   ├── images/
│   │   │   └── sample_0.npy ... sample_N.npy
│   │   │
│   │   └── masks/
│   │       └── mask_0.npy ... mask_N.npy
│   │
│   ├── pre-processing/
│   │   ├── generating-training-data.py
│   │   │   # Complete preprocessing pipeline
│   │   │
│   │   ├── image-sar1200x900.tif
│   │   │   # Original AirSAR San Francisco image
│   │   │
│   │   ├── SF-AIRSAR-label3d.png
│   │   │   # Segmentation labels (6 classes)
│   │   │
│   │   ├── cloudPottier.py
│   │   │   # Cloude-Pottier decomposition
│   │   │
│   │   ├── freemanDecomposition.py
│   │   │   # Freeman-Durden decomposition
│   │   │
│   │   ├── copolarization.py
│   │   │   # Copolarization features
│   │   │
│   │   ├── crossPolarization.py
│   │   │   # Cross-polarization features
│   │   │
│   │   ├── huynenDecomposition.py
│   │   │   # Huynen decomposition
│   │   │
│   │   ├── glcm.py
│   │   │   # Texture features (GLCM)
│   │   │
│   │   ├── edgyLineEnergy.py
│   │   │   # Edge and line features
│   │   │
│   │   ├── leeFilter.py
│   │   │   # Speckle reduction filter
│   │   │
│   │   └── span.py
│   │       # SPAN feature extraction
│   │
│   └── r-files/
│       ├── AirSAR_SanFrancisc_Enxu.RData
│       │   # Raw SAR data
│       │
│       └── creating-tiff-arch.R
│           # Generates the .tif image
│
├── docs/
│   └── images/
│       # README images
│
└── README.md
```
# Model Architecture

```text
Input (16×16×51)
        ↓
Backbone
(Conv5×5 → MaxPool → Conv3×3)
        ↓

┌────────────────────────────────┐
│ ASPP Branch                    │
│ dilation rates = 1, 2, 4, 6   │
└────────────────────────────────┘

                +
                
┌────────────────────────────────┐
│ Low-Level Branch               │
│ Conv1×1 → 48 filters           │
└────────────────────────────────┘

        ↓
     UpSampling
        ↓

────────── Concat ──────────
        ↓

Decoder
(Conv3×3)
        ↓

🔥 Channel Attention Block
        ↓

Final Classifier
(Conv1×1 + Softmax)
        ↓

Output
(16×16×6 classes)
```

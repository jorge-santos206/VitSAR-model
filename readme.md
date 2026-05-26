Estrutura do Repositório:
attentionSarModel/
│
├── architectures/
│   └── deeplabv3.py              # Modelo DeepLabV3+ + Channel Attention + treinamento
│
├── data/
│   ├── dataset/
│   │   ├── images/               # Patches de imagem SAR (sample_0.npy ... sample_N.npy)
│   │   └── masks/                # Masks de segmentação (mask_0.npy ... mask_N.npy)
│   │
│   ├── pre-processing/
│   │   ├── generating-training-data.py   # Pipeline completo de geração de dados
│   │   ├── image-sar1200x900.tif         # Imagem SAR original AirSAR São Francisco
│   │   ├── SF-AIRSAR-label3d.png         # Mapa de rótulos (6 classes)
│   │   ├── cloudPottier.py               # Decomposição Cloude-Pottier (H, α, A)
│   │   ├── freemanDecomposition.py       # Decomposição Freeman-Durden
│   │   ├── copolarization.py             # Feature de copolarização
│   │   ├── crossPolarization.py          # Feature de crosspolarização
│   │   ├── huynenDecomposition.py        # Decomposição Huynen
│   │   ├── glcm.py                       # Features de textura GLCM
│   │   ├── edgyLineEnergy.py             # Features de borda e linha
│   │   ├── leeFilter.py                  # Filtro de speckle Lee
│   │   └── span.py                       # SPAN (potência total)
│   │
│   └── r-files/
│       ├── AirSAR_SanFrancisc_Enxu.RData  # Dados brutos em formato R
│       └── creating-tiff-arch.R            # Script R para geração do .tif
│
├── docs/
│   └── images/                   # Imagens para o README (adicione aqui)
│
└── README.md



Arquitetura do Modelo
O modelo DeeplabV3Plus é composto pelos seguintes módulos em sequência:

Entrada (16×16×51)
       ↓
   Backbone (Conv5×5 → MaxPool → Conv3×3)
       ↓
  ┌────────────────────────────────┐
  │  Branch ASPP (dilation 1,2,4,6)│   Branch Low-Level (Conv1×1, 48 filtros)
  └────────────────────────────────┘
       ↓ UpSampling                      ↓ UpSampling
       └──────────── Concat ─────────────┘
                      ↓
                  Decoder (Conv3×3)
                      ↓
           🔥 Channel Attention Block
                      ↓
           Classificador Final (Conv1×1 Softmax)
                      ↓
          Saída (16×16×6 classes)

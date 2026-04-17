# Dataset — Vietnamese Fake News Detection

## Nguồn dữ liệu

### 1. ViFactCheck (AAAI 2025)
- **Mô tả**: 7,232 claim-evidence pairs cho fact-checking tiếng Việt
- **Source**: [HuggingFace: ngtram/vifactcheck](https://huggingface.co/datasets/ngtram/vifactcheck)
- **Paper**: ViFactCheck: A New Benchmark Dataset and Methods for Multi-domain News Fact-Checking in Vietnamese
- **Labels**: SUPPORTED, REFUTED, NOT ENOUGH INFO
- **Download**: `python dataset/download_datasets.py`

### 2. ReINTEL (VLSP 2020)
- **Mô tả**: Reliable Intelligence Identification on Vietnamese Social Network
- **Source**: VLSP 2020 Shared Task
- **Labels**: reliable (0) / unreliable (1)

### 3. Self-collected
- **Tin thật**: VnExpress.net, Tuổi Trẻ Online (báo chính thống)
- **Tin giả**: Fact-check sources, public fake news databases
- **Số lượng mục tiêu**: 300-500 mẫu
- **Preprocessing**: underthesea word tokenization

## Cấu trúc folder

```
dataset/
├── raw/                    # Dữ liệu gốc (gitignored)
│   ├── vifactcheck/
│   ├── reintel/
│   └── collected/
├── processed/              # Đã clean & merge
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── knowledge_base/    # Vector store cho RAG
└── statistics/             # EDA results
```

## Sử dụng

```bash
# Bước 1: Download datasets
python dataset/download_datasets.py

# Bước 2: Thu thập thêm từ báo VN
python dataset/collect_news.py

# Bước 3: Preprocessing
python dataset/preprocess_vietnamese.py

# Bước 4: Merge & split
python dataset/merge_datasets.py

# Bước 5: Trích xuất stylistic features
python dataset/feature_extraction.py
```

## License
- ViFactCheck: Research use (AAAI 2025)
- ReINTEL: VLSP 2020 shared task license
- Self-collected: Public news articles

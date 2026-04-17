# COMP560 Project: Face Recognition

## Task

Build a system that determines whether two face templates belong to the same person. Given template pairs, produce a similarity score for each pair.

## Datasets

| Dataset | Images | Templates | Pairs | Image Size |
|---------|--------|-----------|-------|------------|
| dataset_a | ~228K | ~12K | ~8M | 112x112 |
| dataset_b | ~469K | ~23K | ~15.7M | 112x112 |

Data format (Parquet):
- `test.parquet`: image metadata (image_path, template_id, media_id, landmarks, detection_score)
- `pairs.parquet`: verification pairs (template_id_1, template_id_2, label)

Each template consists of one or more face images. The evaluation protocol aggregates image-level features into template-level features, then computes cosine similarity between template pairs. See the baseline script for details.

## Submission Format

Submit a CSV file per dataset with columns:

```csv
template_id_1,template_id_2,score
1,11065,0.732
1,11066,0.215
...
```

- `template_id_1`, `template_id_2`: template pair IDs (must match pairs.parquet)
- `score`: similarity score (higher = more likely same identity)

## Evaluation

```bash
# Evaluate on dataset_a
python evaluate.py --student_id YOUR_ID --prediction predictions/dataset_a.csv --datasets dataset_a

# Evaluate on both (pass a directory containing dataset_a.csv and dataset_b.csv)
python evaluate.py --student_id YOUR_ID --prediction predictions/ --datasets dataset_a dataset_b
```

## Baseline

Generate baseline predictions using pretrained ResNet50:

```bash
python models/resnet_baseline.py --dataset_root ./datasets/dataset_a --output predictions/dataset_a.csv
python models/resnet_baseline.py --dataset_root ./datasets/dataset_b --output predictions/dataset_b.csv
```

## Training Example

Train a ResNet50 model with ArcFace or triplet loss:

```bash
python train_example.py --data_root ./datasets/dataset_a --loss arcface --epochs 20
python train_example.py --data_root ./datasets/dataset_a --loss triplet --epochs 20
```

Generate predictions from a trained checkpoint:

```bash
python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_a --output predictions/dataset_a.csv
python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_b --output predictions/dataset_b.csv
```

## Metrics

- **TAR@FAR=1e-6, 1e-5, 1e-4, 1e-3**: True Accept Rate at various False Accept Rates
- **AUC**: Area Under the ROC Curve

## Grading

- 40% Performance (TAR@FAR metrics)
- 30% Efficiency (model design, embedding dimension)
- 30% Report

## Directory Structure

```
project-fr/
├── datasets/
│   ├── dataset_a/
│   │   ├── images/          # Face images
│   │   ├── test.parquet     # Image metadata
│   │   └── pairs.parquet    # Verification pairs
│   └── dataset_b/
│       ├── images/
│       ├── test.parquet
│       └── pairs.parquet
├── models/
│   └── resnet_baseline.py   # Baseline prediction generator
├── evaluate.py              # Evaluation script
├── train_example.py         # Training example
└── results/                 # Output directory
```

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
python evaluate.py --student_id YOUR_ID --prediction predictions/dataset_a.csv --datasets dataset_a --acknowledge_benchmark_labels

# Evaluate on both (pass a directory containing dataset_a.csv and dataset_b.csv)
python evaluate.py --student_id YOUR_ID --prediction predictions/ --datasets dataset_a dataset_b --acknowledge_benchmark_labels
```

## Baseline

Generate baseline predictions using the pretrained ResNet50 baseline:

```bash
python models/resnet_baseline.py --dataset_root ./datasets/dataset_a --output predictions/dataset_a.csv
python models/resnet_baseline.py --dataset_root ./datasets/dataset_b --output predictions/dataset_b.csv
```

Generate predictions from an exact MobileFaceNet checkpoint using the same CSV output and evaluation path:

```bash
python models/resnet_baseline.py --dataset_root ./datasets/dataset_a --output predictions/dataset_a_mobilefacenet.csv --backbone mobilefacenet --checkpoint ./checkpoints_mobilefacenet/best_model.pth
python models/resnet_baseline.py --dataset_root ./datasets/dataset_b --output predictions/dataset_b_mobilefacenet.csv --backbone mobilefacenet --checkpoint ./checkpoints_mobilefacenet/best_model.pth
```

## Training Example

Train a ResNet50 model with ArcFace or triplet loss:

```bash
python train_example.py --data_root ./datasets/dataset_a --loss arcface --epochs 20
python train_example.py --data_root ./datasets/dataset_a --loss triplet --epochs 20
```

Train the same pipeline with the exact MobileFaceNet architecture while preserving the same validation metrics (`TAR@FAR` and `AUC`):

```bash
python train_example.py --data_root ./datasets/dataset_a --loss arcface --epochs 20 --backbone mobilefacenet --embedding_dim 128
python train_example.py --data_root ./datasets/dataset_a --loss triplet --epochs 20 --backbone mobilefacenet --embedding_dim 128
```

Run a Colab-friendly MobileFaceNet sweep that samples the most important training knobs for the exact 128-D backbone and writes ranked CSV/JSON summaries:

```bash
python run_mobilefacenet_sweep.py --data_root ./datasets/dataset_a --train_metadata ./datasets/dataset_a/splits/val_15_seed42/train_metadata.parquet --val_metadata ./datasets/dataset_a/splits/val_15_seed42/val_metadata.parquet --val_pairs ./datasets/dataset_a/splits/val_15_seed42/val_pairs.parquet --preset standard --amp --device cuda
```

The sweep varies:
- learning rate
- weight decay
- batch size
- ArcFace margin (`arcface_m`)
- ArcFace scale (`arcface_s`)
- warmup epochs

Recommended sweep budgets on a single Colab GPU:
- `quick`: 6 trials x 4 epochs for smoke testing
- `standard`: 10 trials x 5 epochs for a useful first pass
- `full`: 16 trials x 6 epochs when you can spend more GPU time

After the sweep, retrain the best 1-2 configurations for roughly 12-20 epochs to get a more stable final ranking.

Generate predictions from a trained checkpoint:

```bash
python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_a --output predictions/dataset_a.csv
python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_b --output predictions/dataset_b.csv
```

Benchmark either backbone and save performance plus efficiency metrics to JSON:

```bash
python run_baseline_benchmark.py --dataset_root ./datasets/dataset_a --output predictions/dataset_a.csv --metrics_output results/dataset_a_resnet50_metrics.json --backbone resnet50
python run_baseline_benchmark.py --dataset_root ./datasets/dataset_a --output predictions/dataset_a_mobilefacenet.csv --metrics_output results/dataset_a_mobilefacenet_metrics.json --backbone mobilefacenet --checkpoint ./checkpoints_mobilefacenet/best_model.pth
```

## Colab / Notebook Usage

In a notebook cell, run the same scripts with `!python`. Because both backbones still write the same prediction CSV format and use `evaluate.py`, they report the same statistics as the existing ResNet50 path:

```bash
!python train_example.py --data_root ./datasets/dataset_a --train_metadata ./datasets/dataset_a/splits/val_15_seed42/train_metadata.parquet --val_metadata ./datasets/dataset_a/splits/val_15_seed42/val_metadata.parquet --val_pairs ./datasets/dataset_a/splits/val_15_seed42/val_pairs.parquet --loss arcface --epochs 5 --backbone mobilefacenet --embedding_dim 128 --device cuda

!python run_baseline_benchmark.py --dataset_root ./datasets/dataset_a --output predictions/dataset_a_mobilefacenet.csv --metrics_output results/dataset_a_mobilefacenet_metrics.json --backbone mobilefacenet --checkpoint ./checkpoints_mobilefacenet/best_model.pth --device cuda

!python run_mobilefacenet_sweep.py --data_root ./datasets/dataset_a --train_metadata ./datasets/dataset_a/splits/val_15_seed42/train_metadata.parquet --val_metadata ./datasets/dataset_a/splits/val_15_seed42/val_metadata.parquet --val_pairs ./datasets/dataset_a/splits/val_15_seed42/val_pairs.parquet --preset standard --amp --device cuda
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

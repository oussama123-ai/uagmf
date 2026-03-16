# UAG-MF Data Format

## Raw data layout (input to preprocessing)

```
data/raw/
├── biovid/subject_001/clip_001/{frames/, hrv.csv, spo2.csv, resp.csv, labels.csv}
├── unbc/subject_001/clip_001/{frames/, labels.csv}
├── emopain/subject_001/clip_001/{frames/, hrv.csv, resp.csv, labels.csv}
└── mdnpl/neonate_001/clip_001/{frames/, labels.csv}   # OOD eval only
```

## Feature directory layout (output of preprocessing)

```
data/features/
└── subject_001/clip_001/
    ├── video.npy        (T, 3, 112, 112) float32 in [0,1]
    ├── hrv.npy          (T, 4) float32 [mean_rr, rmssd, pnn50, sdnn]
    ├── spo2.npy         (T, 1) float32 normalised
    ├── resp.npy         (T, 1) float32 rate in Hz
    └── labels.csv       timestamp, nrs_score
```

## NRS label bins (QWK thresholds per Farrar et al. 2001)

| Level | NRS range |
|-------|-----------|
| No pain | < 1.5 |
| Mild | 1.5 – 3.5 |
| Moderate | 3.5 – 6.5 |
| Severe | ≥ 6.5 |

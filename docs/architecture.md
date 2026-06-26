# NeuroSight Architecture

For a reviewer-friendly architecture walkthrough, see
[`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md). This file keeps the
original compact diagrams for quick reference.

```mermaid
graph TD
    MRI[3D MRI .nii] --> VIT[MONAI ViT\n96^3 -> 768d]
    EEG[EEG .edf] --> PROC[MNE Preprocessing\nBandpass + Epochs]
    PROC --> CNN[Conv1D + Transformer\n-> 256d]
    COG[Cognitive Tests\nMMSE/MoCA/CDR] --> MLP[MLP Encoder\n8 -> 64d]
    VIT --> FUSE[CrossModal\nAttentionFusion]
    CNN --> FUSE
    MLP --> FUSE
    FUSE --> PRED[6-class Prediction\n+ Calibrated Confidence]
    FUSE --> XAI[XAI Engine\nGradCAM++/Attention/SHAP]
    PRED --> KG[Temporal KG\nPatient History]
    KG --> AGENT[LangGraph Agents\nSupervisor -> Specialists -> Safety]
    AGENT --> REPORT[Clinical Report\n+ Safety Validation]
```

## Agent State Machine

```mermaid
stateDiagram-v2
    [*] --> Supervisor
    Supervisor --> MRI_Analyst: has MRI data
    Supervisor --> EEG_Analyst: has EEG data
    Supervisor --> Cognitive_Analyst: has cognitive data
    Supervisor --> KG_Retriever: all modalities processed
    Supervisor --> Report_Writer: KG context ready
    Supervisor --> Safety_Guardian: draft report ready
    Safety_Guardian --> [*]: PASS
    Safety_Guardian --> [*]: BLOCK (drug dosage / overconfidence)
    MRI_Analyst --> Supervisor
    EEG_Analyst --> Supervisor
    Cognitive_Analyst --> Supervisor
    KG_Retriever --> Supervisor
    Report_Writer --> Supervisor
```

## Federated Learning

```mermaid
sequenceDiagram
    participant S as Server (FedAvg)
    participant H1 as Hospital A
    participant H2 as Hospital B
    participant H3 as Hospital C
    loop 5 Rounds
        S->>H1: Global weights
        S->>H2: Global weights
        S->>H3: Global weights
        H1->>H1: Local training (1 epoch)
        H2->>H2: Local training (1 epoch)
        H3->>H3: Local training (1 epoch)
        H1->>S: Updated weights
        H2->>S: Updated weights
        H3->>S: Updated weights
        S->>S: FedAvg aggregation
    end
    S->>S: Evaluate on held-out set
```

## Data Flow

```mermaid
flowchart LR
    A[ADNI CSV\n+MRI .npy\n+EEG .edf] --> B[ADNIDataset\nStratified split]
    B --> C[WeightedRandomSampler\nClass balance]
    C --> D[DataLoader\nBatch 16]
    D --> E[Phase A\nWarmup 10 ep]
    E --> F[Phase B\nFine-tune 40 ep\nCosineAnnealingWR]
    F --> G[Checkpointing\nbest val AUC]
    G --> H[ModelRegistry\nstaging → production]
    H --> I[evaluate.py\nTest split metrics]
    I --> J[MLflow / JSONL\nExperiment log]
```

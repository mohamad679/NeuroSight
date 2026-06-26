from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

class Modality(str, Enum):
    MRI = "mri"
    EEG = "eeg"
    COGNITIVE = "cognitive"
    FUSION = "fusion"

class Diagnosis(str, Enum):
    NORMAL = "normal"
    MCI = "mci"
    AD = "ad"
    FTD = "ftd"
    LBD = "lbd"
    VD = "vd"

class XAIMethod(str, Enum):
    GRAD_CAM = "grad_cam"
    ATTENTION_ROLLOUT = "attention_rollout"
    SHAP = "shap"

class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    MRI_ANALYST = "mri_analyst"
    EEG_ANALYST = "eeg_analyst"
    COGNITIVE_ANALYST = "cognitive_analyst"
    KG_RETRIEVER = "kg_retriever"
    REPORT_WRITER = "report_writer"
    SAFETY_GUARDIAN = "safety_guardian"

@dataclass
class MRIScan:
    data: Any # np.ndarray or torch.Tensor
    metadata: Dict[str, Any]

@dataclass
class EEGRecording:
    data: Any
    sfreq: float
    channels: List[str]

@dataclass
class CognitiveAssessment:
    scores: Dict[str, float]

@dataclass
class PatientRecord:
    patient_id: str
    age: float
    sex: str
    mri: Optional[MRIScan] = None
    eeg: Optional[EEGRecording] = None
    cognitive: Optional[CognitiveAssessment] = None

@dataclass
class ModalityPrediction:
    modality: Modality
    logits: List[float]
    probabilities: List[float]
    embedding: List[float]

@dataclass
class XAIExplanation:
    modality: Modality
    method: XAIMethod
    saliency: Any # np.ndarray or dict
    text_summary: str

@dataclass
class DiagnosisReport:
    patient_id: str
    final_diagnosis: Diagnosis
    confidence: float
    report_text: str
    requires_review: bool
    blocked_by_safety: bool

@dataclass
class KGNode:
    id: str
    label: str
    properties: Dict[str, Any]

@dataclass
class KGEdge:
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]

@dataclass
class TemporalQuery:
    patient_id: str
    target_date: datetime

@dataclass
class SimilarPatientResult:
    patient_id: str
    similarity_score: float
    shared_features: List[str]

@dataclass
class DiagnoseRequest:
    patient_record: PatientRecord
    query: Optional[str] = None

@dataclass
class DiagnoseResponse:
    diagnosis: Diagnosis
    confidence: float
    requires_review: bool
    report_text: str
    model_mode: Optional[str] = None
    checkpoint_id: Optional[str] = None
    trained_on_real_data: bool = False
    clinical_validated: bool = False
    requires_expert_review: bool = True
    disclaimer: Optional[str] = None
    warnings: Optional[List[str]] = None

@dataclass
class KGQueryRequest:
    query: str
    patient_id: Optional[str] = None

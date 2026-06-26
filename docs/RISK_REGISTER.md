# NeuroSight Risk Register

This document outlines the risk register and mitigation matrix for NeuroSight, analyzing 10 core safety, security, and clinical risk vectors associated with the AI portfolio framework.

## Risk Register Matrix

| Risk ID | Risk Vector | Description | Severity | Likelihood | Mitigation Strategy | Residual Risk | Status / Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NS-R-01** | LLM Hallucination | LLM/Agent generating false modality findings or invalid medical claims. | High | Medium |中央 `SafetyService` rephrases direct diagnoses, deterministic validation rules, and agent output sanitization. | Low | Mitigated / AI Systems Engineer |
| **NS-R-02** | Clinician Overreliance | Clinicians or users over-trusting demo model outputs or diagnoses. | High | Medium | Standard disclaimer headers, block medication/treatment queries, deterministic rephrasing of diagnoses. | Low | Mitigated / AI Safety Engineer |
| **NS-R-03** | Demographic Bias | Model performance disparity across age, gender, or ethnicity due to representation bias in training data. | Medium | Medium | Document training source demographic constraints in the Model Card, restrict claims to research use only. | Medium | Documented / ML Scientist |
| **NS-R-04** | Scanner / Site Shift | Model prediction degradation due to scanner parameter variations (MRI field strength, manufacturer). | Medium | High | Shape and format validation in modality contracts, input intensity range checks. | Medium | Mitigated / ML Engineer |
| **NS-R-05** | EEG Artifacts | Muscle, blink, or movement artifacts causing false positive or negative alpha/theta slowing. | Medium | High | Frequency-band checks during `preprocess_eeg` and raising warning flags if data is out-of-distribution. | Medium | Mitigated / ML Scientist |
| **NS-R-06** | Data Leakage | Training data leaking into testing or evaluation sets during development. | High | Low | Deterministic seeding, strict separation of benchmark evaluation subsets, and automated leakage tests. | Low | Mitigated / ML Scientist |
| **NS-R-07** | Privacy Leaks | Accidental upload or log persistence of Protected Health Information (PHI) or Personally Identifiable Information (PII). | High | Low | Ephemeral memory processing, prohibiting persistent storage of upload payloads, and strict log hygiene. | Low | Mitigated / SecOps |
| **NS-R-08** | Insecure Uploads | Upload of zip bombs, path traversal filenames, or nested archives targeting the API. | High | Medium | Filename validation (`_validate_filename`), zip member checking, decompression ratio boundaries, nested zip blocking. | Low | Mitigated / SecOps |
| **NS-R-09** | Synthetic Benchmark Misinterpretation | Evaluators assuming synthetic benchmark metrics reflect real-world clinical performance. | Medium | Low | Explicitly label all benchmark data as synthetic, prefix benchmark tables with warning banners. | Low | Mitigated / ML Scientist |
| **NS-R-10** | Regulatory Misuse | Attempting to deploy this research prototype in a real clinical setting. | High | Low | Plainly document research/educational positioning, reject medication advice, enforce mandatory safety footer on reports. | Low | Mitigated / AI Safety Engineer |

## Risk Mitigation Details

### 1. Centralized Safety Service (`NS-R-01`, `NS-R-02`, `NS-R-10`)
- **Action**: All queries go through `SafetyService.evaluate_query_safety` prior to supervisor execution.
- **Wording Rephrasing**: Direct diagnosis declarations are rephrased to state probabilistic model output boundaries.
- **Refusal**: Treatment, dosage, or emergency queries are rejected with an explicit safety message.

### 2. File Upload Hardening (`NS-R-08`)
- **Traversal Guard**: Filenames containing `..`, `/`, or `\` are rejected at the REST API boundaries.
- **Zip Bomb Defense**: Compression ratios exceeding 100x on individual members or total file sizes are blocked.
- **Nested Archive Guard**: File entries matching zip/tar/rar magic numbers or matching archive extensions are blocked.

### 3. Privacy & Log Hygiene (`NS-R-07`)
- **No Persistence**: Uploaded assets are parsed in-memory (or via short-lived temporary files that are guaranteed to be cleaned up).
- **Log Sanitation**: Tracing spans and standard output logs never record raw medical datasets or draft report texts.

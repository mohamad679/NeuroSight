# NeuroSight Technical & Non-Clinical Positioning

This document defines the technical and non-clinical positioning of the NeuroSight project as an engineering research demo scaffold. It outlines why the repository is positioned strictly as a software architecture mockup and the significant clinical clearance hurdles that differentiate it from a real-world medical application.

## Non-Clinical Scope and Status

> [!IMPORTANT]
> **NEUROSIGHT IS A SOFTWARE DEMO SCAFFOLD AND IS NOT CLINICALLY VALIDATED.**
> This codebase, API, and associated models serve exclusively as a senior AI engineering portfolio project to demonstrate systems architecture, safety agents, MLOps tooling, and synthetic evaluation.

- **Not for Clinical Use**: The software must not be used to diagnose, treat, triage, or manage any medical condition in humans. All outputs are model-generated risk profiles for demo purposes only.
- **No Regulatory Compliance**: No compliance is claimed with clinical standards such as HIPAA, GDPR (in a healthcare context), EU MDR, FDA 21 CFR Part 11, or high-risk medical AI regulations.
- **Synthetic Boundaries**: All model code, benchmarks, API responses, and UI elements run on synthetic datasets or mock inputs. Performance metrics do not estimate real-world diagnostic accuracy.

---

## Clinical Clearance Hurdles vs. Demo Scope

To transition an engineering scaffold like NeuroSight into a real-world Software as a Medical Device (SaMD) product, the following extensive clinical, quality, and regulatory validation phases would have to be completed from scratch:

### 1. Quality Management System (QMS) Implementation
- Establish a QMS compliant with **ISO 13485** (Medical devices — Quality management systems) and **FDA 21 CFR Part 820** (Quality System Regulation).
- Implement document control, design controls, risk management (**ISO 14971**), and software lifecycle processes (**IEC 62304**).

### 2. Prospective Clinical Validation
- **Current Limitation**: Retrospective benchmarks on synthetic data are highly prone to overfitting and scanner shift.
- **Requirement**: Conduct multi-site prospective clinical trials to evaluate clinical classification accuracy (sensitivity, specificity, AUC) on real-world clinical populations.
- Define explicit inclusion and exclusion criteria, and establish reference standards (ground truth) using consensus panel diagnoses.

### 3. Human Factors & Usability Engineering
- Conduct usability testing according to **IEC 62366-1** (Usability engineering to medical devices).
- Evaluate user interfaces with target clinicians to identify use-related hazards, cognitive overload, or overreliance/automation bias (e.g., clinicians rubber-stamping AI recommendations).

### 4. Cybersecurity and Vulnerability Management
- Align with the FDA's Cybersecurity in Medical Devices guidelines.
- Implement threat modeling, secure boot, data encryption at rest and in transit, user authentication (OAuth2/MFA), and a formal vulnerability disclosure program.
- Secure the model supply chain against adversarial attacks (prompt injection, data poisoning, model inversion).

### 5. Regulatory Submission and Clearance
- **United States**: Prepare and submit a **510(k)** premarket notification (demonstrating substantial equivalence to a legally marketed predicate device) or a **De Novo** classification request to the FDA.
- **European Union**: Conduct a conformity assessment under the **EU MDR 2017/745** to obtain a CE mark for a Class IIa/IIb medical device, involving a notified body review.

### 6. Post-Market Surveillance (PMS) & Continuous Monitoring
- Establish a PMS plan to monitor real-world performance, track drift in incoming data distributions, and log clinical feedback.
- Set up a system for reporting adverse events, software bugs, and safety recalls to regulatory authorities.

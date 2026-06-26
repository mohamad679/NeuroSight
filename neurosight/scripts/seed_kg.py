# neurosight/scripts/seed_kg.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from neurosight.contracts import PatientRecord
from knowledge_graph import NeuroKnowledgeGraph

DIAGNOSES = ["normal", "mci", "ad", "ftd", "lbd", "vd"]
DATES = ["2021-03-15", "2022-07-20", "2023-01-01", "2023-11-10", "2024-04-05"]


def seed_kg():
    kg = NeuroKnowledgeGraph()
    n = 50

    for i in range(n):
        pid = f"SYN_{i:04d}"
        age = 60.0 + (i % 30)
        sex = "M" if i % 2 == 0 else "F"
        diagnosis = DIAGNOSES[i % len(DIAGNOSES)]
        date = DATES[i % len(DATES)]
        confidence = round(0.70 + (i % 6) * 0.05, 2)

        record = PatientRecord(patient_id=pid, age=age, sex=sex)
        kg.add_patient(record)
        kg.add_diagnosis(pid, diagnosis, date, confidence, "Clinical")

        if i > 0:
            kg.add_similarity(pid, f"SYN_{(i-1):04d}", 0.9, ["Age", "APOE4"])

    os.makedirs("data", exist_ok=True)
    kg.save()
    print(f"Seeded {n} patients | {len(kg.graph.edges())} edges | "
          f"{len(set(DIAGNOSES))} diagnosis types")


if __name__ == "__main__":
    seed_kg()

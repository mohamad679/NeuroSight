# knowledge_graph.py
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
import networkx as nx

from neurosight.contracts import PatientRecord


def _json_safe(value: Any) -> Any:
    """Convert graph payload values into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


class SimilarPatientResult:
    def __init__(self, patient_id, similarity_score, shared_features):
        self.patient_id = patient_id
        self.similarity_score = similarity_score
        self.shared_features = shared_features


class NeuroKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_patient(self, record: Any):
        self.graph.add_node(record.patient_id, type="patient", record=record)

    def add_diagnosis(self, patient_id: str, diagnosis: str, date: str, confidence: float, source: str):
        diag_node = f"diag_{patient_id}_{date}"
        self.graph.add_node(diag_node, type="diagnosis", diagnosis=diagnosis, confidence=confidence, source=source)
        self.graph.add_edge(patient_id, diag_node, relationship="has_diagnosis", date=date)

    def add_biomarker(self, patient_id: str, name: str, value: float, unit: str, date: str):
        bio_node = f"bio_{patient_id}_{name}_{date}"
        self.graph.add_node(bio_node, type="biomarker", name=name, value=value, unit=unit)
        self.graph.add_edge(patient_id, bio_node, relationship="has_biomarker", date=date)

    def add_drug(self, patient_id: str, drug_name: str, start_date: str, end_date: str, dose: str):
        drug_node = f"drug_{patient_id}_{drug_name}_{start_date}"
        self.graph.add_node(drug_node, type="drug", name=drug_name, dose=dose)
        self.graph.add_edge(patient_id, drug_node, relationship="took_drug", start_date=start_date, end_date=end_date)

    def add_similarity(self, id_a: str, id_b: str, score: float, shared_features: List[str]):
        self.graph.add_edge(id_a, id_b, relationship="similar_to", score=score, features=shared_features)

    def get_patient_history(self, patient_id: str, before_date: Optional[str] = None) -> List[Dict]:
        history = []
        if patient_id in self.graph:
            for _, target, data in self.graph.edges(patient_id, data=True):
                node_data = self.graph.nodes[target]
                if before_date:
                    date = data.get("date") or data.get("start_date")
                    if date and date >= before_date:
                        continue
                history.append({"node": node_data, "edge": data})
        return history

    def find_similar_patients(self, patient_id: str, top_k: int = 5) -> List[SimilarPatientResult]:
        similarities = []
        if patient_id in self.graph:
            for _, target, data in self.graph.edges(patient_id, data=True):
                if data.get("relationship") == "similar_to":
                    similarities.append(SimilarPatientResult(target, data.get("score"), data.get("features")))
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]

    def query_at_date(self, patient_id: str, target_date: str) -> Dict:
        snapshot = {}
        node_id = patient_id
        if node_id not in self.graph: return {}
        for _, target, data in self.graph.edges(node_id, data=True):
            rel = data.get("relationship", "unknown")
            edge_date = data.get("date") or data.get("start_date") or ""
            end_date = data.get("end_date")
            started = edge_date <= target_date if edge_date else True
            ended = (end_date < target_date) if end_date else False
            if started and not ended:
                node_data = dict(self.graph.nodes[target])
                if rel not in snapshot:
                    snapshot[rel] = []
                snapshot[rel].append({
                    "node_id": target,
                    "node_data": node_data,
                    "edge_data": {k: v for k, v in data.items() if k != "relationship"}
                })
        return snapshot

    def get_disease_progression(self, patient_id: str) -> List[Dict]:
        progression = []
        if patient_id not in self.graph: return []
        for _, target, data in self.graph.edges(patient_id, data=True):
            if data.get("relationship") == "has_diagnosis":
                node_data = dict(self.graph.nodes[target])
                progression.append({
                    "date": data.get("date", ""),
                    "diagnosis": node_data.get("diagnosis", ""),
                    "confidence": node_data.get("confidence", 0.0),
                    "source": node_data.get("source", ""),
                })
        return sorted(progression, key=lambda x: x["date"])

    def save(self, filepath="data/neurosight_kg.json"):
        data = nx.node_link_data(self.graph)
        data = _json_safe(data)
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath="data/neurosight_kg.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)

    def migrate_to_neo4j(self, uri, user, password):
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        node_count, edge_count = 0, 0
        with driver.session() as session:
            for node_id, attrs in self.graph.nodes(data=True):
                label = attrs.get("type", "Unknown").capitalize()
                props = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                         for k, v in attrs.items() if k != "type"}
                session.run(
                    f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                    id=node_id, props=props
                )
                node_count += 1
            for src, dst, attrs in self.graph.edges(data=True):
                rel = attrs.get("relationship", "RELATED_TO").upper().replace(" ", "_")
                props = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                         for k, v in attrs.items() if k != "relationship"}
                session.run(
                    f"MATCH (a {{id:$src}}),(b {{id:$dst}}) MERGE (a)-[r:{rel}]->(b) SET r += $props",
                    src=src, dst=dst, props=props
                )
                edge_count += 1
        driver.close()
        return {"nodes_migrated": node_count, "edges_migrated": edge_count}


class KGQueryEngine:
    def __init__(self, kg: "NeuroKnowledgeGraph"):
        self.kg = kg

    def natural_language_to_query(self, nl_query: str, llm_client=None) -> Dict:
        """Parse a natural language question into a structured query dict."""
        nl = nl_query.lower()

        # Extract patient_id: patterns like SYN_0001, P001, patient 5
        pid_match = re.search(r'\b(syn_\d+|p\d+|patient[_\s]?\d+)\b', nl, re.I)
        patient_id = pid_match.group(0).replace(" ", "_").upper() if pid_match else None

        # Extract date: YYYY-MM-DD
        date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', nl)
        target_date = date_match.group(1) if date_match else None

        # Classify query type by keywords
        if any(w in nl for w in ["similar", "like", "resemble", "closest"]):
            query_type = "similar"
        elif any(w in nl for w in ["snapshot", "on date", "at date", "state on"]):
            query_type = "snapshot"
        elif any(w in nl for w in ["progression", "over time", "history of disease", "worsened"]):
            query_type = "progression"
        else:
            query_type = "history"

        query = {"query_type": query_type, "patient_id": patient_id}
        if target_date:
            query["target_date"] = target_date

        # If LLM available, refine with it
        if llm_client and patient_id:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                resp = llm_client.invoke([
                    SystemMessage(content=(
                        "You parse medical queries into JSON. "
                        "Return only: {\"query_type\": \"history|similar|snapshot|progression\", "
                        "\"patient_id\": \"...\", \"target_date\": \"YYYY-MM-DD or null\"}"
                    )),
                    HumanMessage(content=nl_query),
                ])
                parsed = json.loads(resp.content)
                query.update({k: v for k, v in parsed.items() if v is not None})
            except Exception:
                pass  # fall back to keyword result

        return query

    def execute_query(self, query: Dict) -> List[Dict]:
        """Route parsed query dict to the right KG method."""
        qt = query.get("query_type", "history")
        pid = query.get("patient_id")

        if not pid:
            return []

        if qt == "history":
            return self.kg.get_patient_history(
                pid, before_date=query.get("target_date")
            )
        elif qt == "similar":
            results = self.kg.find_similar_patients(
                pid, top_k=int(query.get("top_k", 5))
            )
            return [
                {"patient_id": r.patient_id,
                 "similarity_score": r.similarity_score,
                 "shared_features": r.shared_features}
                for r in results
            ]
        elif qt == "snapshot":
            td = query.get("target_date")
            if not td:
                return []
            return [self.kg.query_at_date(pid, td)]
        elif qt == "progression":
            return self.kg.get_disease_progression(pid)

        return []

    def format_for_agent(self, results: List[Dict]) -> str:
        """Format query results as a concise clinical context string."""
        if not results:
            return "No records found in knowledge graph."

        lines = [f"Knowledge Graph: {len(results)} record(s) found."]
        for i, item in enumerate(results[:5], 1):  # cap at 5 for context window
            if "diagnosis" in item:
                lines.append(
                    f"  {i}. [{item.get('date', '')}] "
                    f"Dx: {item['diagnosis']} "
                    f"(conf: {item.get('confidence', '?')})"
                )
            elif "patient_id" in item:
                lines.append(
                    f"  {i}. Similar: {item['patient_id']} "
                    f"(score: {item.get('similarity_score', '?')})"
                )
            elif "node_data" in item:
                nd = item["node_data"]
                lines.append(
                    f"  {i}. {nd.get('type', '?')}: "
                    + ", ".join(f"{k}={v}" for k, v in nd.items()
                               if k not in ("type", "record"))
                )
            else:
                lines.append(f"  {i}. {str(item)[:80]}")

        if len(results) > 5:
            lines.append(f"  ... and {len(results) - 5} more records.")

        return "\n".join(lines)


class MockPatientRecord:
    """Minimal patient record for testing without real clinical data."""

    def __init__(self, patient_id: str):
        """Initialize a fixed synthetic patient profile for tests.

        Args:
            patient_id: Identifier assigned to the synthetic patient.

        Returns:
            None.
        """
        self.patient_id = patient_id
        self.age = 72.0
        self.sex = "M"
        self.mri = None
        self.eeg = None
        self.cognitive = {
            "mmse": 22.0,
            "moca": 19.0,
            "cdrsb": 4.5,
            "adas11": 18.0,
            "ravlt_immediate": 28.0,
            "ravlt_learning": 2.0,
            "faq": 6.0,
            "age": 72.0,
        }

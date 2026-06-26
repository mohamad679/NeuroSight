# neurosight/models/xai.py
import torch
import torch.nn as nn
import numpy as np


class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients = None
        self._activations = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        out = self.model(input_tensor)
        logits = out["logits"] if isinstance(out, dict) else out[0]
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, target_class].backward()

        grads = self._gradients
        acts = self._activations
        grads_sq = grads.pow(2)
        grads_cu = grads.pow(3)
        sum_acts = acts.sum(dim=list(range(2, acts.dim())), keepdim=True)
        alpha = grads_sq / (2 * grads_sq + sum_acts * grads_cu + 1e-8)
        weights = (alpha * torch.relu(grads)).sum(dim=list(range(2, grads.dim())), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class AttentionRollout:
    def __init__(self, model: nn.Module):
        self.model = model
        self._attention_maps = []
        for module in model.modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                module.self_attn.register_forward_hook(self._capture_attn)

    def _capture_attn(self, module, input, output):
        if isinstance(output, tuple) and output[1] is not None:
            self._attention_maps.append(output[1].detach())

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self._attention_maps = []
        self.model.eval()
        with torch.no_grad():
            self.model(input_tensor)
        if not self._attention_maps:
            n_tokens = input_tensor.shape[1] if input_tensor.dim() == 3 else 19
            return np.ones(n_tokens, dtype=np.float32) / n_tokens

        result = self._attention_maps[0].mean(dim=1)
        for attn in self._attention_maps[1:]:
            result = torch.bmm(attn.mean(dim=1), result)
        rollout = result[0, 0, 1:].cpu().numpy()
        rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-8)
        return rollout.astype(np.float32)


class SHAPExplainer:
    FEATURE_NAMES = ["MMSE", "MOCA", "CDRSB", "ADAS11", "RAVLT_immediate", "RAVLT_learning", "FAQ", "AGE"]

    def __init__(self, model: nn.Module, background: torch.Tensor):
        import shap

        def model_fn(x):
            t = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                _, emb = model(t)
                logits = model.head(emb)
            return torch.softmax(logits, dim=-1).numpy()

        self.explainer = shap.KernelExplainer(model_fn, background.numpy())

    def __call__(self, input_tensor: torch.Tensor, target_class: int = None) -> dict:
        shap_values = self.explainer.shap_values(input_tensor.numpy(), nsamples=100)
        if target_class is None:
            target_class = np.array(shap_values).mean(axis=0).argmax()
        values = shap_values[target_class][0]
        return dict(zip(self.FEATURE_NAMES, [float(v) for v in values]))


class XAIEngine:
    def __init__(self, mri_model=None, eeg_model=None, cognitive_model=None, llm_client=None):
        self.llm = llm_client
        self.cognitive_model = cognitive_model
        self.shap_explainer = None
        if mri_model:
            layers = [m for m in mri_model.modules() if isinstance(m, nn.LayerNorm)]
            conv_layers = [m for m in mri_model.modules() if isinstance(m, nn.Conv3d)]
            uses_monai_vit = bool(getattr(getattr(mri_model, "encoder", None), "_use_vit", False))
            target = layers[-1] if uses_monai_vit and layers else conv_layers[-1] if conv_layers else list(mri_model.modules())[-2]
            self.mri_gradcam = GradCAMPlusPlus(mri_model, target)
        if eeg_model:
            self.eeg_rollout = AttentionRollout(eeg_model)

    def explain_mri(self, mri_tensor: torch.Tensor, target_class: int = None):
        from neurosight.contracts import XAIExplanation, XAIMethod, Modality

        saliency = self.mri_gradcam.generate(mri_tensor, target_class)
        return XAIExplanation(
            modality=Modality.MRI, method=XAIMethod.GRAD_CAM,
            saliency=saliency, text_summary=""
        )

    def explain_eeg(self, eeg_tensor: torch.Tensor):
        from neurosight.contracts import XAIExplanation, XAIMethod, Modality

        importance = self.eeg_rollout(eeg_tensor)
        return XAIExplanation(
            modality=Modality.EEG, method=XAIMethod.ATTENTION_ROLLOUT,
            saliency=importance, text_summary=""
        )

    def explain_cognitive(
        self,
        cog_tensor: torch.Tensor,
        target_class: int = None,
        cognitive_model=None,
    ) -> "XAIExplanation":
        from neurosight.contracts import XAIExplanation, XAIMethod, Modality

        model = cognitive_model or getattr(self, "cognitive_model", None)

        # If SHAPExplainer was initialized, use it
        if hasattr(self, "shap_explainer") and self.shap_explainer is not None:
            scores = self.shap_explainer(cog_tensor, target_class)
            return XAIExplanation(
                modality=Modality.COGNITIVE,
                method=XAIMethod.SHAP,
                saliency=scores,
                text_summary="",
            )

        # Gradient-based fallback: d(logit_target) / d(input_features)
        if model is None:
            # No model available — return uniform importance
            n_features = cog_tensor.shape[-1] if cog_tensor.dim() > 0 else 8
            feature_names = SHAPExplainer.FEATURE_NAMES
            uniform = {f: 1.0 / n_features for f in feature_names}
            return XAIExplanation(
                modality=Modality.COGNITIVE,
                method=XAIMethod.SHAP,
                saliency=uniform,
                text_summary="Uniform importance — no model provided",
            )

        model.eval()
        x = cog_tensor.clone().detach().requires_grad_(True)
        logits, _ = model(x)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        model.zero_grad()
        logits[0, target_class].backward()

        # Gradient × input as importance score (integrated-gradients proxy)
        importance = (x.grad * x).squeeze(0).detach().cpu().numpy()
        # Normalize to [0, 1]
        abs_imp = np.abs(importance)
        norm = abs_imp / (abs_imp.max() + 1e-8)
        scores = dict(zip(SHAPExplainer.FEATURE_NAMES, [float(v) for v in norm]))

        return XAIExplanation(
            modality=Modality.COGNITIVE,
            method=XAIMethod.SHAP,
            saliency=scores,
            text_summary="",
        )

    @staticmethod
    def _resolve_modality_weights(report, explanations: list) -> dict:
        """Resolve and normalize modality weights from report context.

        Args:
            report: Diagnosis report-like object that may carry `modality_weights`.
            explanations: List of modality explanation objects (currently unused for
                weight extraction but kept for compatibility with call sites).

        Returns:
            Dictionary with normalized `mri`, `eeg`, and `cog` float weights.
        """
        del explanations  # reserved for future integration
        raw_weights = getattr(report, "modality_weights", None)
        if not isinstance(raw_weights, dict):
            return {"mri": 1.0 / 3.0, "eeg": 1.0 / 3.0, "cog": 1.0 / 3.0}

        mri = float(raw_weights.get("mri", 0.0))
        eeg = float(raw_weights.get("eeg", 0.0))
        cog = float(raw_weights.get("cog", 0.0))
        arr = np.array([mri, eeg, cog], dtype=np.float32)
        arr = np.clip(arr, a_min=0.0, a_max=None)
        total = float(arr.sum())
        if total <= 0.0:
            arr = np.full((3,), 1.0 / 3.0, dtype=np.float32)
        else:
            arr = arr / total
        return {"mri": float(arr[0]), "eeg": float(arr[1]), "cog": float(arr[2])}

    def generate_nl_explanation(self, report, explanations: list, llm_client=None) -> str:
        client = llm_client or self.llm
        conf = float(getattr(report, "confidence", 1.0))
        caveat = " Note: confidence below 0.75 — clinical correlation required." if conf < 0.75 else ""
        if client is None:
            diag = getattr(report, "final_diagnosis", "unknown")
            diag_label = diag.value if hasattr(diag, "value") else str(diag)
            requires_review = bool(getattr(report, "requires_review", conf < 0.75))
            modality_weights = self._resolve_modality_weights(report, explanations)
            dominant = max(modality_weights, key=modality_weights.get)
            dominant_map = {
                "mri": "structural MRI abnormalities",
                "eeg": "EEG spectral patterns",
                "cog": "neuropsychological assessment scores",
            }
            follow_up = (
                "Clinical review required — confidence below threshold."
                if requires_review
                else "Recommend specialist confirmation before clinical action."
            )
            return (
                f"AI assessment suggests {diag_label} (confidence {conf:.0%}). "
                f"Primary evidence: {dominant_map[dominant]}. "
                f"{follow_up}"
            )
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = (
            f"Patient diagnosis: {report.final_diagnosis}, confidence: {conf:.0%}.\n"
            f"Caveat: {caveat}\n"
            "Write exactly 3 sentences for a neurologist: "
            "1) Main finding. 2) Strongest evidence modality. 3) Recommended next step."
        )
        response = client.invoke([
            SystemMessage(content="You are a clinical AI assistant. Be concise and evidence-based."),
            HumanMessage(content=prompt)
        ])
        return response.content

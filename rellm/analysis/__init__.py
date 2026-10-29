from .lens import next_token, generate, logit_lens
from .attention import attention_patterns, head_summary, head_catalog
from .patching import activation_patching, ablate_heads
from .attribution import direct_logit_attribution
from .sae import sae_features
from .static import inspect_model_artifact, reconstruct_architecture
from .compare import compare_circuits

# authority of each operation's claims (see evidence.py)
AUTHORITY = {
    "next_token": "observation", "generate": "observation", "activation_patching": "observation", "ablate_heads": "observation",
    "logit_lens": "inference", "attention_patterns": "inference", "head_summary": "inference", "head_catalog": "inference",
    "sae_features": "inference", "direct_logit_attribution": "derived", "compare_circuits": "derived",
    "inspect_model_artifact": "declared", "open_model": "declared", "session": "declared", "tokenize": "declared", "chat_prompt": "declared",
}

__all__ = ["next_token", "generate", "logit_lens", "attention_patterns", "head_summary", "head_catalog", "activation_patching",
           "ablate_heads", "direct_logit_attribution", "sae_features", "inspect_model_artifact", "reconstruct_architecture",
           "compare_circuits", "AUTHORITY"]

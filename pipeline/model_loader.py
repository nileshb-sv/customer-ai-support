"""
pipeline/model_loader.py
Loads T5 exactly ONCE and shares it across predict.py and reply_engine.py.
"""
import os, torch

MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "support_llm")
)

_tokenizer = _model = _device = None
_loaded = False
use_trained_model = False


def load_model():
    global _tokenizer, _model, _device, _loaded, use_trained_model
    if _loaded:
        return _tokenizer, _model, _device, use_trained_model
    _loaded = True
    if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            print("Loading T5 model (shared loader)...")
            _tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
            _model     = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
            _device    = "cuda" if torch.cuda.is_available() else "cpu"
            _model     = _model.to(_device)
            _model.eval()
            use_trained_model = True
            print(f"  T5 loaded on {_device}")
        except Exception as e:
            print(f"  T5 load failed: {e}")
            _tokenizer = _model = _device = None
            use_trained_model = False
    else:
        print("No trained model found — keyword classifier will be used.")
    return _tokenizer, _model, _device, use_trained_model

"""
Conference Video Transcriber with Speaker Diarization
=====================================================
Transcribes conference videos locally with speaker separation.
Uses faster-whisper + pyannote.audio, with a Gradio web UI.

Compatible with Python 3.9+ and Apple Silicon Macs.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import tempfile
import traceback
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# ---------------------------------------------------------------------------
# Monkey-patch: huggingface_hub >= 1.0 removed HfFolder, but gradio 4.29
# imports it.  Inject a shim BEFORE importing gradio.
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import HfFolder  # noqa: F401  — already present, nothing to do
except ImportError:
    import huggingface_hub as _hfhub

    class _HfFolderShim:
        """Minimal stand-in for the removed HfFolder class."""

        @staticmethod
        def get_token():
            try:
                return _hfhub.get_token()
            except Exception:
                return None

        @staticmethod
        def save_token(token):
            try:
                _hfhub.login(token=token, add_to_git_credential=False)
            except Exception:
                pass

        @staticmethod
        def delete_token():
            try:
                _hfhub.logout()
            except Exception:
                pass

    _hfhub.HfFolder = _HfFolderShim

import gradio as gr

# ---------------------------------------------------------------------------
# Monkey-patch gradio_client bug: "const" in schema crashes when schema is a
# bool (valid JSON Schema, but gradio_client doesn't handle it).
# See: gradio_client/utils.py get_type() line ~862
# ---------------------------------------------------------------------------
try:
    import gradio_client.utils as _gc_utils
    _original_get_type = _gc_utils.get_type

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "bool"
        return _original_get_type(schema)

    _gc_utils.get_type = _patched_get_type

    # Also patch _json_schema_to_python_type which can receive a bool schema
    _original_jstpt = _gc_utils._json_schema_to_python_type

    def _patched_jstpt(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _original_jstpt(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_jstpt
except Exception:
    pass  # If the structure changes in a future version, just skip

# ---------------------------------------------------------------------------
# Monkey-patch: Gradio's ProgressUnit (Pydantic v2) requires `length` to be
# int, but libraries may pass float durations.  Safety net: patch the model
# validator so floats are silently cast to int.
# ---------------------------------------------------------------------------
try:
    from gradio.queueing import ProgressUnit as _PU
    _orig_pu_init = _PU.__init__

    def _patched_pu_init(self, **data):
        if "length" in data and isinstance(data["length"], float):
            data["length"] = int(data["length"])
        return _orig_pu_init(self, **data)

    _PU.__init__ = _patched_pu_init
except Exception:
    pass

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
OUTPUT_DIR = APP_DIR / "output"
CONFIG_FILE = APP_DIR / "config.json"

_whisper_model = None
_whisper_model_size = None  # type: Optional[str]
_whisper_backend = None  # "faster-whisper" or "openai-whisper"
_diarization_pipeline = None

# ---------------------------------------------------------------------------
# Rolling log file — keeps last 3 x 2 MB files in logs/ folder
# ---------------------------------------------------------------------------
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger("transcriber")
log.setLevel(logging.DEBUG)

# File handler — rolling 2 MB, keep 3 backups
_fh = logging.handlers.RotatingFileHandler(
    str(LOG_DIR / "transcriber.log"),
    maxBytes=2 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
log.addHandler(_fh)

# Also log to console (stdout) — force flush after every line
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))


# Force-flush both handlers after every log message so we never lose output
class _FlushingHandler(logging.Handler):
    """Mixin-style wrapper that flushes the underlying handler after every emit."""
    def __init__(self, wrapped):
        super().__init__(wrapped.level)
        self._wrapped = wrapped
    def emit(self, record):
        self._wrapped.emit(record)
        self._wrapped.flush()
    def setLevel(self, level):
        super().setLevel(level)
        self._wrapped.setLevel(level)

_fh_flushing = _FlushingHandler(_fh)
_fh_flushing.setLevel(logging.DEBUG)
_ch_flushing = _FlushingHandler(_ch)
_ch_flushing.setLevel(logging.INFO)

# Replace the direct handlers with flushing wrappers
log.removeHandler(_fh)
log.addHandler(_fh_flushing)
log.addHandler(_ch_flushing)

log.info("=" * 60)
log.info("Conference Transcriber starting up")
log.info("App dir: %s", APP_DIR)
log.info("Python:  %s", sys.version.split()[0])


# ---------------------------------------------------------------------------
# Config file (stores HuggingFace token so you don't re-enter it)
# ---------------------------------------------------------------------------
def load_config() -> Dict:
    """Load config from config.json, returning defaults if missing."""
    defaults = {"hf_token": ""}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            # merge with defaults so new keys are always present
            for k, v in defaults.items():
                data.setdefault(k, v)
            return data
        except Exception:
            pass
    return defaults


def save_config(config: Dict) -> None:
    """Write config to config.json."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print("Warning: could not save config: %s" % e)


def save_token(token: str) -> None:
    """Convenience: save just the HF token."""
    cfg = load_config()
    cfg["hf_token"] = token.strip()
    save_config(cfg)


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
def _check_ffmpeg() -> str:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return r.stdout.split("\n")[0] if r.stdout else "found"
    except FileNotFoundError:
        pass

    # Try to locate Gyan.FFmpeg from winget directory and append to PATH
    import glob
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        winget_dir = os.path.join(user_profile, "AppData", "Local", "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe")
        if os.path.isdir(winget_dir):
            ffmpeg_subdirs = glob.glob(os.path.join(winget_dir, "ffmpeg-*"))
            for d in ffmpeg_subdirs:
                bin_dir = os.path.join(d, "bin")
                if os.path.isfile(os.path.join(bin_dir, "ffmpeg.exe")):
                    os.environ["PATH"] = os.environ["PATH"] + os.path.pathsep + bin_dir
                    try:
                        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
                        return r.stdout.split("\n")[0] if r.stdout else "found"
                    except FileNotFoundError:
                        pass

    raise RuntimeError(
        "ffmpeg is not installed.\nInstall it with:  brew install ffmpeg"
    )


_torch_patched = False

def _import_torch():
    global _torch_patched
    import torch
    if not _torch_patched:
        # PyTorch 2.6+ defaults torch.load to weights_only=True, which breaks
        # pyannote/speechbrain model loading.  Multiple strategies:

        # Strategy 1: Allowlist known globals that pyannote/speechbrain need
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
            try:
                safe_classes = [torch.torch_version.TorchVersion]
                # Add any other commonly needed classes
                for cls_path in ("torch.nn.parameter.Parameter",
                                 "collections.OrderedDict",
                                 "numpy.core.multiarray.scalar",
                                 "numpy.dtype"):
                    try:
                        parts = cls_path.rsplit(".", 1)
                        mod = __import__(parts[0], fromlist=[parts[1]])
                        safe_classes.append(getattr(mod, parts[1]))
                    except Exception:
                        pass
                torch.serialization.add_safe_globals(safe_classes)
                log.debug("Added %d safe globals for torch.load", len(safe_classes))
            except Exception as e:
                log.debug("Could not add safe globals: %s", e)

        # Strategy 2: Patch torch.load so weights_only never causes problems.
        # Some PyTorch versions accept weights_only in the signature but crash
        # internally when passing it to Unpickler.  Safest approach: always
        # strip it out so the old default (False / no check) is used.
        _orig_load = torch.load

        def _safe_load(*args, **kwargs):
            kwargs.pop("weights_only", None)
            return _orig_load(*args, **kwargs)

        torch.load = _safe_load
        if hasattr(torch, "serialization"):
            torch.serialization.load = _safe_load

        # Strategy 3: Patch the internal _load function if it exists
        if hasattr(torch.serialization, "_load"):
            _orig_internal = torch.serialization._load

            def _safe_internal_load(*args, **kwargs):
                kwargs.pop("weights_only", None)
                return _orig_internal(*args, **kwargs)

            torch.serialization._load = _safe_internal_load

        # Strategy 4: Patch torch.inference_mode with torch.no_grad on Windows/DirectML
        # to prevent "RuntimeError: Cannot set version_counter for inference tensor" on DirectML
        try:
            class inference_mode_mock:
                def __init__(self, mode=True):
                    self.mode = mode
                    self.no_grad = torch.no_grad()
                def __enter__(self):
                    if self.mode:
                        return self.no_grad.__enter__()
                def __exit__(self, t, v, tb):
                    if self.mode:
                        return self.no_grad.__exit__(t, v, tb)
                def __call__(self, func):
                    return self.no_grad(func)
            
            torch.inference_mode = inference_mode_mock
            log.info("Patched torch.inference_mode with torch.no_grad for DirectML compatibility")
        except Exception as e:
            log.debug("Could not patch torch.inference_mode: %s", e)

        _torch_patched = True
        log.debug("Patched torch.load to strip weights_only param")
    return torch


def _import_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel


_diarization_patched = False

def _import_diarization():
    global _diarization_patched
    # Patch torch.load BEFORE importing pyannote (speechbrain captures it at import time)
    _import_torch()

    # ---------------------------------------------------------------------------
    # CRITICAL: Patch hf_hub_download / snapshot_download across EVERY module
    # that has imported them.  The user's pyannote.pipeline base class calls
    # hf_hub_download(use_auth_token=...) but huggingface-hub >= 1.0 removed
    # that param.  We can't predict which module holds the reference, so we
    # scan sys.modules and patch every copy we find.
    # ---------------------------------------------------------------------------
    if not _diarization_patched:
        _diarization_patched = True

        def _make_compat(orig_fn):
            """Wrap a HF download function to translate use_auth_token → token."""
            def _wrapper(*args, **kwargs):
                if "use_auth_token" in kwargs:
                    tok = kwargs.pop("use_auth_token")
                    if tok is not None:
                        kwargs.setdefault("token", tok)
                return orig_fn(*args, **kwargs)
            _wrapper._compat_patched = True
            return _wrapper

        patched = []
        import huggingface_hub as _hfhub_mod_inner

        for fn_name in ("hf_hub_download", "snapshot_download"):
            # Get the REAL original from huggingface_hub itself
            real_fn = getattr(_hfhub_mod_inner, fn_name, None)
            if real_fn is None:
                continue
            wrapper = _make_compat(real_fn)

            # Patch it on huggingface_hub itself
            setattr(_hfhub_mod_inner, fn_name, wrapper)
            patched.append("huggingface_hub.%s" % fn_name)

            # Now scan ALL loaded modules and replace any reference to the
            # original function with our wrapper
            import warnings as _warnings
            for mod_name, mod in list(sys.modules.items()):
                if mod is None:
                    continue
                try:
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore")
                        cur = getattr(mod, fn_name, None)
                except Exception:
                    continue
                # Patch if it's the original (unwrapped) function
                if cur is real_fn:
                    try:
                        setattr(mod, fn_name, wrapper)
                        patched.append("%s.%s" % (mod_name, fn_name))
                    except (AttributeError, TypeError):
                        pass

        if patched:
            log.info("Patched use_auth_token->token compat on: %s", ", ".join(patched))

    from pyannote.audio import Pipeline
    return Pipeline


# ---------------------------------------------------------------------------
# Reusable error messages for download UI
# ---------------------------------------------------------------------------
def _log_access_denied(ui_log, err_str: str) -> None:
    """Show a clear access-denied message in the download log."""
    ui_log("")
    ui_log("  ERROR: Access denied to pyannote models.")
    ui_log("")
    if "fine-grained" in err_str.lower() or "token settings" in err_str.lower() \
            or "permissions" in err_str.lower():
        ui_log("  Your token is a fine-grained token that lacks gated repo access.")
        ui_log("")
        ui_log("  EASIEST FIX — create a new classic token:")
        ui_log("  1. Go to https://huggingface.co/settings/tokens")
        ui_log("  2. Click 'Create new token' → choose 'Read' role")
        ui_log("  3. Copy the new token (starts with hf_)")
        ui_log("  4. Paste it here and click Download Models again")
        ui_log("")
        ui_log("  OR edit your existing token and enable")
        ui_log("  'Access to public gated repositories'.")
    else:
        ui_log("  Two things are needed:")
        ui_log("")
        ui_log("  A) Your token must allow gated repo access:")
        ui_log("     Go to https://huggingface.co/settings/tokens")
        ui_log("     Create a token with role 'Read' (classic, not fine-grained)")
        ui_log("     OR enable 'Access to public gated repositories' on your token.")
        ui_log("")
        ui_log("  B) You must accept the model licenses:")
        ui_log("     Open BOTH links below and click 'Agree and access repository':")
        ui_log("     1. https://huggingface.co/pyannote/speaker-diarization-3.1")
        ui_log("     2. https://huggingface.co/pyannote/segmentation-3.0")
    ui_log("")


def _log_token_rejected(ui_log) -> None:
    """Show a clear token-rejected message in the download log."""
    ui_log("")
    ui_log("  ERROR: HuggingFace token was rejected (401).")
    ui_log("  Your token may be expired or invalid.")
    ui_log("  Create a new token at https://huggingface.co/settings/tokens")
    ui_log("  (choose role 'Read' and enable 'Access to public gated repos').")
    ui_log("")


# ---------------------------------------------------------------------------
# Model download / setup
# ---------------------------------------------------------------------------
def download_models(
    hf_token: str,
    model_size: str = "large-v3",
    progress: gr.Progress = gr.Progress(),
) -> str:
    """Pre-download all required models."""
    if not hf_token or not hf_token.strip().startswith("hf_"):
        return "Error: please provide a valid HuggingFace token (starts with hf_)."

    hf_token = hf_token.strip()
    MODELS_DIR.mkdir(exist_ok=True)
    lines = []  # type: List[str]

    def ui_log(msg: str) -> None:
        lines.append(msg)
        log.info("[download] %s", msg)

    # 1 — Whisper
    progress(0.0, desc="Downloading Whisper %s..." % model_size)
    ui_log("Downloading Whisper '%s' model..." % model_size)
    try:
        # Pre-download for both possible backends (faster-whisper and openai-whisper)
        import whisper
        ui_log("  Downloading OpenAI Whisper model...")
        _m_open = whisper.load_model(model_size, device="cpu")
        del _m_open
        
        ui_log("  Downloading faster-whisper model...")
        WhisperModel = _import_whisper()
        _m_fast = WhisperModel(model_size, device="cpu", compute_type="int8")
        del _m_fast
        
        ui_log("  OK - Whisper models downloaded and cached.")
    except Exception as e:
        ui_log("  FAILED - Whisper download failed: %s" % e)
        log.error("Whisper download traceback:\n%s", traceback.format_exc())
        return "\n".join(lines)

    # 2 — Pyannote
    progress(0.4, desc="Downloading pyannote diarization...")
    ui_log("Downloading pyannote speaker-diarization-3.1...")
    try:
        # Log versions for diagnostics
        import pyannote.audio as _pa
        import huggingface_hub as _hfhub_mod
        ui_log("  pyannote.audio version: %s" % getattr(_pa, "__version__", "unknown"))
        ui_log("  huggingface-hub version: %s" % _hfhub_mod.__version__)

        # Step 2a: Verify the token actually works with HuggingFace
        try:
            user_info = _hfhub_mod.whoami(token=hf_token)
            ui_log("  Token valid — logged in as: %s" % user_info.get("name", "unknown"))
        except Exception as auth_err:
            ui_log("  FAILED - Token validation failed: %s" % auth_err)
            ui_log("  Check your token at https://huggingface.co/settings/tokens")
            return "\n".join(lines)

        # Step 2b: Check if user has accepted the gated model licenses
        # (model_info can succeed even without gated access, so we try
        #  downloading a small file to verify actual download permission)
        for repo in ("pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0"):
            try:
                _hfhub_mod.hf_hub_download(
                    repo, "config.yaml", token=hf_token,
                    force_download=True,  # bypass cache to test real access
                )
                ui_log("  Access OK: %s" % repo)
            except Exception as repo_err:
                # Get the full chain of causes for better diagnostics
                err_str = str(repo_err)
                cause = repo_err.__cause__ or repo_err.__context__
                cause_str = str(cause) if cause else ""
                full_err = "%s | cause: %s" % (err_str, cause_str) if cause else err_str
                log.error("Access check for %s failed: %s\n%s", repo, full_err, traceback.format_exc())

                if "403" in full_err or "gated" in full_err.lower():
                    ui_log("")
                    ui_log("  ERROR: Access denied to %s" % repo)
                    ui_log("")
                    if "fine-grained" in full_err.lower() or "token settings" in full_err.lower():
                        ui_log("  Your token is a fine-grained token that lacks gated repo access.")
                        ui_log("")
                        ui_log("  EASIEST FIX: Create a new classic token instead:")
                        ui_log("  1. Go to https://huggingface.co/settings/tokens")
                        ui_log("  2. Click 'Create new token' → choose 'Read' role")
                        ui_log("  3. Copy the new token (starts with hf_)")
                        ui_log("  4. Paste it here and click Download Models again")
                        ui_log("")
                        ui_log("  OR: Edit your existing token and enable")
                        ui_log("  'Access to public gated repositories'.")
                    else:
                        ui_log("  You need to accept the model license:")
                        ui_log("  1. Open https://huggingface.co/%s" % repo)
                        ui_log("  2. Log in as '%s'" % user_info.get("name", "your account"))
                        ui_log("  3. Click 'Agree and access repository'")
                        ui_log("  4. Come back here and click Download Models again")
                    return "\n".join(lines)
                elif "401" in full_err:
                    ui_log("")
                    ui_log("  ERROR: Token rejected (401) for %s" % repo)
                    ui_log("  Create a new token at https://huggingface.co/settings/tokens")
                    return "\n".join(lines)
                elif "404" in err_str:
                    # segmentation-3.0 may not have config.yaml; just check model_info
                    try:
                        _hfhub_mod.model_info(repo, token=hf_token)
                        ui_log("  Access OK: %s (metadata only)" % repo)
                    except Exception:
                        ui_log("  WARNING checking %s: %s" % (repo, repo_err))
                else:
                    # Network error, SSL error, or other — show full detail
                    ui_log("")
                    ui_log("  ERROR downloading from %s:" % repo)
                    ui_log("  %s" % err_str[:200])
                    if cause:
                        ui_log("  Caused by: %s" % cause_str[:200])
                    ui_log("")
                    ui_log("  Check your internet connection and try again.")
                    ui_log("  If this persists, check logs/transcriber.log for details.")
                    return "\n".join(lines)

        # Step 2c: Log in globally so from_pretrained picks up the token
        # automatically (avoids use_auth_token vs token param mismatch).
        ui_log("  Setting HF token globally via login()...")
        _hfhub_mod.login(token=hf_token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = hf_token  # belt-and-suspenders

        # Step 2d: Load the pipeline
        Pipeline = _import_diarization()
        ui_log("  Loading pipeline (this may take a minute)...")

        # Detect which token kwarg from_pretrained accepts
        import inspect
        _fp_sig = inspect.signature(Pipeline.from_pretrained)
        _fp_params = list(_fp_sig.parameters.keys())
        log.info("from_pretrained signature: %s", _fp_params)
        ui_log("  from_pretrained params: %s" % _fp_params)

        if "token" in _fp_params:
            token_kw = {"token": hf_token}
            token_param_name = "token"
        elif "use_auth_token" in _fp_params:
            token_kw = {"use_auth_token": hf_token}
            token_param_name = "use_auth_token"
        else:
            # Neither found — rely on global login only
            token_kw = {}
            token_param_name = "(global login)"
        log.info("Using token param: %s", token_param_name)

        _p = None
        last_err = None

        # Attempt 1: explicit token (using detected param name)
        try:
            log.info("Attempt 1: from_pretrained with %s — starting...", token_param_name)
            _p = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", **token_kw,
            )
            if _p is not None:
                ui_log("  Pipeline loaded successfully.")
                log.info("Pipeline loaded OK via %s", token_param_name)
            else:
                log.warning("from_pretrained with %s returned None", token_param_name)
        except Exception as ex:
            last_err = ex
            log.error("Attempt 1 FAILED (%s): %s: %s",
                      token_param_name, type(ex).__name__, ex)
            log.debug("Traceback:\n%s", traceback.format_exc())

        # Attempt 2: global login only (no explicit token)
        if _p is None and token_kw:
            try:
                log.info("Attempt 2: from_pretrained with global login only — starting...")
                _p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                if _p is not None:
                    ui_log("  Pipeline loaded successfully (via global login).")
                    log.info("Pipeline loaded OK via global login")
                else:
                    log.warning("from_pretrained (global login) returned None")
            except Exception as ex:
                last_err = ex
                log.error("Attempt 2 FAILED (global login): %s: %s",
                          type(ex).__name__, ex)
                log.debug("Traceback:\n%s", traceback.format_exc())

        # Fallback: download config.yaml manually and pass the local file path
        if _p is None:
            ui_log("  Standard loading returned None — trying direct path fallback...")
            log.info("Attempting direct download fallback for pipeline config")
            try:
                config_path = _hfhub_mod.hf_hub_download(
                    "pyannote/speaker-diarization-3.1",
                    "config.yaml",
                    token=hf_token,
                )
                log.info("Downloaded config.yaml to: %s", config_path)
                if config_path:
                    # Pass local path + token using whichever param name works
                    _p = Pipeline.from_pretrained(config_path, **token_kw)
                    if _p is not None:
                        ui_log("  Pipeline loaded via direct download fallback.")
                    else:
                        log.warning("from_pretrained(local path) also returned None")
            except Exception as fallback_err:
                log.error("Direct download fallback failed: %s\n%s",
                          fallback_err, traceback.format_exc())
                if last_err is None:
                    last_err = fallback_err

        if _p is None:
            err_str = str(last_err) if last_err else ""
            log.error("Pipeline.from_pretrained failed. Last error:\n%s",
                      traceback.format_exc() if last_err else "returned None (no exception)")
            if "403" in err_str or "gated" in err_str.lower():
                _log_access_denied(ui_log, err_str)
            elif "401" in err_str:
                _log_token_rejected(ui_log)
            elif last_err is not None:
                ui_log("  ERROR: Pipeline loading failed: %s" % last_err)
                ui_log("  Check logs/transcriber.log for details.")
            else:
                ui_log("")
                ui_log("  ERROR: Pipeline loaded no data (returned None).")
                ui_log("")
                ui_log("  This usually means the download succeeded but")
                ui_log("  pyannote could not parse the pipeline config.")
                ui_log("")
                ui_log("  Possible fixes:")
                ui_log("  1. Delete cached models and re-download:")
                ui_log("     rm -rf ~/.cache/huggingface/hub/models--pyannote*")
                ui_log("  2. Ensure you accepted BOTH model licenses:")
                ui_log("     https://huggingface.co/pyannote/speaker-diarization-3.1")
                ui_log("     https://huggingface.co/pyannote/segmentation-3.0")
                ui_log("  3. Check logs/transcriber.log for detailed diagnostics.")
                ui_log("")
            return "\n".join(lines)

        del _p
        ui_log("  OK - Pyannote pipeline ready.")
    except Exception as e:
        msg = str(e)
        log.error("Pyannote download traceback:\n%s", traceback.format_exc())
        if "403" in msg or "gated" in msg.lower():
            _log_access_denied(ui_log, msg)
        elif "401" in msg:
            _log_token_rejected(ui_log)
        else:
            ui_log("  ERROR: Pyannote download failed: %s" % e)
            ui_log("  Check logs/transcriber.log for details.")
        return "\n".join(lines)

    # 3 — ffmpeg
    progress(0.85, desc="Checking ffmpeg...")
    ui_log("Checking ffmpeg...")
    try:
        ver = _check_ffmpeg()
        ui_log("  OK - %s" % ver)
    except RuntimeError as e:
        ui_log("  WARNING - %s" % e)

    # Save the token to config so it auto-loads next time
    save_token(hf_token)
    ui_log("  Token saved to config.json — it will auto-fill next time.")

    progress(1.0, desc="Done!")
    ui_log("\nAll set! Switch to Single Video or Batch tab to transcribe.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def get_whisper_model(model_size: str = "large-v3"):
    global _whisper_model, _whisper_model_size, _whisper_backend
    if _whisper_model is None or _whisper_model_size != model_size:
        import torch
        
        # Check if we should use DirectML + openai-whisper
        use_dml = False
        try:
            import torch_directml
            if torch_directml.is_available() and not torch.cuda.is_available():
                use_dml = True
        except ImportError:
            pass

        if use_dml:
            log.info("Loading OpenAI Whisper '%s' on GPU (DirectML)...", model_size)
            import whisper
            import torch_directml
            
            model = whisper.load_model(model_size, device="cpu")
            
            # Convert any sparse buffers to dense (e.g. alignment_heads) to prevent DirectML crash
            for name, buf in list(model.named_buffers()):
                if buf is not None and buf.is_sparse:
                    log.debug("Converting sparse buffer '%s' to dense for DirectML compatibility", name)
                    dense_buf = buf.to_dense()
                    parts = name.split('.')
                    submodule = model
                    for part in parts[:-1]:
                        submodule = getattr(submodule, part)
                    submodule.register_buffer(parts[-1], dense_buf)
                    
            device = torch_directml.device()
            _whisper_model = model.to(device)
            
            # DirectML compatibility patches
            if hasattr(_whisper_model, "alignment_heads"):
                log.info("Restoring alignment_heads to CPU sparse layout for DirectML timing compatibility")
                _whisper_model.alignment_heads = _whisper_model.alignment_heads.to("cpu").to_sparse()
                
            try:
                import whisper.timing
                orig_median_filter = whisper.timing.median_filter
                def patched_median_filter(x, filter_width):
                    orig_device = x.device
                    x_cpu = x.to("cpu")
                    res_cpu = orig_median_filter(x_cpu, filter_width)
                    return res_cpu.to(orig_device)
                whisper.timing.median_filter = patched_median_filter
                log.info("Patched whisper.timing.median_filter to CPU to bypass DirectML reflect pad bug")
            except Exception as patch_err:
                log.warning("Could not patch whisper.timing.median_filter: %s", patch_err)
                
            _whisper_backend = "openai-whisper"
        elif torch.cuda.is_available():
            WhisperModel = _import_whisper()
            log.info("Loading Whisper '%s' on GPU (cuda, float16)...", model_size)
            _whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
            _whisper_backend = "faster-whisper"
        else:
            WhisperModel = _import_whisper()
            log.info("Loading Whisper '%s' on CPU (int8)...", model_size)
            _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            _whisper_backend = "faster-whisper"
            
        _whisper_model_size = model_size
        log.info("Whisper model loaded using %s backend.", _whisper_backend)
    return _whisper_model


def get_diarization_pipeline(hf_token: str):
    global _diarization_pipeline
    if _diarization_pipeline is None:
        torch = _import_torch()
        Pipeline = _import_diarization()
        import huggingface_hub as _hfhub_mod
        log.info("Loading pyannote speaker-diarization-3.1...")

        # Set token globally so from_pretrained picks it up automatically
        _hfhub_mod.login(token=hf_token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = hf_token

        local_cache = MODELS_DIR / "pyannote"
        cache_kw = {}
        if local_cache.exists():
            cache_kw["cache_dir"] = str(local_cache)

        # Detect which token kwarg from_pretrained accepts
        import inspect
        _fp_params = list(inspect.signature(Pipeline.from_pretrained).parameters.keys())
        if "token" in _fp_params:
            token_kw = {"token": hf_token}
        elif "use_auth_token" in _fp_params:
            token_kw = {"use_auth_token": hf_token}
        else:
            token_kw = {}

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            **token_kw,
            **cache_kw,
        )

        # Fallback: try downloading config.yaml directly and loading from path
        if pipeline is None:
            log.warning("from_pretrained returned None, trying direct path fallback...")
            try:
                config_path = _hfhub_mod.hf_hub_download(
                    "pyannote/speaker-diarization-3.1",
                    "config.yaml",
                    token=hf_token,
                )
                if config_path:
                    pipeline = Pipeline.from_pretrained(config_path, **token_kw)
            except Exception as e:
                log.error("Direct path fallback failed: %s", e)

        if pipeline is None:
            raise RuntimeError(
                "Pyannote pipeline failed to load (returned None).\n\n"
                "Try these fixes:\n\n"
                "  1. Delete cached models and re-download:\n"
                "     rm -rf ~/.cache/huggingface/hub/models--pyannote*\n"
                "     Then go to Setup tab and click Download Models.\n\n"
                "  2. Ensure you accepted BOTH model licenses:\n"
                "     https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "     https://huggingface.co/pyannote/segmentation-3.0\n\n"
                "  3. Check logs/transcriber.log for detailed diagnostics.\n\n"
                "  4. Your HuggingFace token may be invalid or expired —\n"
                "     create a new one at https://huggingface.co/settings/tokens"
            )

        # Determine the best available PyTorch device (CUDA -> MPS -> DirectML -> CPU)
        device_str = "cpu"
        device_obj = torch.device("cpu")
        
        if torch.cuda.is_available():
            device_str = "cuda"
            device_obj = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
            device_obj = torch.device("mps")
        else:
            try:
                import torch_directml
                if torch_directml.is_available():
                    # Note: Pyannote uses an LSTM-heavy architecture (PyanNet) which crashes on DirectML
                    # due to missing native LSTM operators (aten::_thnn_fused_lstm_cell).
                    # Since diarization is lightweight, we fall back to CPU.
                    log.info("DirectML detected. Falling back to CPU for Pyannote LSTM compatibility.")
                    device_str = "cpu"
                    device_obj = torch.device("cpu")
            except ImportError:
                pass
                
        log.info("Diarization device: %s", device_str)
        if device_str != "cpu":
            pipeline.to(device_obj)
        _diarization_pipeline = pipeline
    return _diarization_pipeline


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------
def extract_audio(video_path: str, output_path: str) -> str:
    _check_ffmpeg()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n%s" % r.stderr[-500:])
    return output_path


# ---------------------------------------------------------------------------
# Speaker diarization
# ---------------------------------------------------------------------------
def run_diarization(
    audio_path: str,
    hf_token: str,
    num_speakers: Optional[int] = None,
    video_path: Optional[str] = None,
) -> List[Dict]:
    import hashlib
    # Generate cache key based on video path if available, else audio path
    cache_file = None
    try:
        ref_path = Path(video_path) if video_path else Path(audio_path)
        stat = ref_path.stat()
        key_str = f"{ref_path.name}_{stat.st_size}_{stat.st_mtime}_{num_speakers}"
        cache_key = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        cache_dir = APP_DIR / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{cache_key}_diarization.json"
        
        if cache_file.exists():
            log.info("Found cached diarization results at %s. Loading...", cache_file)
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_segments = json.load(f)
            log.info("Loaded %d segments from diarization cache.", len(cached_segments))
            return cached_segments
    except Exception as cache_err:
        log.warning("Could not check/load diarization cache: %s", cache_err)

    pipeline = get_diarization_pipeline(hf_token)
    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    
    # Pre-load audio into memory with soundfile to bypass buggy/missing torchcodec library on Windows
    log.info("Pre-loading audio into memory with soundfile to bypass torchcodec...")
    try:
        import soundfile as sf
        import torch
        waveform_np, sample_rate = sf.read(audio_path, always_2d=False, dtype='float32')
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.ndim == 1:
            waveform = waveform[None, :]  # Shape: (1, time)
        elif waveform.ndim == 2:
            waveform = waveform.T  # Shape: (channels, time)
        audio_input = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        log.info("Audio loaded successfully into memory. Waveform shape: %s, Sample rate: %d", waveform.shape, sample_rate)
    except Exception as e:
        log.warning("Failed to pre-load audio with soundfile: %s. Falling back to file path.", e)
        audio_input = audio_path

    log.debug("Running diarization on audio_input (num_speakers=%s)...", num_speakers)
    diarization = pipeline(audio_input, **kwargs)
    
    # Support pyannote.audio 4.x+ where pipeline returns a DiarizeOutput object
    # instead of an Annotation object directly.
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    if cache_file:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=4)
            log.info("Saved diarization results to cache: %s", cache_file)
        except Exception as cache_err:
            log.warning("Could not save diarization cache: %s", cache_err)

    return segments


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def run_transcription(
    audio_path: str,
    model_size: str,
    language: Optional[str] = None,
    progress_fn=None,
) -> Tuple[List[Dict], str]:
    model = get_whisper_model(model_size)
    
    audio_dur = _get_audio_duration(audio_path)
    
    if _whisper_backend == "openai-whisper":
        kw = {
            "word_timestamps": True,
            "fp16": False,  # Crucial: DirectML doesn't support fp16 well for Whisper
        }
        if language and language != "auto":
            kw["language"] = language
            
        log.info("Starting Whisper transcription on GPU (DirectML)...")
        if progress_fn and audio_dur > 0:
            progress_fn(0.1, 0, audio_dur)
            
        result = model.transcribe(audio_path, **kw)
        
        detected_lang = result.get("language", "en")
        log.info("Detected language: %s", detected_lang)
        
        segments = []
        raw_segs = result.get("segments", [])
        for seg in raw_segs:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })
            
        if progress_fn and audio_dur > 0:
            progress_fn(1.0, audio_dur, audio_dur)
            
        return segments, detected_lang
        
    else:
        kw = {
            "word_timestamps": True,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
        }
        if language and language != "auto":
            kw["language"] = language
            
        segments_gen, info = model.transcribe(audio_path, **kw)
        info_dur = info.duration if hasattr(info, "duration") and info.duration else 0
        if info_dur > 0:
            audio_dur = info_dur
            
        log.info("Detected language: %s (%.0f%%), duration=%.1fs",
                 info.language, info.language_probability * 100, audio_dur)
                 
        segments = []
        for seg in segments_gen:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            # Report per-segment progress
            if progress_fn and audio_dur > 0:
                pct = min(seg.end / audio_dur, 1.0)
                progress_fn(pct, seg.end, audio_dur)
                
        return segments, info.language


# ---------------------------------------------------------------------------
# Merge transcription + diarization
# ---------------------------------------------------------------------------
def assign_speakers(transcription_segments, diarization_segments):
    results = []
    for t in transcription_segments:
        t_start, t_end = t["start"], t["end"]
        best_speaker, best_overlap = "UNKNOWN", 0.0
        for d in diarization_segments:
            overlap = max(0.0, min(t_end, d["end"]) - max(t_start, d["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]
        if best_speaker == "UNKNOWN":
            mid = (t_start + t_end) / 2
            for d in diarization_segments:
                if d["start"] <= mid <= d["end"]:
                    best_speaker = d["speaker"]
                    break
        results.append({
            "start": t_start, "end": t_end,
            "speaker": best_speaker, "text": t["text"],
        })
    return results


# ---------------------------------------------------------------------------
# SRT generation
# ---------------------------------------------------------------------------
def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def generate_srt(merged_segments: List[Dict]) -> str:
    blocks = []
    for i, seg in enumerate(merged_segments, 1):
        speaker = seg["speaker"].replace("SPEAKER_", "Speaker ")
        blocks.append(
            "%d\n%s --> %s\n[%s] %s\n"
            % (i, _srt_ts(seg["start"]), _srt_ts(seg["end"]), speaker, seg["text"])
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def transcribe_video(
    video_path,
    hf_token,
    model_size="large-v3",
    language="auto",
    num_speakers=0,
    output_dir="",
    progress=gr.Progress(),
):
    """Generator that yields (status, live_text, srt_preview, srt_file) tuples.

    The live_text field updates in real-time as segments are transcribed,
    showing detected speech scrolling in the UI.
    """
    # Outputs: status, live_text, srt_preview (hidden until done), srt_file
    if not video_path:
        yield "No video file provided.", "", "", None
        return
    if not hf_token or not hf_token.strip().startswith("hf_"):
        hf_token = ""
    else:
        hf_token = hf_token.strip()

    num_speakers = int(num_speakers)
    video_name = Path(video_path).stem

    log.info("=" * 40)
    log.info("Starting transcription: %s", Path(video_path).name)
    log.info("  model=%s  language=%s  num_speakers=%d", model_size, language, num_speakers)

    if output_dir and os.path.isdir(output_dir):
        save_dir = output_dir
    else:
        save_dir = str(OUTPUT_DIR)
        os.makedirs(save_dir, exist_ok=True)

    import time as _time

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        t0 = _time.time()

        # ── Step 1/4: Extract audio ──────────────────────────────────
        progress(0.0, desc="Step 1/4 — Extracting audio from video...")
        yield "Step 1/4 — Extracting audio from video...", "", "", None
        log.info("Step 1/4: Extracting audio...")
        try:
            extract_audio(video_path, audio_path)
            audio_mb = os.path.getsize(audio_path) / 1e6
            audio_dur = _get_audio_duration(audio_path)
            dur_str = "%d:%02d" % (int(audio_dur) // 60, int(audio_dur) % 60)
            log.info("  Audio extracted (%.1f MB, %s)", audio_mb, dur_str)
            progress(0.05, desc="Step 1/4 — Audio extracted (%s, %.0f MB)" % (dur_str, audio_mb))
        except Exception as e:
            log.error("Audio extraction failed:\n%s", traceback.format_exc())
            yield "Audio extraction failed:\n%s" % e, "", "", None
            return

        # ── Step 2/4: Speaker diarization ────────────────────────────
        if hf_token:
            progress(0.08, desc="Step 2/4 — Identifying speakers...")
            yield ("Step 2/4 — Identifying speakers...\n"
                   "  Video: %s (%s)\n"
                   "  This usually takes 1-3 minutes on CPU."
                   % (Path(video_path).name, dur_str)), "", "", None
            log.info("Step 2/4: Running speaker diarization...")
            try:
                diar_segs = run_diarization(
                    audio_path, hf_token,
                    num_speakers=num_speakers if num_speakers > 0 else None,
                    video_path=video_path,
                )
                n_speakers = len(set(s["speaker"] for s in diar_segs))
                elapsed = _time.time() - t0
                log.info("  Found %d segments, %d speakers (%.0fs elapsed)",
                         len(diar_segs), n_speakers, elapsed)
                progress(0.35, desc="Step 2/4 — Found %d speakers" % n_speakers)
            except Exception as e:
                log.error("Speaker diarization failed:\n%s", traceback.format_exc())
                yield "Speaker diarization failed:\n%s" % e, "", "", None
                return
        else:
            diar_segs = []
            n_speakers = 0
            log.info("Step 2/4: Skipping speaker diarization (no HuggingFace token provided).")
            progress(0.35, desc="Step 2/4 — Skipping speaker diarization (no HuggingFace token)")

        # ── Step 3/4: Transcription with live text ───────────────────
        progress(0.38, desc="Step 3/4 — Loading Whisper %s..." % model_size)
        status_3 = ("Step 3/4 — Transcribing with Whisper %s...\n"
                     "  Found %d speakers, %d diarization segments\n"
                     "  Audio: %s" % (model_size, n_speakers, len(diar_segs), dur_str))
        yield status_3, "Waiting for first segment...", "", None
        log.info("Step 3/4: Transcribing with Whisper %s...", model_size)

        # We'll consume the generator segment by segment so we can yield
        # live text updates to the UI after each segment.
        live_lines = []
        try:
            model = get_whisper_model(model_size)
            if _whisper_backend == "openai-whisper":
                kw = {
                    "word_timestamps": False,
                    "fp16": False,  # Crucial: DirectML doesn't support fp16 well for Whisper
                }
                if language and language != "auto":
                    kw["language"] = language

                log.info("Starting Whisper transcription on GPU (DirectML)...")
                progress(0.40, desc="Step 3/4 — Transcribing with Whisper (DirectML GPU)...")
                
                result = model.transcribe(audio_path, **kw)
                
                detected_lang = result.get("language", "en")
                log.info("Detected language: %s", detected_lang)
                
                trans_segs = []
                raw_segs = result.get("segments", [])
                for seg in raw_segs:
                    trans_segs.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                    })
                    ts = "%d:%02d" % (int(seg["start"]) // 60, int(seg["start"]) % 60)
                    live_lines.append("[%s] %s" % (ts, seg["text"].strip()))

                progress(0.85, desc="Step 3/4 — Transcription completed.")
                live_display = "\n".join(live_lines[-50:])
                yield status_3, live_display, "", None
                
            else:
                kw = {
                    "word_timestamps": True,
                    "vad_filter": True,
                    "vad_parameters": {"min_silence_duration_ms": 500},
                }
                lang_arg = language if language != "auto" else None
                if lang_arg:
                    kw["language"] = lang_arg
                segments_gen, info = model.transcribe(audio_path, **kw)
                t_audio_dur = info.duration if hasattr(info, "duration") and info.duration else audio_dur
                if t_audio_dur <= 0:
                    t_audio_dur = audio_dur
                log.info("Detected language: %s (%.0f%%)",
                         info.language, info.language_probability * 100)
                detected_lang = info.language

                trans_segs = []
                for seg in segments_gen:
                    trans_segs.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                    })
                    # Format timestamp for display
                    ts = "%d:%02d" % (int(seg.start) // 60, int(seg.start) % 60)
                    live_lines.append("[%s] %s" % (ts, seg.text.strip()))
                    # Update progress bar
                    if t_audio_dur > 0:
                        pct = min(seg.end / t_audio_dur, 1.0)
                        pos_str = "%d:%02d" % (int(seg.end) // 60, int(seg.end) % 60)
                        tot_str = "%d:%02d" % (int(t_audio_dur) // 60, int(t_audio_dur) % 60)
                        overall = 0.40 + pct * 0.45
                        progress(overall,
                                 desc="Step 3/4 — Transcribing %s / %s (%d%%)"
                                 % (pos_str, tot_str, int(pct * 100)))
                    # Yield live text (show last 50 lines to keep it scrollable)
                    live_display = "\n".join(live_lines[-50:])
                    yield status_3, live_display, "", None

            elapsed = _time.time() - t0
            log.info("  Transcribed %d segments, language: %s (%.0fs elapsed)",
                     len(trans_segs), detected_lang, elapsed)
        except Exception as e:
            log.error("Transcription failed:\n%s", traceback.format_exc())
            yield "Transcription failed:\n%s" % e, "\n".join(live_lines) if live_lines else "", "", None
            return

    # ── Step 4/4: Merge + generate SRT ───────────────────────────
    progress(0.87, desc="Step 4/4 — Assigning speakers to segments...")
    yield "Step 4/4 — Assigning speakers to text...", "\n".join(live_lines[-50:]), "", None
    log.info("Step 4/4: Merging speakers + transcription...")
    merged = assign_speakers(trans_segs, diar_segs)

    progress(0.92, desc="Step 4/4 — Writing SRT file...")
    srt_content = generate_srt(merged)

    srt_path = os.path.join(save_dir, "%s_transcribed.srt" % video_name)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    total_elapsed = _time.time() - t0
    elapsed_str = "%d:%02d" % (int(total_elapsed) // 60, int(total_elapsed) % 60)
    speakers = sorted(set(s["speaker"] for s in merged))
    speaker_list = ", ".join(s.replace("SPEAKER_", "Speaker ") for s in speakers)

    # Build a nice live view with speaker labels from the final SRT
    final_lines = []
    for seg in merged:
        ts = "%d:%02d" % (int(seg["start"]) // 60, int(seg["start"] ) % 60)
        spk = seg["speaker"].replace("SPEAKER_", "Speaker ")
        final_lines.append("[%s] [%s] %s" % (ts, spk, seg["text"]))

    summary = (
        "Transcription complete!\n\n"
        "  Video:    %s\n"
        "  Duration: %s\n"
        "  Language: %s\n"
        "  Speakers: %d (%s)\n"
        "  Segments: %d\n"
        "  Time:     %s\n"
        "  SRT saved: %s\n"
        % (Path(video_path).name, dur_str, detected_lang,
           len(speakers), speaker_list, len(merged), elapsed_str, srt_path)
    )

    log.info("Done! %d speakers, %d segments, %s elapsed → %s",
             len(speakers), len(merged), elapsed_str, srt_path)
    progress(1.0, desc="Done! %d speakers, %d segments (%s)" % (len(speakers), len(merged), elapsed_str))
    yield summary, "\n".join(final_lines), srt_content, srt_path


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def transcribe_batch(
    files,
    hf_token,
    model_size,
    language,
    num_speakers,
    output_dir="",
    progress=gr.Progress(),
):
    if not files:
        return "No files provided."
    num_speakers = int(num_speakers)
    results = []
    for i, file_obj in enumerate(files):
        path = file_obj if isinstance(file_obj, str) else str(file_obj)
        name = Path(path).name
        progress(i / len(files), desc="%s (%d/%d)..." % (name, i + 1, len(files)))
        summary, _, _ = transcribe_video(
            path, hf_token, model_size, language, num_speakers, output_dir, progress,
        )
        results.append("--- %s ---\n%s" % (name, summary))
    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    # Load saved token from config.json
    saved_token = load_config().get("hf_token", "")

    with gr.Blocks(title="Conference Video Transcriber", theme=gr.themes.Soft()) as app:

        gr.Markdown(
            "# Conference Video Transcriber\n"
            "Transcribe conference videos locally with automatic speaker separation. "
            "Everything runs on your Mac — no data leaves your machine."
        )

        # ---- Setup tab ----
        with gr.Tab("Setup / Download Models"):
            gr.Markdown(
                "### First-time setup (do this once)\n\n"
                "**Step 1 — Create a HuggingFace account** (free):  \n"
                "Go to [huggingface.co/join](https://huggingface.co/join) and sign up.\n\n"
                "**Step 2 — Create an access token:**  \n"
                "Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens), "
                "click **Create new token**, name it anything (e.g. \"transcriber\"), "
                "set type to **Read** and make sure **\"Access to public gated repos\"** is enabled. "
                "Then copy the token (starts with `hf_`).\n\n"
                "**Step 3 — Accept the model licenses** (click \"Agree and access\" on each page):  \n"
                "- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)  \n"
                "- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)\n\n"
                "**Step 4 — Paste your token below and click Download Models.**  \n"
                "This downloads ~3-4 GB of AI models into this app's folder. Only needed once."
            )
            with gr.Row():
                with gr.Column():
                    setup_token = gr.Textbox(
                        label="HuggingFace Token", placeholder="hf_...", type="password",
                        value=saved_token,
                    )
                    setup_model = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        value="medium",
                        label="Whisper Model to Download",
                    )
                    setup_btn = gr.Button("Download Models", variant="primary", size="lg")
                with gr.Column():
                    setup_log = gr.Textbox(label="Download Log", lines=18, interactive=False)
            setup_btn.click(fn=download_models, inputs=[setup_token, setup_model], outputs=[setup_log])

        # ---- Single video tab ----
        with gr.Tab("Single Video"):
            gr.Markdown(
                "Upload a video, paste your HuggingFace token, and click **Transcribe**. "
                "The output is an SRT subtitle file with speaker labels."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload or drag-drop a conference video")
                    hf_token_input = gr.Textbox(
                        label="HuggingFace Token", placeholder="hf_...", type="password",
                        value=saved_token,
                        info="Same token from the Setup tab. Get one at huggingface.co/settings/tokens",
                    )
                    with gr.Row():
                        model_size_input = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                            value="medium", label="Whisper Model",
                            info="large-v3 = best quality (~3 GB). medium = faster (~1.5 GB).",
                        )
                        language_input = gr.Dropdown(
                            choices=["auto", "en", "el", "de", "fr", "es", "it", "pt", "nl", "ja", "zh", "ko"],
                            value="auto", label="Language",
                            info="Auto-detect usually works. Pick manually if you know.",
                        )
                    num_speakers_input = gr.Slider(
                        minimum=0, maximum=20, step=1, value=0,
                        label="Number of Speakers (0 = auto-detect)",
                        info="Set this if you know how many people speak — improves accuracy.",
                    )
                    output_dir_input = gr.Textbox(
                        label="Output Directory (optional)",
                        placeholder="Default: %s" % str(OUTPUT_DIR),
                        info="Leave empty to save SRT files in the app's output/ folder.",
                    )
                    transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
                with gr.Column(scale=1):
                    status_out = gr.Textbox(label="Status", lines=6, interactive=False)
                    live_text = gr.Textbox(
                        label="Live Transcription",
                        lines=12, interactive=False, autoscroll=True,
                        info="Text appears here as each segment is transcribed.",
                    )
                    srt_preview = gr.Textbox(label="SRT Output", lines=15, interactive=False)
                    srt_file = gr.File(label="Download SRT File")
            transcribe_btn.click(
                fn=transcribe_video,
                inputs=[video_input, hf_token_input, model_size_input, language_input, num_speakers_input, output_dir_input],
                outputs=[status_out, live_text, srt_preview, srt_file],
            )

        # ---- Batch tab ----
        with gr.Tab("Batch Processing"):
            gr.Markdown(
                "Upload multiple videos at once. They will be processed one after another. "
                "SRT files are saved to the output directory with the same name as each video."
            )
            batch_files = gr.File(label="Upload Videos", file_count="multiple", file_types=["video"])
            with gr.Row():
                b_token = gr.Textbox(label="HuggingFace Token", placeholder="hf_...", type="password", value=saved_token)
                b_model = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    value="medium", label="Whisper Model",
                )
                b_lang = gr.Dropdown(
                    choices=["auto", "en", "el", "de", "fr", "es", "it", "pt", "nl", "ja", "zh", "ko"],
                    value="auto", label="Language",
                )
                b_speakers = gr.Slider(minimum=0, maximum=20, step=1, value=0, label="Speakers (0=auto)")
            b_outdir = gr.Textbox(label="Output Directory (optional)", placeholder="Default: %s" % str(OUTPUT_DIR))
            batch_btn = gr.Button("Transcribe All", variant="primary", size="lg")
            batch_out = gr.Textbox(label="Results", lines=20, interactive=False)
            batch_btn.click(
                fn=transcribe_batch,
                inputs=[batch_files, b_token, b_model, b_lang, b_speakers, b_outdir],
                outputs=[batch_out],
            )

        gr.Markdown(
            "---\n"
            "**How it works:** Audio is extracted from your video, then two AI models run locally: "
            "one identifies *who* is speaking (pyannote), the other converts speech to *text* (Whisper). "
            "The results are merged into a single SRT subtitle file with speaker labels like `[Speaker 00]`.\n\n"
            "**Troubleshooting:**  \n"
            "- *Diarization failed (401/403)* — your token is wrong or you haven't accepted the model licenses in the Setup tab  \n"
            "- *Slow transcription* — try the `medium` model instead of `large-v3`  \n"
            "- *Wrong language detected* — set the language dropdown manually instead of auto  \n"
            "- *Speakers mixed up* — set the exact number of speakers if you know it"
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    # We run in CLI mode if CLI arguments are provided
    if len(sys.argv) > 1 and sys.argv[1] not in ("--gui", "gui"):
        parser = argparse.ArgumentParser(
            description="Conference Video Transcriber (CLI Mode)"
        )
        parser.add_argument("-v", "--video", type=str, required=True, help="Path to video file")
        parser.add_argument("-t", "--token", type=str, default="", help="HuggingFace Token (starts with hf_). If omitted, loads from config.json")
        parser.add_argument("-m", "--model", type=str, default="medium", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"], help="Whisper model size")
        parser.add_argument("-l", "--language", type=str, default="auto", help="Language (e.g. en, el, de, auto)")
        parser.add_argument("-s", "--speakers", type=int, default=0, help="Number of speakers (0 = auto-detect)")
        parser.add_argument("-o", "--output-dir", type=str, default="", help="Output directory for SRT file")
        parser.add_argument("--setup", action="store_true", help="Pre-download and cache models before transcribing")

        args = parser.parse_args()

        # 1. Resolve token
        hf_token = args.token.strip() if args.token else load_config().get("hf_token", "")
        if not hf_token or not hf_token.startswith("hf_"):
            print("WARNING: No valid HuggingFace token provided. Speaker diarization will be skipped.")
            print("To enable speaker diarization, specify a token with -t/--token or run the GUI setup once.")
            hf_token = ""

        # 2. Check setup
        if args.setup:
            print(f"Checking/Downloading models (Model: {args.model})...")
            class SetupProgress:
                def __call__(self, value, desc=""):
                    print(f"[{int(value*100)}%] {desc}", flush=True)
            log_res = download_models(hf_token, args.model, progress=SetupProgress())
            print(log_res)
            print("-" * 50)

        # 3. Perform transcription
        print(f"Initializing transcription pipeline for: {args.video}")
        class CLIProgress:
            def __call__(self, value, desc=""):
                percent = int(value * 100) if value is not None else 0
                print(f"[PROGRESS] {percent}%: {desc}", flush=True)

        try:
            # Consume the generator to completion
            final_step = None
            for step in transcribe_video(
                video_path=args.video,
                hf_token=hf_token,
                model_size=args.model,
                language=args.language,
                num_speakers=args.speakers,
                output_dir=args.output_dir,
                progress=CLIProgress(),
            ):
                final_step = step

            if final_step:
                summary, final_lines, srt_content, srt_path = final_step
                print("\n" + "=" * 60)
                print(summary)
                print("=" * 60)
                print(f"SUCCESS: SRT file saved to {srt_path}")
            else:
                print("ERROR: Transcription pipeline produced no results.")
                sys.exit(1)
        except Exception as err:
            print(f"\nERROR running transcription: {err}")
            traceback.print_exc()
            sys.exit(1)

    else:
        # Default Gradio GUI mode
        app = build_ui()
        app.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


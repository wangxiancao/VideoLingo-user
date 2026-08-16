import argparse
import json
import os
import sys
from importlib.metadata import distribution
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start the vLLM-Omni OpenAI server for IndexTTS2."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18721)
    parser.add_argument("--model", default="IndexTeam/IndexTTS-2")
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument(
        "--deploy-name",
        default="indextts2.yaml",
        help="vLLM-Omni deploy config basename, e.g. cosyvoice3.yaml",
    )
    return parser.parse_args()


def default_deploy_config(name="indextts2.yaml"):
    package_root = Path(str(distribution("vllm-omni").locate_file("vllm_omni")))
    path = package_root / "deploy" / name
    if not path.exists():
        raise FileNotFoundError(f"vLLM-Omni deploy config not found: {path}")
    return path


def prepare_legacy_local_checkpoint(model):
    """Adapt the upstream IndexTTS2 checkpoint layout without duplicating assets."""
    model_path = Path(model).expanduser().resolve()
    if not model_path.is_dir():
        return model
    if not (model_path / "config.yaml").exists() or not (model_path / "gpt.pth").exists():
        return str(model_path)

    legacy_cache = model_path / "hf_cache"
    vocoder_dir = legacy_cache / "bigvgan"
    config_path = model_path / "config.json"
    generated_config = {
        "architectures": ["IndexTTS2TalkerForConditionalGeneration"],
        "model_type": "indextts2",
    }
    if vocoder_dir.is_dir():
        generated_config["vocoder"] = {
            "type": "bigvgan",
            "name": str(vocoder_dir.resolve()),
        }
    config_path.write_text(
        json.dumps(generated_config, indent=2) + "\n",
        encoding="utf-8",
    )

    local_assets = {
        model_path / "wav2vec2bert": legacy_cache / "w2v-bert-2.0",
        model_path / "campplus.pth": legacy_cache / "campplus_cn_common.bin",
    }
    for destination, source in local_assets.items():
        if not destination.exists() and source.exists():
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())

    semantic_target = model_path / "semantic_codec.pth"
    semantic_source = legacy_cache / "semantic_codec" / "model.safetensors"
    if not semantic_target.exists() and semantic_source.exists():
        import torch
        from safetensors.torch import load_file

        torch.save(load_file(semantic_source, device="cpu"), semantic_target)

    return str(model_path)


def main():
    args = parse_args()
    args.model = prepare_legacy_local_checkpoint(args.model)
    deploy_config = (
        Path(args.deploy_config).expanduser().resolve()
        if args.deploy_config
        else default_deploy_config(args.deploy_name)
    )
    vllm_executable = Path(sys.executable).with_name("vllm")
    if not vllm_executable.exists():
        raise FileNotFoundError(
            f"vLLM executable not found beside the active Python: {vllm_executable}"
        )

    command = [
        str(vllm_executable),
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--omni",
        "--trust-remote-code",
        "--deploy-config",
        str(deploy_config),
    ]
    environment = os.environ.copy()
    environment.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    # This host's system nvcc is CUDA 10.1 while the vLLM wheel is CUDA 12.9.
    # Use vLLM's PyTorch sampler so FlashInfer does not JIT against old nvcc.
    environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    compat_dir = Path(__file__).resolve().parent / "vllm_omni_py310_compat"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{compat_dir}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(compat_dir)
    )
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()

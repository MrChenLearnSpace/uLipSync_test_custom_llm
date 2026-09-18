"""Export the trained Transformer to the fixed ONNX tensor contract used by Unity."""

import argparse
from pathlib import Path

import torch

from model import LipSyncTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="artifacts/lipsync_transformer.onnx")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    input_size = checkpoint["input_size"]
    model = LipSyncTransformer(
        len(checkpoint["output_labels"]),
        hidden_size=checkpoint["hidden_size"],
        layers=checkpoint["layers"],
        heads=checkpoint["heads"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example_input = torch.zeros(1, input_size, dtype=torch.float32)

    # ONNX INPUT: audio_pcm, float32 [1, 1024] by default.
    # ONNX OUTPUT: viseme_logits, float32 [1, 5] by default in [A, I, U, E, O] order.
    # AiLipSyncTransformer applies softmax before forwarding the values to uLipSync consumers.
    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=["audio_pcm"],
        output_names=["viseme_logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported {output_path} with labels: {checkpoint['output_labels']}")


if __name__ == "__main__":
    main()

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _ensure_src_on_path() -> None:
    # eval.py lives at: src/umm/eval/internvl_chat/eval/mme/eval.py
    # Add src/ to sys.path so `import umm` works regardless of cwd.
    src_root = Path(__file__).resolve().parents[5]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


def post_processing(response: str) -> str:
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response


def infer_backbone_name(explicit_backbone: str, checkpoint: str) -> str:
    if explicit_backbone:
        return explicit_backbone
    ckpt = (checkpoint or "").lower()
    if "janus" in ckpt:
        return "janus_pro"
    if "bagel" in ckpt:
        return "bagel"
    raise ValueError(
        "Cannot infer backbone from checkpoint path. Please pass --backbone "
        "(e.g., janus_pro, bagel, show_o, emu3, omnigen2, blip3o, tokenflow)."
    )


def extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "answer", "response", "output", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        results = output.get("results")
        if isinstance(results, dict):
            for key in ("text", "answer", "response", "output"):
                value = results.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(results, list):
            for item in results:
                text = extract_text(item)
                if text:
                    return text
    if isinstance(output, list):
        for item in output:
            text = extract_text(item)
            if text:
                return text
    return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--root', type=str, default='./Your_Results')
    parser.add_argument('--backbone', type=str, default='')
    parser.add_argument('--max-new-tokens', type=int, default=20)
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--auto', action='store_true')  # keep compatibility with evaluate.sh
    parser.add_argument('--dynamic', action='store_true')  # compatibility no-op for adapter inference
    parser.add_argument('--max-num', type=int, default=6)  # compatibility no-op
    args = parser.parse_args()

    _ensure_src_on_path()
    from umm.inference.pipeline import InferencePipeline

    backbone_name = infer_backbone_name(args.backbone, args.checkpoint)
    backbone_cfg = {'model_path': args.checkpoint} if args.checkpoint else {}
    pipeline = InferencePipeline(backbone_name=backbone_name, backbone_cfg=backbone_cfg)

    output = os.path.basename(args.checkpoint) if args.checkpoint else backbone_name
    os.makedirs(output, exist_ok=True)
    prompt_suffix = 'Answer the question using a single word or phrase.'

    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(output, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')
            question = question + ' ' + prompt_suffix
            img_path = os.path.join('../../data/mme/MME_Benchmark_release_version', filename, img)
            assert os.path.exists(img_path), img_path

            result = pipeline.run(
                {
                    "backbone": backbone_name,
                    "task": "understanding",
                    "prompt": question,
                    "images": [img_path],
                    "params": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": bool(args.do_sample),
                    },
                }
            )
            response = post_processing(extract_text(result))
            print(img, question, gt, response, sep='\t', file=fout)
        fin.close()
        fout.close()

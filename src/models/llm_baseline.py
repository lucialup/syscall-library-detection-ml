import os
import json
import random
import glob
from pathlib import Path
from typing import List, Set
import google.generativeai as genai
from tqdm import tqdm

from src.data.parser import parse_line, normalize_thread, is_noise, normalize_path
from src.config import TARGET_LIBRARIES, DataConfig

TEST_LIBRARY = "Room"


def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('models/gemini-2.5-flash')


def generate_app_summary(log_path: Path) -> str:
    unique_threads: Set[str] = set()
    unique_paths: Set[str] = set()
    syscall_sequence: List[str] = []

    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                record = parse_line(line)
                if not record:
                    continue

                path = record.get("path", "")
                comm = record.get("comm", "")
                syscall = record.get("syscall", "")

                if is_noise(path):
                    continue

                clean_thread = normalize_thread(comm)
                if clean_thread:
                    unique_threads.add(clean_thread)

                clean_path = normalize_path(path)
                if clean_path and clean_path not in ["", "."]:
                    if any(x in clean_path for x in [".db", "cache", "lib", "xml", "preferences"]):
                        unique_paths.add(clean_path)

                if len(syscall_sequence) < 50 and syscall:
                    syscall_sequence.append(f"{clean_thread}:{syscall}")

    except Exception as e:
        return f"Error reading trace: {e}"

    threads = ', '.join(sorted(unique_threads))
    files = '\n'.join(sorted(unique_paths)[:50])
    syscalls = ' -> '.join(syscall_sequence[:20])

    return f"Threads: {threads}\n\nFiles:\n{files}\n\nSyscalls:\n{syscalls}"


"""Check if library is present in SBOM."""
def get_ground_truth(sbom_path: Path, target_lib: str) -> bool:
    if not sbom_path.exists():
        return False

    signatures = TARGET_LIBRARIES.get(target_lib, [])

    try:
        with open(sbom_path, 'r', encoding='utf-8') as f:
            sbom = json.load(f)

        libraries = sbom.get('libraries', [])

        for lib in libraries:
            package = lib.get('package', '') or ''
            name = lib.get('name', '') or ''

            for sig in signatures:
                sig = sig.lower()
                if sig in package.lower() or sig in name.lower():
                    return True

    except Exception:
        return False

    return False


def build_prompt(summary: str, lib_name: str) -> str:
    return f"""
You are analyzing a runtime system call trace from an Android application.

TASK: Determine if the third-party library "{lib_name}" is present based only on the behavioral patterns in the trace below.

TRACE DATA:
{summary}

INSTRUCTIONS:
- Analyze thread names, file access patterns, and syscall sequences for behavioral signatures
- Consider what distinct runtime behaviors this specific library would exhibit
- Look for patterns in thread naming conventions, file I/O operations, and resource access
- Do not rely solely on naive substring matching (e.g., thread name containing library name)
- Reason about the library's PURPOSE and what syscall-level behaviors that purpose would produce

ANSWER FORMAT:
Return ONLY a JSON object: {{"verdict": "YES" or "NO", "confidence": "HIGH" or "LOW", "reason": "behavioral evidence observed"}}
"""


def run_experiment():
    model = setup_gemini()
    config = DataConfig()
    all_logs = glob.glob(str(config.syscall_dir / "*.log"))

    pos_samples = []
    neg_samples = []

    for log_file in tqdm(all_logs, desc="Preparing dataset"):
        p = Path(log_file)
        package_name = p.name.replace(".syscall.log", "")
        sbom_file = config.sbom_dir / f"{package_name}.sbom.json"

        has_lib = get_ground_truth(sbom_file, TEST_LIBRARY)

        if has_lib:
            pos_samples.append((p, sbom_file))
        else:
            neg_samples.append((p, sbom_file))

        if len(pos_samples) >= 10 and len(neg_samples) >= 10:
            break

    pilot_batch = pos_samples[:10] + neg_samples[:10]
    random.shuffle(pilot_batch)

    correct = 0

    for log_path, sbom_path in tqdm(pilot_batch, desc=f"Testing {TEST_LIBRARY}"):
        actual = get_ground_truth(sbom_path, TEST_LIBRARY)
        summary = generate_app_summary(log_path)
        prompt = build_prompt(summary, TEST_LIBRARY)
        response = model.generate_content(prompt)

        try:
            txt = response.text.strip().replace("```json", "").replace("```", "")
            result = json.loads(txt)
            verdict = result.get("verdict", "NO").upper()
        except:
            verdict = "ERROR"

        predicted = (verdict == "YES")
        if predicted == actual:
            correct += 1

    accuracy = correct / len(pilot_batch)
    print(f"\n{TEST_LIBRARY}: {correct}/{len(pilot_batch)} ({accuracy:.1%})")


if __name__ == "__main__":
    run_experiment()
import os
import json
import time
import glob
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime
import google.generativeai as genai
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from src.data.parser import parse_line, normalize_thread, is_noise, normalize_path
from src.config import TARGET_LIBRARIES, DataConfig

RESULTS_DIR = Path("results/llm_app_by_app")


def build_prompt(library_names: List[str], summary: str) -> str:
    library_list_str = ", ".join(library_names)
    json_entries = [f'  "{lib}": "YES" or "NO"' for lib in library_names]
    json_format = "{\n" + ",\n".join(json_entries) + "\n}"

    return f"""You are analyzing a runtime system call trace from an Android application.

TASK: Determine which of the following {len(library_names)} third-party libraries are present based ONLY on the patterns in the trace below.

LIBRARIES TO CHECK (alphabetically ordered):
{library_list_str}

TRACE DATA:
{summary}

INSTRUCTIONS:
- Analyze the thread names, file access patterns, and syscall sequences
- For EACH library, determine if it's present based on patterns you observe
- Use any distinguishing patterns in the data to make your determination
- Be thorough and consider all available evidence

ANSWER FORMAT (JSON only, no markdown):
{json_format}
"""


def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('models/gemini-2.5-flash')


def get_ground_truth_all_libraries(sbom_path: Path) -> Dict[str, bool]:
    result = {lib: False for lib in TARGET_LIBRARIES.keys()}

    if not sbom_path.exists():
        return result

    try:
        with open(sbom_path, 'r', encoding='utf-8') as f:
            sbom = json.load(f)
        libraries = sbom.get('libraries', [])

        for lib_name, signatures in TARGET_LIBRARIES.items():
            for lib in libraries:
                package = lib.get('package', '') or ''
                name = lib.get('name', '') or ''
                for sig in signatures:
                    if sig.lower() in package.lower() or sig.lower() in name.lower():
                        result[lib_name] = True
                        break
                if result[lib_name]:
                    break
    except:
        pass

    return result


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
                    if any(x in clean_path for x in [
                        ".db", "cache", "lib", "xml", "preferences",
                        ".so", ".json", "app_flutter", "files", "assets"
                    ]):
                        unique_paths.add(clean_path)

                if len(syscall_sequence) < 40 and syscall:
                    syscall_sequence.append(f"{clean_thread}:{syscall}")

    except Exception as e:
        return f"Error: {e}"

    sorted_paths = sorted(list(unique_paths))
    if len(sorted_paths) > 60:
        priority = [p for p in sorted_paths if any(x in p for x in [".so", ".db", "cache"])]
        remainder = [p for p in sorted_paths if p not in priority]
        final_paths = priority + remainder
        final_paths = final_paths[:60]
    else:
        final_paths = sorted_paths

    threads = ', '.join(sorted(list(unique_threads)))
    files = '\n    '.join(final_paths)
    syscalls = ' -> '.join(syscall_sequence[:20])

    return f"""
    Active Threads ({len(unique_threads)}): {threads}

    Relevant Files ({len(final_paths)}):
    {files}

    Syscall Head:
    {syscalls}
    """


def run_benchmark():
    """Run zero-shot LLM inference on all apps for multi-library detection."""
    model = setup_gemini()
    config = DataConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    library_names = sorted(TARGET_LIBRARIES.keys())
    app_results_file = RESULTS_DIR / "app_predictions.jsonl"
    summary_file = RESULTS_DIR / "evaluation_summary.json"

    all_logs = glob.glob(str(config.syscall_dir / "*.syscall.log"))
    print(f"Running LLM evaluation on {len(all_logs)} apps")
    print(f"Model: gemini-2.5-flash | Results: {RESULTS_DIR}\n")

    processed_packages = set()
    if app_results_file.exists():
        with open(app_results_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                    processed_packages.add(pred['package'])
                except:
                    pass
        if processed_packages:
            print(f"Resuming from {len(processed_packages)} already processed apps\n")

    all_predictions = []

    for log_path in tqdm(all_logs, desc="Processing apps"):
        log_path = Path(log_path)
        package_name = log_path.stem.replace(".syscall", "")
        sbom_path = config.sbom_dir / f"{package_name}.sbom.json"

        if package_name in processed_packages:
            continue

        ground_truth = get_ground_truth_all_libraries(sbom_path)
        summary = generate_app_summary(log_path)
        prompt = build_prompt(library_names, summary)

        try:
            response = model.generate_content(prompt)
            txt = response.text.strip()

            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0]
            elif "```" in txt:
                txt = txt.split("```")[1].split("```")[0]

            predictions_dict = json.loads(txt)

            normalized_predictions = {}
            for lib in library_names:
                if lib in predictions_dict:
                    normalized_predictions[lib] = (predictions_dict[lib].upper() == "YES")
                else:
                    found = False
                    for key, value in predictions_dict.items():
                        if key.lower() == lib.lower():
                            normalized_predictions[lib] = (value.upper() == "YES")
                            found = True
                            break
                    if not found:
                        normalized_predictions[lib] = False

        except Exception as e:
            normalized_predictions = {lib: False for lib in library_names}
            time.sleep(2)

        prediction = {
            "package": package_name,
            "ground_truth": ground_truth,
            "predictions": normalized_predictions,
            "timestamp": datetime.now().isoformat()
        }
        all_predictions.append(prediction)

        with open(app_results_file, 'a') as f:
            f.write(json.dumps(prediction) + '\n')

        time.sleep(0.5)

    all_predictions = []
    with open(app_results_file) as f:
        for line in f:
            try:
                all_predictions.append(json.loads(line))
            except:
                pass

    print(f"\nCalculating metrics...")

    per_library_results = []

    for lib_name in library_names:
        y_true = []
        y_pred = []

        for pred in all_predictions:
            gt = pred['ground_truth'].get(lib_name, False)
            p = pred['predictions'].get(lib_name, False)
            y_true.append(1 if gt else 0)
            y_pred.append(1 if p else 0)

        # Calculate metrics using sklearn
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        lib_result = {
            "library": lib_name,
            "sample_size": len(all_predictions),
            "positive_samples": sum(y_true),
            "negative_samples": len(y_true) - sum(y_true),
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            },
            "confusion_matrix": {
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn)
            }
        }
        per_library_results.append(lib_result)

        print(f"{lib_name:20s} → F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}")

    mean_f1 = sum(r["metrics"]["f1"] for r in per_library_results) / len(per_library_results)
    mean_precision = sum(r["metrics"]["precision"] for r in per_library_results) / len(per_library_results)
    mean_recall = sum(r["metrics"]["recall"] for r in per_library_results) / len(per_library_results)
    mean_accuracy = sum(r["metrics"]["accuracy"] for r in per_library_results) / len(per_library_results)

    summary = {
        "timestamp": timestamp,
        "model": "gemini-2.5-flash",
        "approach": "app-by-app-multi-library",
        "prompt_type": "zero-shot-no-hints",
        "total_apps": len(all_predictions),
        "total_libraries": len(library_names),
        "total_classifications": len(all_predictions) * len(library_names),
        "mean_f1": mean_f1,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_accuracy": mean_accuracy,
        "per_library": per_library_results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete:")
    print(f"Apps: {len(all_predictions)} | Libraries: {len(per_library_results)}")
    print(f"Mean F1: {mean_f1:.3f} | Precision: {mean_precision:.3f} | Recall: {mean_recall:.3f}")
    print(f"Results: {RESULTS_DIR}\n")


if __name__ == "__main__":
    run_benchmark()

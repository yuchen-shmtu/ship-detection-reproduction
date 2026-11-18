"""Train and test a queue of baselines, one run at a time: train_detector.py, then eval_detector.py.

Jobs are (model, protocol, seed), ordered seed by seed so every model gets seed 0 on both
protocols before any model gets seed 1. The next job started is the first one in that
order whose weights/<model> exists, so training goes on while fetch_weights.py is still
downloading. With no job ready it waits; it stops when every job is done or failed.

A run whose eval/summary.json exists is skipped, so the queue can be restarted. A run dir
without it (a crashed training) makes train_detector.py fail on exist_ok=False: the job is
reported failed and the queue moves on; delete the dir to redo it.

Logs go to runs/logs/<run>.log and eval_<run>.log. One job at a time is a memory rule (see
train_detector.py), so nothing else heavy should run next to this.

    python src/models/run_suite.py --models yolo26n.pt rtdetr-l.pt yolov8n.pt --seeds 0 1 2
    python src/models/run_suite.py --models faster-rcnn-r50 ssd300-vgg16
    python src/models/run_suite.py --models rtdetr-r18 --aboships-imgsz 640
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS, LOGS = ROOT / "runs" / "seaships", ROOT / "runs" / "logs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--protocols", nargs="+", default=["random", "session"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smd-imgsz", nargs="+", default=["640", "1280"])
    ap.add_argument("--aboships-imgsz", nargs="*", default=[], help="passed to eval_detector.py; none (default): skip")
    ap.add_argument("--max-wait", type=float, default=6, help="hours to wait for missing weights before giving up")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)

    todo = [(m, p, s) for s in a.seeds for m in a.models for p in a.protocols]
    failed, waited = [], 0.0
    while todo:
        # names without .pt are torchvision archs (tv_detector.py), whose weights are in weights/torch_hub/
        ready = [j for j in todo if not j[0].endswith(".pt") or (ROOT / "weights" / j[0]).exists()]
        if not ready:
            if waited > a.max_wait * 3600:
                failed += ["%s_%s_s%d (no weights)" % (Path(m).stem, p, s) for m, p, s in todo]
                break
            time.sleep(60)
            waited += 60
            continue
        waited = 0.0
        m, p, s = job = ready[0]
        todo.remove(job)
        run = "%s_%s_s%d" % (Path(m).stem, p, s)
        if (RUNS / run / "eval" / "summary.json").exists():
            print(run, "already evaluated, skipped", flush=True)
            continue
        t0 = time.time()
        with open(LOGS / (run + ".log"), "w", encoding="utf-8") as log:
            rc = subprocess.run([sys.executable, str(ROOT / "src" / "models" / "train_detector.py"), "--model", m,
                                 "--protocol", p, "--seed", str(s)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        t1 = time.time()
        if rc == 0:
            with open(LOGS / ("eval_%s.log" % run), "w", encoding="utf-8") as log:
                rc = subprocess.run([sys.executable, str(ROOT / "src" / "evaluation" / "eval_detector.py"), "--run", run,
                                     "--smd-imgsz", *a.smd_imgsz, "--aboships-imgsz", *a.aboships_imgsz], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT).returncode
            print("%s train %.0f min, eval exit %d (%.0f min)" % (run, (t1 - t0) / 60, rc, (time.time() - t1) / 60),
                  flush=True)
        else:
            print("%s train exit %d after %.0f min" % (run, rc, (t1 - t0) / 60), flush=True)
        if rc:
            failed.append(run)
    print("done; failed:", failed or "none", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

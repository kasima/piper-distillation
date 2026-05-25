"""Orchestrate Phase 2 wait -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6.

Designed to run as a long-lived nohup process. Survives shell exit.
Writes status to state/orchestrate_status.json so the operator can poll progress.

Halt conditions surface to state/notifications.txt; the script exits non-zero.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
VENV = RUN / ".venv"
LOG = RUN / "logs/orchestrate.log"
STATUS = RUN / "state/orchestrate_status.json"
NOTIFY = RUN / "state/notifications.txt"

CORPUS_SIZE = 12000


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def notify(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with NOTIFY.open("a") as f:
        f.write(f"[{ts}] {msg}\n")


def write_status(d: dict) -> None:
    d["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    STATUS.write_text(json.dumps(d, indent=2))


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> int:
    log(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    return proc.returncode


def wait_for_phase2() -> None:
    log("waiting for Phase 2 synthesis to finish")
    done_file = RUN / "state/phase2/done.txt"
    pid_file = RUN / "state/phase2/pid"
    last_done = -1
    last_change = time.time()
    while True:
        try:
            n_done = sum(1 for _ in done_file.open()) if done_file.exists() else 0
        except FileNotFoundError:
            n_done = 0
        if n_done >= CORPUS_SIZE:
            log(f"Phase 2 done ({n_done} clips)")
            return
        # Check process still alive
        alive = False
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                alive = True
            except (ProcessLookupError, ValueError, OSError):
                alive = False
        if n_done > last_done:
            last_done = n_done
            last_change = time.time()
        idle_min = (time.time() - last_change) / 60
        if not alive and n_done < CORPUS_SIZE:
            notify(f"Phase 2 process not running but only {n_done}/{CORPUS_SIZE} done — restarting")
            log(f"restarting Phase 2 (process died, {n_done}/{CORPUS_SIZE})")
            relaunch_phase2()
            time.sleep(15)
            continue
        if idle_min > 60:
            notify(f"Phase 2 stalled — no progress for {idle_min:.0f} min at {n_done}/{CORPUS_SIZE}; restarting")
            log(f"Phase 2 stalled — kill PID and restart")
            try:
                if pid_file.exists():
                    os.kill(int(pid_file.read_text().strip()), 15)
            except Exception:
                pass
            time.sleep(5)
            relaunch_phase2()
            last_change = time.time()
            time.sleep(15)
            continue
        write_status({
            "phase": "phase2_wait", "done": n_done,
            "total": CORPUS_SIZE,
            "pct": round(100 * n_done / CORPUS_SIZE, 1),
        })
        time.sleep(60)


def relaunch_phase2() -> None:
    log("relaunching Phase 2")
    out = RUN / "logs/phase2.log"
    cmd = [str(VENV / "bin/python3"), str(RUN / "scripts/phase2_synthesize.py")]
    with out.open("a") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(RUN),
            stdout=logf, stderr=logf,
            start_new_session=True,
        )
    (RUN / "state/phase2/pid").write_text(f"{proc.pid}\n")
    log(f"Phase 2 relaunched PID {proc.pid}")


def run_phase3() -> None:
    manifest_path = RUN / "state/phase3/manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        log(f"Phase 3 already done (manifest present): retained {m.get('retained')} clips ({m.get('retained_hours', 0):.2f}h) — skipping")
        return
    log("=== Phase 3: QA filter ===")
    write_status({"phase": "phase3", "stage": "filtering"})
    rc = run_cmd([str(VENV / "bin/python3"), str(RUN / "scripts/phase3_filter.py")], cwd=RUN)
    if rc != 0:
        notify("Phase 3 filter failed (non-zero exit)")
        raise SystemExit(2)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("below_hard_floor"):
        notify(f"Phase 3 retained only {manifest['retained_hours']:.2f}h — below 2h hard floor. Halt.")
        raise SystemExit(3)
    log(f"Phase 3 done: retained {manifest['retained']} clips ({manifest['retained_hours']:.2f}h)")


def stop_vllm_aeon() -> None:
    log("stopping vllm-aeon for GPU 1")
    rc = subprocess.run(["sudo", "-n", "systemctl", "stop", "vllm-aeon"]).returncode
    if rc != 0:
        notify("Could not stop vllm-aeon (sudo non-zero)")
        raise SystemExit(4)
    time.sleep(10)
    # Confirm GPU 1 freed
    out = subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    used = int(out)
    log(f"GPU 1 used: {used} MiB")
    if used > 2000:
        notify(f"GPU 1 still has {used} MiB used after vllm-aeon stop")
        raise SystemExit(5)


def start_vllm_aeon() -> None:
    log("restarting vllm-aeon")
    subprocess.run(["sudo", "-n", "systemctl", "start", "vllm-aeon"])
    time.sleep(5)
    subprocess.run(["sudo", "-n", "systemctl", "is-active", "vllm-aeon"])


def run_phase4_prep() -> None:
    prep_manifest = RUN / "state/phase4/prep_manifest.json"
    if prep_manifest.exists():
        log("Phase 4 prep already done — skipping")
        return
    log("=== Phase 4 prep: train/eval split ===")
    write_status({"phase": "phase4_prep"})
    rc = run_cmd([str(VENV / "bin/python3"), str(RUN / "scripts/phase4_prepare.py")], cwd=RUN)
    if rc != 0:
        notify("Phase 4 prep failed")
        raise SystemExit(6)


def run_phase4_train() -> None:
    # Skip if Run A already completed (40k-step checkpoint exists).
    ckpt_dir = RUN / "phase4/checkpoints"
    if list(ckpt_dir.glob("*step=40000*")) or list(ckpt_dir.glob("*step=4????*")):
        log(f"Run A already completed (checkpoints in {ckpt_dir}) — skipping")
        return
    # If a piper-train-takashii unit is already active (e.g., we restarted
    # orchestrator mid-training), DO NOT clobber it — just monitor.
    st = subprocess.run(["systemctl", "--user", "is-active", "piper-train-takashii"],
                        capture_output=True, text=True)
    if st.stdout.strip() == "active":
        log("piper-train-takashii unit already active — monitoring existing run")
        write_status({"phase": "phase4_train", "stage": "monitoring_existing"})
    else:
        log("=== Phase 4: train ===")
        write_status({"phase": "phase4_train", "stage": "launching"})
        # Clear any prior failed unit so systemd-run can re-register
        subprocess.run(["systemctl", "--user", "reset-failed", "piper-train-takashii"],
                       capture_output=True)
        rc = run_cmd(["bash", str(RUN / "scripts/phase4_train.sh")])
        if rc != 0:
            notify("Phase 4 train launch failed")
            raise SystemExit(7)
    log("waiting for piper-train-takashii unit to complete (this can take 24-72h)")
    while True:
        st = subprocess.run(
            ["systemctl", "--user", "is-active", "piper-train-takashii"],
            capture_output=True, text=True,
        )
        state = st.stdout.strip()
        write_status({"phase": "phase4_train", "stage": state})
        if state in ("inactive", "failed"):
            break
        time.sleep(600)  # 10 min cadence


def pick_top_checkpoints(k: int = 3) -> list[Path]:
    ckpt_dir = RUN / "phase4/checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    # MVP: pick last k checkpoints (best converged + last 2 prior for variance check)
    return ckpts[-k:] if ckpts else []


def run_phase5() -> None:
    log("=== Phase 5: eval ===")
    write_status({"phase": "phase5"})
    ckpts = pick_top_checkpoints(3)
    if not ckpts:
        notify("Phase 5 cannot find any checkpoints in phase4/checkpoints/")
        raise SystemExit(8)
    cmd = [str(VENV / "bin/python3"), str(RUN / "scripts/phase5_eval.py")]
    for c in ckpts:
        cmd.extend(["--checkpoint", str(c)])
    rc = run_cmd(cmd, cwd=RUN)
    if rc != 0:
        notify("Phase 5 eval failed")
        raise SystemExit(9)


def run_phase6() -> None:
    """Phase 6 for Run A — install winning curated checkpoint into piper service."""
    log("=== Phase 6: Run A install ===")
    write_status({"phase": "phase6"})
    p5 = json.loads((RUN / "state/phase5/manifest.json").read_text())
    winner = p5["winner"]
    metrics_json = Path(winner).parent.parent.parent / "state/phase5" / Path(winner).stem / "metrics.json"
    rc = run_cmd([
        str(VENV / "bin/python3"), str(RUN / "scripts/phase6_install.py"),
        "--checkpoint", winner,
        "--voice-name", "en_US-takashii-medium",
        "--training-config", str(RUN / "phase4/config.json"),
        "--dataset-label", "takashii_curated_8143",
        "--metrics-json", str(metrics_json),
        "--no-restart-service",
    ], cwd=RUN)
    if rc != 0:
        notify("Phase 6 (Run A) install failed")
        raise SystemExit(10)


def run_phase4b_train() -> None:
    """Run B: train on FULL dataset (curated + recovered WER drops)."""
    # Skip if Run B already completed (40k-step checkpoint exists).
    ckpt_dir = RUN / "phase4-full/checkpoints"
    if list(ckpt_dir.glob("*step=40000*")) or list(ckpt_dir.glob("*step=4????*")):
        log(f"Run B already completed — skipping")
        return
    st = subprocess.run(["systemctl", "--user", "is-active", "piper-train-takashii-full"],
                        capture_output=True, text=True)
    if st.stdout.strip() == "active":
        log("piper-train-takashii-full already active — monitoring existing run")
    else:
        log("=== Run B: train on full dataset ===")
        write_status({"phase": "phase4b_train", "stage": "launching"})
        # Phase 4 prep B
        run_cmd([str(VENV / "bin/python3"), str(RUN / "scripts/phase4b_prepare.py")], cwd=RUN)
        subprocess.run(["systemctl", "--user", "reset-failed", "piper-train-takashii-full"],
                       capture_output=True)
        rc = run_cmd(["bash", str(RUN / "scripts/phase4_train_full.sh")])
        if rc != 0:
            notify("Run B train launch failed")
            raise SystemExit(11)
    log("waiting for piper-train-takashii-full unit to complete")
    while True:
        st = subprocess.run(["systemctl", "--user", "is-active", "piper-train-takashii-full"],
                            capture_output=True, text=True)
        state = st.stdout.strip()
        write_status({"phase": "phase4b_train", "stage": state})
        if state in ("inactive", "failed"):
            break
        time.sleep(600)


def run_phase5b() -> None:
    log("=== Run B Phase 5: eval ===")
    write_status({"phase": "phase5b"})
    ckpt_dir = RUN / "phase4-full/checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)[-3:]
    if not ckpts:
        notify("Run B Phase 5: no checkpoints found")
        raise SystemExit(12)
    # phase5_eval.py hardcodes phase4/eval_metadata.csv + state/phase5/.
    # For Run B we want phase4-full/eval_metadata.csv + state/phase5-full/.
    # Easiest: run via env override + a tiny wrapper.
    env = dict(os.environ)
    env["PIPER_DISTILL_EVAL_VARIANT"] = "full"
    cmd = [str(VENV / "bin/python3"), str(RUN / "scripts/phase5_eval.py")]
    for c in ckpts:
        cmd.extend(["--checkpoint", str(c)])
    rc = subprocess.run(cmd, cwd=str(RUN), env=env).returncode
    if rc != 0:
        notify("Run B Phase 5 eval failed")
        raise SystemExit(12)


def run_phase6b() -> None:
    """Install Run B's Phase 5 winner as en_US-takashii-medium-full."""
    log("=== Run B Phase 6: install ===")
    write_status({"phase": "phase6b"})
    p5b_manifest_path = RUN / "state/phase5-full/manifest.json"
    if p5b_manifest_path.exists():
        p5 = json.loads(p5b_manifest_path.read_text())
        winner = p5["winner"]
        metrics_json = RUN / "state/phase5-full" / Path(winner).stem / "metrics.json"
    else:
        # fallback to last checkpoint
        ckpt_dir = RUN / "phase4-full/checkpoints"
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        winner = str(ckpts[-1]) if ckpts else None
        metrics_json = None
    if not winner:
        notify("Run B Phase 6: no winner / no checkpoints")
        raise SystemExit(13)
    args = [
        str(VENV / "bin/python3"), str(RUN / "scripts/phase6_install.py"),
        "--checkpoint", winner,
        "--voice-name", "en_US-takashii-medium-full",
        "--training-config", str(RUN / "phase4-full/config.json"),
        "--dataset-label", "takashii_full_11713",
        "--no-restart-service",
    ]
    if metrics_json and Path(metrics_json).exists():
        args.extend(["--metrics-json", str(metrics_json)])
    rc = run_cmd(args, cwd=RUN)
    if rc != 0:
        notify("Run B Phase 6 install failed")
        raise SystemExit(13)


def restart_piper_service() -> None:
    log("restarting piper.service so it picks up new voices")
    subprocess.run(["sudo", "-n", "systemctl", "restart", "piper"])


def run_phase4c_train() -> None:
    """Run C: train LOW-quality (16 kHz) on the full dataset."""
    ckpt_dir = RUN / "phase4-low/checkpoints"
    # Exact-step match (the previous *step=3????* glob falsely matched step=3000)
    if list(ckpt_dir.glob("*step=30000.ckpt")):
        log("Run C already completed — skipping")
        return
    st = subprocess.run(["systemctl", "--user", "is-active", "piper-train-takashii-low"],
                        capture_output=True, text=True)
    if st.stdout.strip() == "active":
        log("piper-train-takashii-low already active — monitoring existing run")
    else:
        log("=== Run C: train (LOW quality) ===")
        write_status({"phase": "phase4c_train", "stage": "launching"})
        subprocess.run(["systemctl", "--user", "reset-failed", "piper-train-takashii-low"],
                       capture_output=True)
        rc = run_cmd(["bash", str(RUN / "scripts/phase4_train_low.sh")])
        if rc != 0:
            notify("Run C train launch failed")
            raise SystemExit(15)
        # Also kick off the Run C watchdog as a sibling process
        wd_log = (RUN / "logs/watchdog_c.log").open("a")
        subprocess.Popen(
            ["bash", str(RUN / "scripts/phase4c_watchdog.sh")],
            stdout=wd_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log("Run C watchdog spawned")
    log("waiting for piper-train-takashii-low (30k steps, ~2-3h)")
    while True:
        # Exact match — prior glob *step=3????* falsely matched step=3000.ckpt
        if list(ckpt_dir.glob("*step=30000.ckpt")):
            log("Run C target reached")
            break
        st = subprocess.run(["systemctl", "--user", "is-active", "piper-train-takashii-low"],
                            capture_output=True, text=True)
        state = st.stdout.strip()
        write_status({"phase": "phase4c_train", "stage": state})
        # Watchdog handles restarts; we just wait until either target OR
        # watchdog signals exhaustion via notifications. Poll at 5-min cadence.
        time.sleep(300)


def run_phase5c() -> None:
    log("=== Run C Phase 5: eval (low) ===")
    write_status({"phase": "phase5c"})
    ckpt_dir = RUN / "phase4-low/checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)[-3:]
    if not ckpts:
        notify("Run C Phase 5: no checkpoints found")
        raise SystemExit(16)
    env = dict(os.environ)
    env["PIPER_DISTILL_EVAL_VARIANT"] = "low"
    cmd = [str(VENV / "bin/python3"), str(RUN / "scripts/phase5_eval.py")]
    for c in ckpts:
        cmd.extend(["--checkpoint", str(c)])
    rc = subprocess.run(cmd, cwd=str(RUN), env=env).returncode
    if rc != 0:
        notify("Run C Phase 5 eval failed")
        raise SystemExit(16)


def run_phase6c() -> None:
    log("=== Run C Phase 6: install (low) ===")
    write_status({"phase": "phase6c"})
    p5c_manifest_path = RUN / "state/phase5-low/manifest.json"
    if p5c_manifest_path.exists():
        p5 = json.loads(p5c_manifest_path.read_text())
        winner = p5["winner"]
        metrics_json = RUN / "state/phase5-low" / Path(winner).stem / "metrics.json"
    else:
        ckpt_dir = RUN / "phase4-low/checkpoints"
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        winner = str(ckpts[-1]) if ckpts else None
        metrics_json = None
    if not winner:
        notify("Run C Phase 6: no winner / no checkpoints")
        raise SystemExit(17)
    args = [
        str(VENV / "bin/python3"), str(RUN / "scripts/phase6_install.py"),
        "--checkpoint", winner,
        "--voice-name", "en_US-takashii-low",
        "--training-config", str(RUN / "phase4-low/config.json"),
        "--dataset-label", "takashii_full_11713_16khz",
        "--no-restart-service",
    ]
    if metrics_json and Path(metrics_json).exists():
        args.extend(["--metrics-json", str(metrics_json)])
    rc = run_cmd(args, cwd=RUN)
    if rc != 0:
        notify("Run C Phase 6 install failed")
        raise SystemExit(17)


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("orchestrator start")
    write_status({"phase": "starting"})
    try:
        wait_for_phase2()
        # Stop vllm-aeon before Phase 3 — whisper medium needs ~2 GB on GPU 1
        # alongside ECAPA + WavLM, and 46 GB are parked by vllm-aeon. Moving the
        # stop earlier just extends the LLM downtime by ~30 min (Phase 3 duration).
        stop_vllm_aeon()
        run_phase3()
        run_phase4_prep()
        run_phase4_train()   # Run A: curated (8143 clips)
        run_phase5()         # eval Run A
        run_phase6()         # install Run A → en_US-takashii-medium
        # ---- A/B: do Run B before restarting vllm-aeon ----
        run_phase4b_train()  # Run B: full (11713 clips)
        run_phase5b()        # eval Run B
        run_phase6b()        # install Run B → en_US-takashii-medium-full
        # ---- Run C: low-quality 16 kHz ----
        run_phase4c_train()  # Run C: low-quality 16 kHz, same full dataset
        run_phase5c()        # eval Run C
        run_phase6c()        # install Run C → en_US-takashii-low
        restart_piper_service()  # one restart, all three voices picked up
        start_vllm_aeon()
        log("orchestrator finished successfully (Run A + B + C)")
        write_status({"phase": "complete"})
        notify("All three runs complete. Voices live: en_US-takashii-medium, en_US-takashii-medium-full, en_US-takashii-low")
    except SystemExit as e:
        log(f"orchestrator exited with code {e.code}")
        # Even on failure, try to restart vllm-aeon if it was stopped
        try:
            start_vllm_aeon()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

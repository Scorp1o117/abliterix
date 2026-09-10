# Unified-memory suicide switch for Strix Halo / large UMA boxes.
#
# systemd-oomd prefers desktop scopes (Edge, terminals) over the Python job
# that actually ate the pool. A 1s MemAvailable poll is too slow for a
# single forward that spikes tens of GB. This watcher therefore:
#   * polls ~10 Hz
#   * trips on PSI memory pressure and swap, not just MemAvailable
#   * is started from SteeringEngine so scripts cannot skip it
#
# Pair with a systemd --user --scope MemoryMax= on the job so the kernel
# refuses the allocation instead of harvesting the desktop.
#
# Env:
#   ABLITERIX_UMA_GUARD=0          disable
#   ABLITERIX_UMA_MIN_FREE_GB=20   suicide if system available drops below
#   ABLITERIX_UMA_MAX_RSS_GB=88    suicide if this process RSS exceeds
#   ABLITERIX_UMA_POLL_SEC=0.1
#   ABLITERIX_UMA_PSI_SOME_AVG10=20  suicide if /proc/pressure/memory some avg10
#   ABLITERIX_UMA_MAX_SWAP_GB=0.5    suicide if this process swap exceeds

from __future__ import annotations

import os
import sys
import threading
import time
import traceback

_started = False
_lock = threading.Lock()


def _read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            key, raw, *_rest = line.split()
            out[key.rstrip(":")] = int(raw) * 1024
    return out


def _read_self_status() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/self/status", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(("VmRSS:", "VmSwap:")):
                out[line.split(":")[0]] = int(line.split()[1]) * 1024
    return out


def _read_self_rss() -> int:
    return _read_self_status().get("VmRSS", 0)


def _read_psi_some_avg10() -> float | None:
    try:
        with open("/proc/pressure/memory", encoding="utf-8") as fh:
            line = fh.readline()
    except OSError:
        return None
    # some avg10=0.00 avg60=0.00 avg300=0.00 total=...
    if not line.startswith("some "):
        return None
    for part in line.split():
        if part.startswith("avg10="):
            return float(part.split("=", 1)[1])
    return None


def _fmt_gb(n: int) -> str:
    return f" {n / (1024**3):.1f}G"


def _torch_reserved() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_reserved(0))
    except Exception:
        return None
    return None


def snapshot() -> str:
    info = _read_meminfo()
    avail = info.get("MemAvailable", 0)
    total = info.get("MemTotal", 0)
    rss = _read_self_rss()
    reserved = _torch_reserved()
    bits = [
        f"avail={_fmt_gb(avail).strip()}/{_fmt_gb(total).strip()}",
        f"rss={_fmt_gb(rss).strip()}",
    ]
    if reserved is not None:
        bits.append(f"cuda_reserved={_fmt_gb(reserved).strip()}")
    return " ".join(bits)


def _suicide(reason: str) -> None:
    print(f"\n[uma_guard] SUICIDE: {reason} | {snapshot()}", file=sys.stderr, flush=True)
    sys.stderr.flush()
    # Hard exit so we cannot keep allocating during unwind.
    # Do not os.sync() — under reclaim it stalls and lets oomd win.
    os._exit(137)


def _watch(
    min_free: int,
    max_rss: int,
    max_swap: int,
    psi_avg10: float,
    poll: float,
) -> None:
    warn_sent = False
    while True:
        try:
            info = _read_meminfo()
            avail = info.get("MemAvailable", 0)
            status = _read_self_status()
            rss = status.get("VmRSS", 0)
            swap = status.get("VmSwap", 0)
            psi = _read_psi_some_avg10()
            if avail < min_free:
                _suicide(
                    f"MemAvailable {_fmt_gb(avail).strip()} "
                    f"< min_free {_fmt_gb(min_free).strip()}"
                )
            if rss > max_rss:
                _suicide(
                    f"VmRSS {_fmt_gb(rss).strip()} "
                    f"> max_rss {_fmt_gb(max_rss).strip()}"
                )
            if swap > max_swap:
                _suicide(
                    f"VmSwap {_fmt_gb(swap).strip()} "
                    f"> max_swap {_fmt_gb(max_swap).strip()}"
                )
            if psi is not None and psi >= psi_avg10:
                _suicide(f"PSI some avg10={psi:.2f} >= {psi_avg10:.2f}")
            if not warn_sent and (
                avail < min_free + 8 * 1024**3
                or rss > max_rss - 8 * 1024**3
                or (psi is not None and psi >= psi_avg10 * 0.5)
            ):
                print(f"[uma_guard] approaching limit | {snapshot()}", flush=True)
                warn_sent = True
        except Exception:
            traceback.print_exc()
        time.sleep(poll)


def start_uma_guard(
    *,
    min_free_gb: float | None = None,
    max_rss_gb: float | None = None,
    poll_sec: float | None = None,
) -> None:
    """Start the watcher once per process. Safe to call repeatedly."""
    global _started
    if os.environ.get("ABLITERIX_UMA_GUARD", "1") in {"0", "false", "off"}:
        return
    with _lock:
        if _started:
            return
        _started = True

    min_free = int(
        (min_free_gb if min_free_gb is not None else float(os.environ.get("ABLITERIX_UMA_MIN_FREE_GB", "20")))
        * 1024**3
    )
    max_rss = int(
        (max_rss_gb if max_rss_gb is not None else float(os.environ.get("ABLITERIX_UMA_MAX_RSS_GB", "88")))
        * 1024**3
    )
    max_swap = int(float(os.environ.get("ABLITERIX_UMA_MAX_SWAP_GB", "0.5")) * 1024**3)
    psi_avg10 = float(os.environ.get("ABLITERIX_UMA_PSI_SOME_AVG10", "20"))
    poll = float(poll_sec if poll_sec is not None else os.environ.get("ABLITERIX_UMA_POLL_SEC", "0.1"))

    thread = threading.Thread(
        target=_watch,
        args=(min_free, max_rss, max_swap, psi_avg10, poll),
        name="uma-guard",
        daemon=True,
    )
    thread.start()
    print(
        f"[uma_guard] on  min_free={_fmt_gb(min_free).strip()}  "
        f"max_rss={_fmt_gb(max_rss).strip()}  poll={poll:.2f}s  "
        f"psi_avg10>={psi_avg10:.0f}  max_swap={_fmt_gb(max_swap).strip()}  "
        f"| {snapshot()}",
        flush=True,
    )

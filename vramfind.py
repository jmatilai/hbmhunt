#!/usr/bin/env python3
"""Physical VRAM address of hbmhunt's marker pages, and CPU re-reads of a page.

usage (root):
  vramfind.py find   DRI END_MIB          print physical addresses of pages starting "HBMHUNT:"
  vramfind.py reread DRI BDF ADDR [N]     read the 4 KiB page at ADDR N times, print CE delta
DRI is the card's directory under /sys/kernel/debug/dri: its PCI address (kernel >= 6.7) or minor number.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PAGE = 4096
MAGIC = b'HBMHUNT:'


def find(dri, end_mib):
    fd = os.open(f'/sys/kernel/debug/dri/{dri}/amdgpu_vram', os.O_RDONLY)
    pages = (end_mib << 20) // PAGE
    threads = 16
    step = -(-pages // threads)

    def part(t):
        hits = []
        for pg in range(t * step, min(pages, (t + 1) * step)):
            head = os.pread(fd, 64, pg * PAGE)
            if head.startswith(MAGIC):
                hits.append((pg * PAGE, head.split(b'\0')[0].decode(errors='replace')))
        return hits

    t0 = time.time()
    with ThreadPoolExecutor(threads) as pool:
        hits = sorted(h for part_hits in pool.map(part, range(threads)) for h in part_hits)
    print(f'searched {pages} page starts in {time.time() - t0:.0f} s', flush=True)
    for addr, text in hits:
        print(f'FOUND 0x{addr:011x} ({addr >> 20} MiB)  {text}', flush=True)


def reread(dri, bdf, addr, n):
    fd = os.open(f'/sys/kernel/debug/dri/{dri}/amdgpu_vram', os.O_RDONLY)
    cnt = f'/sys/bus/pci/devices/{bdf}/ras/umc_err_count'
    ce = lambda: int(open(cnt).read().split('ce:')[1].split()[0])
    before = ce()
    for _ in range(n):
        os.pread(fd, PAGE, addr)
    print(f'CPU re-read 0x{addr:011x} x{n}: CE +{ce() - before}', flush=True)


if sys.argv[1] == 'find':
    find(sys.argv[2], int(sys.argv[3]))
else:
    reread(sys.argv[2], sys.argv[3], int(sys.argv[4], 0), int(sys.argv[5]) if len(sys.argv) > 5 else 100)

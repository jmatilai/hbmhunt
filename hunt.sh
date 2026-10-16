#!/bin/bash
# Usage: ./hunt.sh <pci bdf> [quick|extended]
#
# Description:
#   Executes hbmhunt on an idle GPU to identify the physical VRAM addresses of 
#   faulty pages. It verifies each faulty page using CPU reads and uses 
#   adjacent pages as controls. The script prints the required retirement 
#   commands but does not execute them.
#
# Output:
#   Logs are stored in: runs/<timestamp>_<bdf>_<mode>/
#
# Execution Time:
#   - quick: ~3 minutes
#   - extended: ~25 minutes
#
# Execution Instructions:
#   1. Run as a standard user; the script invokes sudo for debugfs access.
#   2. For background execution, use: 
#      setsid nohup ./hunt.sh 0000:05:00.0 extended > hunt.log 2>&1 &
#   3. Background runs require passwordless sudo. If sudo requires a password, 
#      run the script inside tmux or screen to handle the prompt.

set -u
BDF=${1:?usage: hunt.sh <pci bdf> [quick|extended]}
MODE=${2:-quick}
HERE=$(cd "$(dirname "$0")" && pwd)
case $MODE in
  quick)    OPTS="-s 15 -r 60 -n 1" ;;
  extended) OPTS="-s 60 -r 120 -n 4" ;;
  *) echo "mode is quick or extended"; exit 2 ;;
esac
DEV=/sys/bus/pci/devices/$BDF
[ -e "$DEV/ras/umc_err_count" ] || { echo "$BDF: no amdgpu RAS counters"; exit 1; }
USED_MIB=$(( $(cat $DEV/mem_info_vram_used) >> 20 ))
[ "$USED_MIB" -lt 256 ] || { echo "$BDF has $USED_MIB MiB VRAM in use: stop whatever runs on it first"; exit 1; }
sudo true || { echo "hunt.sh needs sudo for debugfs: run it where sudo can ask for a password (e.g. tmux), or set up passwordless sudo"; exit 1; }
# debugfs dir of the card: dri/<bdf> on kernels >= 6.7, else the minor number (dri/0, dri/1, ...)
if sudo test -e /sys/kernel/debug/dri/$BDF/ras/ras_ctrl; then
  DRI=$BDF
else
  DRI=$(sudo sh -c "grep -l 'dev=$BDF' /sys/kernel/debug/dri/*/name" 2>/dev/null | head -1 | xargs -r dirname | xargs -r basename)
fi
[ -n "$DRI" ] || { echo "no debugfs dri entry for $BDF"; exit 1; }
END_MIB=$(( ($(cat $DEV/mem_info_vram_total) >> 20) - 16 ))   # top 16 MiB: PSP TMR and page tables

if [ ! -x "$HERE/hbmhunt" ] || [ "$HERE/hbmhunt.hip" -nt "$HERE/hbmhunt" ]; then
  ${HIPCC:-hipcc} --offload-arch=gfx906 -O2 -std=c++17 -o "$HERE/hbmhunt" "$HERE/hbmhunt.hip" || exit 1
fi
OUT=$HERE/runs/$(date +%Y%m%d-%H%M%S)_${BDF//:/-}_$MODE
mkdir -p "$OUT"
echo "$BDF dri/$DRI mode $MODE ($OPTS), output $OUT"
{
  echo "start $(date -Is) $(uname -r)"
  echo "unique_id $(cat $DEV/unique_id) vbios $(cat $DEV/vbios_version)"
  echo "umc $(tr '\n' ' ' < $DEV/ras/umc_err_count)"
  echo "retired pages: $(tr '\n' ' ' < $DEV/ras/gpu_vram_bad_pages)"
} > "$OUT/info.txt"

"$HERE/hbmhunt" $OPTS "$BDF" "$OUT" > "$OUT/hbmhunt.txt" 2>&1 &
PID=$!
until grep -q "MARKERS READY" "$OUT/hbmhunt.txt" || ! kill -0 $PID 2>/dev/null; do sleep 5; done

if grep -q "MARKERS READY" "$OUT/hbmhunt.txt"; then
  sudo python3 "$HERE/vramfind.py" find "$DRI" "$END_MIB" > "$OUT/vramfind.txt"
  touch "$OUT/hunt_go"
  until grep -q "PAGES RESTORED" "$OUT/hbmhunt.txt"; do sleep 1; done
  for a in $(awk '/^FOUND/{print $2}' "$OUT/vramfind.txt"); do
    sudo python3 "$HERE/vramfind.py" reread "$DRI" "$BDF" "$a" 100
    sudo python3 "$HERE/vramfind.py" reread "$DRI" "$BDF" $(printf '0x%x' $((a + 4096))) 100
  done > "$OUT/cpu_reread.txt"
  touch "$OUT/hunt_done"
fi
wait $PID
echo "hbmhunt exit $?, end $(date -Is), umc $(tr '\n' ' ' < $DEV/ras/umc_err_count)" >> "$OUT/info.txt"

cat "$OUT/info.txt"
sed -n '/^SUMMARY/,$p' "$OUT/hbmhunt.txt" | grep -vE "^waiting|^marker"
[ -s "$OUT/vramfind.txt" ] && cat "$OUT/vramfind.txt" "$OUT/cpu_reread.txt"
for a in $(awk '/^FOUND/{print $2}' "$OUT/vramfind.txt" 2>/dev/null); do
  echo "retire: echo \"retire_page $a\" | sudo tee /sys/kernel/debug/dri/$DRI/ras/ras_ctrl"
done

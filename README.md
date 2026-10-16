# hbmhunt: Utility to Identify and Retire Faulty HBM Pages (MI50 / Vega20)

`hbmhunt` identifies faulty 4 KiB pages in the High Bandwidth Memory (HBM) of AMD Vega20 (gfx906) GPUs that support HBM Error Correction Code (ECC), such as the Radeon Instinct MI50. The tool determines the physical VRAM addresses of these faulty pages so the user can instruct the `amdgpu` driver to retire them.

GPUs with HBM ECC track corrected bit errors in `ras/umc_err_count`. A continuously increasing error count under load indicates faulty memory. `hbmhunt` isolates the specific pages causing these errors and identifies their physical VRAM addresses. Once identified, the driver can retire these pages. The driver reserves the faulty pages and records them in the card's EEPROM to ensure they are not used again, even after a system reboot or if the card is moved to another machine.

The script `hunt.sh` identifies and verifies faulty pages. It does not retire pages automatically; instead, it provides the necessary commands for the user to execute.

This tool was tested on one MI50 32 GB GPU (Ubuntu kernel 6.8, ROCm 6.4.3). During testing, the tool identified, retired, and verified one faulty page. Refer to the [Example](#example-one-mi50-32-gb) section for details.


## File Overview

| File | Description |
|---|---|
| [hunt.sh](hunt.sh) | The main execution script. Use this to run the process for a specific GPU. |
| [hbmhunt.hip](hbmhunt.hip) | The HIP-based GPU pattern test. `hunt.sh` builds this automatically during the first run. |
| [vramfind.py](vramfind.py) | A Python script that locates a page's physical address using markers or CPU-based page reads. |
| `runs/` | A directory created by `hunt.sh` containing the output logs for each execution. |

## Operational Workflow

1. **Memory Allocation:** The tool allocates approximately 99.5% of available VRAM. It allocates memory in 256 MiB blocks, followed by smaller blocks, and fills them with specific data patterns.
2. **Iterative Testing:** The tool repeatedly reads the memory (achieving 530–730 GB/s on an MI50), compares the data against the expected pattern, and monitors the ECC error counters.
3. **Test Patterns:** The tool employs several patterns to trigger faults:
    - Zeros and ones: Test every bit for both 0 and 1 states.
    - Pseudo-random patterns and their inverses: Ensure every bit holds both values relative to adjacent bits.
4. **Block Identification:** After the iterative test, the tool reads each block 10 times individually to identify which blocks trigger the error counter.
5. **Recursive Isolation:** The tool recursively splits failing blocks into 16 smaller parts until it isolates the specific 4 KiB page. Each part is read 20 times. Between reads, the tool reads 16 MiB from a known "clean" block to bypass the 4 MiB L2 cache and force reads from the HBM.
6. **Physical Address Mapping:** `hbmhunt` writes a marker (`HBMHUNT:`) into the faulty page. The `vramfind.py find` command then scans the first 64 bytes of every page via `/sys/kernel/debug/dri/<card>/amdgpu_vram` to locate the marker and determine the physical address.
7. **Verification:** To confirm the result, `hbmhunt` writes the failing pattern back to the page. The `vramfind.py reread` command reads the page 100 times from the CPU and compares it to a neighboring control page to verify the address.

**Safety Feature:** The tool automatically terminates if the junction or memory temperature exceeds 85 °C.

**Note:** The tool does not use a simple read scan of free VRAM. The driver fills freed VRAM with `0xbe` bytes, which may not trigger a fault. In testing, CPU scans of free VRAM failed to find errors that `hbmhunt` successfully identified.

## Requirements

- **Operating System:** Linux with the `amdgpu` driver.
- **Hardware:** A Vega20 card with ECC and RAS enabled. The file `/sys/bus/pci/devices/<pci address>/ras/umc_err_count` must exist.
- **Compiler:** `hipcc` capable of building for gfx906. The program requires only the HIP compiler and runtime. `hunt.sh` searches for `hipcc` in the system `PATH`. Alternatively, you can specify the path using the `HIPCC` environment variable (e.g., `HIPCC=/opt/rocm/bin/hipcc ./hunt.sh ...`).
- **Python:** `python3` (standard library only).
- **Permissions:** The user must be a member of the `render` and `video` groups to run HIP programs: `sudo usermod -aG render,video $USER`.
- **Privileges:** `sudo` access is required for specific tasks.
- **Filesystem:** `debugfs` must be mounted at `/sys/kernel/debug` (this is the default on most distributions).
- **GPU State:** The GPU must be idle. `hunt.sh` will not execute if more than 256 MiB of VRAM is currently in use.

### Root Privilege Requirements

`hunt.sh` is executed as a normal user but calls `sudo` for the following operations:
- Identifying the card's `debugfs` directory under `/sys/kernel/debug/dri/`.
- Executing `vramfind.py` to read VRAM via the `amdgpu_vram` debugfs file.

The build process and the GPU tests run with user privileges. Retiring pages requires root privileges and must be performed manually by the user.

**Execution Methods:**
- **Passwordless sudo:** Use `setsid nohup` to run the tool in the background.
- **Standard sudo:** Run the tool in the foreground using `tmux` or `screen` so you can enter the sudo password when prompted. The tool prompts for the password at the start and again when it identifies faulty pages.

`hunt.sh` verifies `sudo` access before starting the test and will stop if access is denied.

## Running the Tool

### 1. Verify Hardware Configuration

Find the PCI address (Bus Device Function - BDF) of the GPU:

```sh
# List all cards with amdgpu RAS counters
ls -d /sys/bus/pci/devices/*/ras | cut -d/ -f6

# List all AMD GPUs; identify the "Vega 20" card
lspci -D -d 1002: | grep -iE 'vga|display|3d'
```

Use the full PCI address, including the domain (e.g., `0000:05:00.0`). If multiple cards are present, use `amd-smi list` or `cat /sys/bus/pci/devices/<address>/unique_id` to distinguish them.

Replace `$BDF` with your card's address in the following commands:

```sh
BDF=0000:05:00.0
DEV=/sys/bus/pci/devices/$BDF

# Check for corrected (ce) and uncorrectable (ue) errors
cat $DEV/ras/umc_err_count

# Check for pages already retired
cat $DEV/ras/gpu_vram_bad_pages

# Check current VRAM usage
echo $(( $(cat $DEV/mem_info_vram_used) >> 20 )) MiB VRAM in use

# Verify ECC and RAS status
sudo dmesg | grep -E 'amdgpu.*(ECC|RAS)'
```

Ensure the output includes "MEM ECC is active" and "ras initialized successfully."

For kernels 6.7 and newer, the `debugfs` directory is located at `/sys/kernel/debug/dri/<pci address>`. Older kernels use DRM minor numbers (e.g., `/sys/kernel/debug/dri/0`). `hunt.sh` identifies the correct directory automatically. To manually map minor numbers to cards, use:

```sh
sudo sh -c 'grep -H dev= /sys/kernel/debug/dri/*/name'
```

### 2. Preparation

- Terminate all processes using the GPU. Use `amd-smi process` to identify active processes.
- **Optional:** Limit the power consumption to prevent overheating, as MI50 GPUs are often passively cooled. The test aborts above 85 °C.
  To set a 100 W limit:
  ```sh
  echo 100000000 | sudo tee $DEV/hwmon/hwmon*/power1_cap
  ```
  Or using `amd-smi`: `sudo amd-smi set -g $BDF --power-cap 100`.

### 3. Execution

```sh
git clone https://github.com/jmatilai/hbmhunt
cd hbmhunt

# Quick mode: 6 patterns, approximately 3-4 minutes
./hunt.sh $BDF quick

# Extended mode: 12 patterns, approximately 25 minutes
./hunt.sh $BDF extended
```

**Background Execution (Passwordless sudo):**
```sh
setsid nohup ./hunt.sh $BDF extended > hunt.log 2>&1 &
tail -f hunt.log
```

**Interactive Execution (using tmux):**
```sh
tmux new -s hunt
BDF=0000:05:00.0
./hunt.sh $BDF extended 2>&1 | tee hunt.log
```

Each execution creates a directory in `runs/<timestamp>_<bdf>_<mode>/` containing:
- `info.txt`: Hardware ID, VBIOS version, kernel version, and error counters.
- `hbmhunt.txt`: The complete GPU test log.
- (If faults are found) `hunt_markers.txt`, `vramfind.txt` (containing the `FOUND` lines with physical addresses), and `cpu_reread.txt`.

At completion, `hunt.sh` prints a summary, the identified addresses, and the specific retirement commands.

## Interpreting Results

### Summary Metrics
- **CE per pass:** A value of 1.00 indicates a single bad bit. A value of 2.00 indicates two bad bits. Fractional values indicate intermittent errors.
- **Pattern/Inverse both 0.00:** This suggests a neighbor-dependent fault, where the bit only fails when adjacent bits are in a specific state.
- **Hit blocks:** Blocks identified for the isolation phase. If no part raises the error count individually, the error is too intermittent to isolate a specific page.
- **Mismatches:** Data that was read incorrectly. This should be 0, as ECC corrects single-bit errors. Non-zero values or an increasing `ue:` (uncorrectable error) count indicate critical hardware failure.
- **CPU re-read:** The faulty page must trigger a Corrected Error (CE) when read from the CPU, while the neighboring page must not. This confirms the address is accurate.
- **NO BAD PAGES:** No blocks triggered the error counter. This is the expected result after successful retirement.

### Limitations
The tool cannot detect:
- The 0.5% of VRAM reserved by the driver.
- Neighbor-dependent faults not triggered by the 12 provided patterns.
- Errors that occur less frequently than once per 0.7 PB of reads.
- Faults that only appear at temperatures higher than those reached during the test.

## How to Retire a Page

Retiring a page creates a permanent record in the card's EEPROM. This action cannot be undone for individual pages.

```sh
D=/sys/kernel/debug/dri/$BDF        # For kernels 6.7+; use the minor number for older kernels
ADDR=0x2a4065000                    # Replace with the address from vramfind.txt

# Review current bad pages and EEPROM records
cat /sys/bus/pci/devices/$BDF/ras/gpu_vram_bad_pages
sudo cat $D/ras/ras_eeprom_table

# Retire the page
echo "retire_page $ADDR" | sudo tee $D/ras/ras_ctrl

# Verify retirement
sudo dmesg | tail -4
sudo cat $D/ras/ras_eeprom_table
sudo cat $D/amdgpu_vram_mm | sed -n '/reserved:/,$p'
```

The driver reserves the page immediately and writes the record to the RAS EEPROM.

**Retirement Notes:**
- **Warning:** The kernel may warn that this command "will corrupt RAS EEPROM." This occurs because the record is being added manually rather than being triggered by an uncorrectable error.
- **Status `F`:** In some kernels (e.g., Ubuntu 6.8.0-147), `gpu_vram_bad_pages` may mark the page as `F` instead of `R` (Reserved). A line such as `0x002a4065 : 0x00001000 : F` refers to the page at `0x2a4065000`. Verify the reservation in `amdgpu_vram_mm`.
- **Failure Threshold:** If the number of retired pages exceeds the threshold in `$D/ras/bad_page_cnt_threshold` (e.g., 327 on a 32 GB card), the driver will mark the card as failed.
- **Permission Errors:** If you receive "Operation not permitted" while running as root, check if kernel lockdown is enabled via `cat /sys/kernel/security/lockdown`. Secure Boot often enables this, which blocks `debugfs` writes.

## How to Unretire Pages

The kernel cannot remove individual retirement records. To remove them, you must clear the entire table:

```sh
sudo cat $D/ras/ras_eeprom_table        # Backup existing records
echo 1 | sudo tee $D/ras/ras_eeprom_reset
sudo reboot                             # Reboot is required to apply changes
```
*Note: This action also removes records created by the driver following actual uncorrectable errors.*

## Step-by-Step Validation Process

1. **Initial Setup:** Install the card and boot the system. Verify the card appears in `lspci`, ensure `dmesg` shows no `amdgpu` errors, and check the PCIe link and `ras/*` counters.
2. **Baseline:** Record the current `umc_err_count` and any existing EEPROM entries.
3. **Preparation:** Stop all GPU processes and apply power limits.
4. **Detection:** Run `./hunt.sh $BDF extended`.
5. **Analysis:**
    - **Stable Faults:** Pages with exactly 1.00 CE per pass, confirmed by CPU re-read, with no neighbor errors $\rightarrow$ **Retire these pages.**
    - **Degrading Memory:** Intermittent errors, fractional CE counts, or errors spread across many blocks $\rightarrow$ **The hardware is unreliable; do not trust the card.**
6. **Verification:** Retire the identified pages and reboot. Run `./hunt.sh $BDF quick` to ensure it reports `NO BAD PAGES`. Finally, run a heavy workload (e.g., LLM inference) and verify that `umc_err_count` does not increase.
7. **Documentation:** Maintain the `runs/` directories and the retirement logs as a hardware record.

## Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `no amdgpu RAS counters` | Incorrect address, card lacks HBM ECC, or RAS is disabled. | Use the `0000:` domain for the address. Check `sudo dmesg \| grep -E 'amdgpu.*(ECC\|RAS)'` and the `amdgpu.ras_enable` kernel parameter. |
| `... MiB VRAM in use` | The GPU is not idle. | Identify and terminate active processes using `amd-smi process`. |
| `hunt.sh needs sudo for debugfs` | Sudo cannot prompt for a password (e.g., during background execution). | Use `tmux` or configure passwordless sudo. |
| `no debugfs dri entry` | `debugfs` is not mounted. | Run `sudo mount -t debugfs none /sys/kernel/debug`. |
| `no HIP device at ...` | User permissions or environment variables are restricting access. | Add the user to the `render` and `video` groups. Check `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`. |
| Build failure / `hipcc: command not found` | `hipcc` is not in the path or is incompatible. | Set the `HIPCC` environment variable to the correct `hipcc` binary supporting gfx906. |
| `ABORT: junction ... memory ...` | GPU temperature exceeded 85 °C. | Lower the power limit or improve cooling. |
| Writing `ras_ctrl` fails as root | Kernel lockdown is active. | See the retirement notes regarding Secure Boot and lockdown. |

## Example: One MI50 32 GB

**Environment:** Ubuntu kernel 6.8.0-147, ROCm 6.4.3, VBIOS 113-D1640200-043.

| Step | Result |
|---|---|
| Quick Test | Identified one page: `0x2a4065000`. CPU re-read: +306 CE in 100 reads. |
| Extended Test | Only the same page identified. 0 mismatches. All patterns showed 1.00 or 0.00 CE/pass. |
| Retire | Created 1 EEPROM record; page successfully reserved. |
| Reboot | Record loaded; page remained reserved. |
| Final Quick Test | **NO BAD PAGES** |
| Stress Test | gemma3-27B Q8 $\times$ 2, 120 W: 0 additional CE. |

**Fault Analysis:**
The fault is neighbor-dependent: one bit in the page cannot hold a 0, but only when a second bit is also 0. It failed with zeros, `0x55`, and 3 of the 8 random patterns, but passed with ones and `0xAA`. This indicates a fault at an odd bit position. No temperature dependence was observed between 39 °C and 72 °C.

## License

Apache 2.0, see [LICENSE](LICENSE).
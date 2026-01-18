#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Settings
# -----------------------------
VM_NAME="WinGuest3"
BASELINE_SNAP="snapshot1"

START_ID=1
END_ID=56


GUEST_USER="test"
GUEST_PASS="test"

# Timing (in seconds)
VM_BOOT_WAIT=300
POST_LAUNCH_WAIT=5
DUMP_COUNT=100
DUMP_INTERVAL=10

# Tools
VBOX="VBoxManage"
PYTHON=$(which python3)   # 実際に存在する python3
SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ELF2IMG="${SCRIPT_ROOT}/elf2img.py"

# Working directories
DUMP_ROOT="${SCRIPT_ROOT}/dump"
IMG_ROOT="${SCRIPT_ROOT}/images256"
LOG_DIR="${SCRIPT_ROOT}/logs"

mkdir -p "${DUMP_ROOT}" "${IMG_ROOT}" "${LOG_DIR}"

# -----------------------------
# Logging
# -----------------------------
log() {
    local msg="$1"
    local logfile="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" | tee -a "$logfile"
}

# -----------------------------
# Main
# -----------------------------
for n in $(seq "${START_ID}" "${END_ID}"); do
    ID=$(printf "id%02d" "$n")
    LOGFILE="${LOG_DIR}/${ID}.log"
    TMP_DUMP_DIR="${DUMP_ROOT}/${ID}"
    IMG_ID_DIR="${IMG_ROOT}/${ID}"

    echo "===================================================" > "$LOGFILE"
    log "[*] Start processing ${ID}" "$LOGFILE"
    echo "===================================================" >> "$LOGFILE"

    # Power off & snapshot restore
    log "[*] Powering off VM..." "$LOGFILE"
    ${VBOX} controlvm "${VM_NAME}" poweroff 2>/dev/null || true
    sleep 5

    log "[*] Restoring snapshot '${BASELINE_SNAP}'..." "$LOGFILE"
    ${VBOX} snapshot "${VM_NAME}" restore "${BASELINE_SNAP}"

    # Boot VM
    log "[*] Starting VM..." "$LOGFILE"
    ${VBOX} startvm "${VM_NAME}" --type headless

    log "[*] Waiting ${VM_BOOT_WAIT}s for OS to boot..." "$LOGFILE"
    sleep "${VM_BOOT_WAIT}"

    # Start target program inside guest
    log "[*] Launching ${ID} in guest..." "$LOGFILE"
    GUEST_CMD="if exist C:\\app\\${ID}.exe (start C:\\app\\${ID}.exe) else if exist C:\\app\\${ID}.bat (start C:\\app\\${ID}.bat) else exit /b 1"

    ${VBOX} guestcontrol "${VM_NAME}" run \
        --exe "C:\\Windows\\System32\\cmd.exe" \
        --username "${GUEST_USER}" \
        --password "${GUEST_PASS}" \
        --no-wait-stdout --no-wait-stderr -- \
        /c "${GUEST_CMD}" || true

    sleep "${POST_LAUNCH_WAIT}"

    # Memory dumps
    mkdir -p "${TMP_DUMP_DIR}"
    log "[*] Collecting memory dumps..." "$LOGFILE"

    for i in $(seq 1 "${DUMP_COUNT}"); do
        TS=$(date +"%Y%m%d_%H%M%S")
        ELF="${TMP_DUMP_DIR}/${VM_NAME}_${ID}_${TS}_${i}.elf"
        ${VBOX} debugvm "${VM_NAME}" dumpvmcore --filename "${ELF}"
        [[ $i -lt ${DUMP_COUNT} ]] && sleep "${DUMP_INTERVAL}"
    done

    # ELF -> images
    log "[*] Converting ELF to images..." "$LOGFILE"
    "${PYTHON}" "${ELF2IMG}" \
        --input_root "${DUMP_ROOT}" \
        --output_root "${IMG_ROOT}" \
        --id "${ID}" | tee -a "$LOGFILE"

    # Cleanup
    log "[*] Removing ELF files..." "$LOGFILE"
    rm -rf "${TMP_DUMP_DIR}"

    log "[*] Finished ${ID}" "$LOGFILE"
done

echo "==================================================="
echo "[*] All samples processed at $(date)"
echo "==================================================="

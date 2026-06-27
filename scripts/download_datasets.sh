#!/usr/bin/env bash
#
# download_datasets.sh — fetch real-world GPU-cluster traces for EvoSentinel DC.
#
# Grounds the simulator/classifier in production DCGM-style telemetry instead of
# purely synthetic data. Idempotent: re-running skips datasets already present and
# valid. Works on macOS (bash 3.2) and Linux.
#
# Usage (run from the hack/ project root, or anywhere — paths self-resolve):
#   scripts/download_datasets.sh                 # default set: pai + acme
#   scripts/download_datasets.sh all             # pai + acme + philly + extra
#   scripts/download_datasets.sh pai acme        # named subset
#   scripts/download_datasets.sh --list          # show datasets and exit
#   DATA_DIR=/some/path scripts/download_datasets.sh pai
#
# Data lands in hack/data/ (gitignored) as: data/{pai,acme,philly,clusterdata}/
#
# Datasets:
#   pai     Alibaba PAI 2020 GPU trace (~3.9 GB extracted). 7 CSV tables incl.
#           per-worker GPU/CPU/mem utilization sensors. Fractional GPU sharing.
#   acme    Acme LLM-datacenter trace (NSDI'24). Job traces w/ failure states;
#           raw 80 GB utilization stays on HuggingFace (not pulled here).
#   philly  Microsoft Philly DNN trace (2017). BEST-EFFORT ONLY — the 1.1 GB data
#           blob is behind GitHub LFS whose budget is exhausted (HTTP 403). The
#           repo (README + analysis notebook) is fetched; the data may fail.
#   extra   Newer Alibaba traces (v2023 K8s GPU-sharing, v2025 DLRM,
#           v2026-GenAI serving) — committed in-repo, pulled via shallow clone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"   # project root (scripts/ lives one level down)
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
PAI_OSS="https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces"
ACME_REPO="https://github.com/InternLM/AcmeTrace.git"
PHILLY_REPO="https://github.com/msr-fiddle/philly-traces.git"
CLUSTERDATA_REPO="https://github.com/alibaba/clusterdata.git"

# ---- portability helpers ----------------------------------------------------

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

sha256() {  # print sha256 of $1 (mac: shasum, linux: sha256sum)
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

require() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }

# Official PAI v2020 SHA-256 checksums (from alibaba/clusterdata).
pai_sum() {
  case "$1" in
    pai_job_table.tar.gz)      echo 5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0 ;;
    pai_task_table.tar.gz)     echo cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5 ;;
    pai_instance_table.tar.gz) echo 1bf1e423a7ce3f8d086699801c362fd56a7182abdb234139e5ebbed97995ca06 ;;
    pai_sensor_table.tar.gz)   echo 9a0b82e8bdf3949281e4ba1423d9b4b34847e52799eecb138966de46da69c7a0 ;;
    pai_group_tag_table.tar.gz)echo 722fef30b7fb7aa50dabd79155614b5423a9d65cf45a9b26c590d57725423a14 ;;
    pai_machine_spec.tar.gz)   echo cc0d38a4045af1b1af8179de8b1b54b1ddd995e6160d6d061a6b1000f1276c2d ;;
    pai_machine_metric.tar.gz) echo 53ad917193d3b1dd0f3055e723148b1f36c2f81789b014ea2930a7875892eef5 ;;
    *) echo "" ;;
  esac
}

# PAI CSVs ship headerless; emit the column names as <table>.header sidecars.
pai_header() {
  case "$1" in
    pai_job_table)      echo "job_name,inst_id,user,status,start_time,end_time" ;;
    pai_task_table)     echo "job_name,task_name,inst_num,status,start_time,end_time,plan_cpu,plan_mem,plan_gpu,gpu_type" ;;
    pai_instance_table) echo "job_name,task_name,inst_name,worker_name,inst_id,status,start_time,end_time,machine" ;;
    pai_sensor_table)   echo "job_name,task_name,worker_name,inst_id,machine,gpu_name,cpu_usage,gpu_wrk_util,avg_mem,max_mem,avg_gpu_wrk_mem,max_gpu_wrk_mem,read,write,read_count,write_count" ;;
    pai_group_tag_table)echo "inst_id,user,gpu_type_spec,group,workload" ;;
    pai_machine_spec)   echo "machine,gpu_type,cap_cpu,cap_mem,cap_gpu" ;;
    pai_machine_metric) echo "worker_name,machine,start_time,end_time,machine_cpu_iowait,machine_cpu_kernel,machine_cpu_usr,machine_gpu,machine_load_1,machine_net_receive,machine_num_worker,machine_cpu" ;;
    *) echo "" ;;
  esac
}

# ---- datasets ---------------------------------------------------------------

fetch_pai() {
  local dest="$DATA_DIR/pai" tables tbl tgz want got
  tables="pai_job_table pai_task_table pai_instance_table pai_sensor_table pai_group_tag_table pai_machine_spec pai_machine_metric"
  mkdir -p "$dest"
  log "PAI 2020 -> $dest"
  for tbl in $tables; do
    tgz="$tbl.tar.gz"
    want="$(pai_sum "$tgz")"
    if [ -f "$dest/$tbl.csv" ]; then
      log "  $tbl.csv present, skipping"
      continue
    fi
    if [ ! -f "$dest/$tgz" ]; then
      log "  downloading $tgz"
      curl -fSL --retry 3 -o "$dest/$tgz" "$PAI_OSS/$tgz" || die "download failed: $tgz"
    fi
    got="$(sha256 "$dest/$tgz")"
    [ "$got" = "$want" ] || die "checksum mismatch for $tgz (got $got, want $want)"
    log "  checksum OK, extracting $tgz"
    tar -xzf "$dest/$tgz" -C "$dest"
    printf '%s\n' "$(pai_header "$tbl")" > "$dest/$tbl.header"
    rm -f "$dest/$tgz"   # drop the tarball; CSV is the artifact
  done
  log "PAI done. Headerless CSVs paired with <table>.header sidecars."
}

fetch_acme() {
  local dest="$DATA_DIR/acme"
  if [ -d "$dest/.git" ]; then
    log "Acme present, pulling latest"
    git -C "$dest" pull --ff-only || warn "acme pull failed (continuing)"
    return
  fi
  log "Acme -> $dest (shallow clone, job traces only)"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$ACME_REPO" "$dest"
  log "Acme done. Job traces: data/job_trace/trace_{seren,kalos}.csv"
  log "  (raw 80 GB utilization is on HuggingFace: Qinghao/AcmeTrace)"
}

fetch_philly() {
  local dest="$DATA_DIR/philly"
  log "Philly -> $dest (BEST-EFFORT; data blob behind exhausted LFS budget)"
  if [ ! -d "$dest/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "$PHILLY_REPO" "$dest" || { warn "philly clone failed"; return; }
  fi
  log "  attempting LFS data pull (expected to 403 until Microsoft refills budget)"
  if git -C "$dest" lfs pull 2>/dev/null; then
    log "Philly data downloaded — LFS budget restored."
  else
    warn "Philly trace-data.tar.gz unavailable (LFS 403). Repo README + notebook fetched."
    warn "Use PAI 2020 as the closest analog. Retry later or watch the upstream repo."
  fi
}

fetch_extra() {
  local dest="$DATA_DIR/clusterdata"
  if [ -d "$dest/.git" ]; then
    log "clusterdata present, pulling latest"
    git -C "$dest" pull --ff-only || warn "clusterdata pull failed (continuing)"
    return
  fi
  log "Extra Alibaba traces -> $dest (shallow clone of full clusterdata repo)"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$CLUSTERDATA_REPO" "$dest"
  log "Extra done. v2023 (K8s GPU-share), v2025 (DLRM), v2026-GenAI (serving)."
  log "  Note: cluster-trace-gpu-v2020/data/ here only has the download script;"
  log "  the full v2020 data lives under the 'pai' dataset of this script."
}

# clusterdata's bundled v2020 analysis notebook + simulator read CSVs from
# cluster-trace-gpu-v2020/data/. The full data lives in data/pai/, so symlink it
# into that path. Relative links survive moving DATA_DIR. No-op unless both the
# 'pai' CSVs and the 'extra' clusterdata clone are present.
link_pai_into_clusterdata() {
  local pai="$DATA_DIR/pai"
  local v2020="$DATA_DIR/clusterdata/cluster-trace-gpu-v2020/data"
  [ -d "$pai" ] && [ -d "$v2020" ] || return 0
  ls "$pai"/pai_*.csv >/dev/null 2>&1 || return 0
  log "linking PAI CSVs into clusterdata v2020 tooling path"
  local f base
  for f in "$pai"/pai_*.csv; do
    base="$(basename "$f")"
    ln -sf "../../../pai/$base" "$v2020/$base"   # relative to $v2020
  done
}

# ---- driver -----------------------------------------------------------------

usage() {
  # Print the leading comment block (everything after the shebang up to the
  # first non-comment line), with the leading "# " stripped.
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
  exit "${1:-0}"
}

require curl
require git

targets=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage 0 ;;
    --list) printf 'pai acme philly extra (default: pai acme; "all" = everything)\n'; exit 0 ;;
    all) targets=(pai acme philly extra) ;;
    pai|acme|philly|extra) targets+=("$arg") ;;
    *) die "unknown argument: $arg (try --help)" ;;
  esac
done
[ "${#targets[@]}" -eq 0 ] && targets=(pai acme)

mkdir -p "$DATA_DIR"
log "data dir: $DATA_DIR"
log "targets: ${targets[*]}"

for t in "${targets[@]}"; do
  case "$t" in
    pai)    fetch_pai ;;
    acme)   fetch_acme ;;
    philly) fetch_philly ;;
    extra)  fetch_extra ;;
  esac
done

link_pai_into_clusterdata

log "all requested datasets processed."

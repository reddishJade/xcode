#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/../.." && pwd)"

repeat_count="${1:-10}"
warmup_count="${2:-1}"
default_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="${3:-${repository_root}/benchmark-results/tool_scheduling/worker-sweep-${default_stamp}}"

if [[ ! "${repeat_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "repeat must be a positive integer: ${repeat_count}" >&2
  exit 2
fi

if [[ ! "${warmup_count}" =~ ^[0-9]+$ ]]; then
  echo "warmup must be a non-negative integer: ${warmup_count}" >&2
  exit 2
fi

if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite existing output root: ${output_root}" >&2
  exit 2
fi

mkdir -p "${output_root}"
cd "${repository_root}"

workers=(1 2 4 8 16)
for worker_count in "${workers[@]}"; do
  worker_output="${output_root}/workers-${worker_count}"
  worker_log="${output_root}/workers-${worker_count}.log"
  echo "running workers=${worker_count}; output=${worker_output}"
  uv run python -m benchmarks.runners.run_tool_scheduling \
    benchmarks/tasks/parallel_reads \
    --repeat "${repeat_count}" \
    --warmup "${warmup_count}" \
    --workers "${worker_count}" \
    --output-dir "${worker_output}" \
    2>&1 | tee "${worker_log}"
done

echo "worker sweep completed: ${output_root}"
for worker_count in "${workers[@]}"; do
  echo "workers=${worker_count}: ${output_root}/workers-${worker_count}/report.md"
done

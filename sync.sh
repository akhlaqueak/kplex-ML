#!/bin/bash

BASE_DIR="$(pwd)"
STAMP_FILE=".last_sync_time"
GIT_STAMP=".last_git_commit"
SSH_CONTROL_DIR="${HOME}/.ssh/cm"
SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/$(basename "$BASE_DIR")-%r@%h:%p"
SSH_OPTS="-o ControlMaster=auto -o ControlPersist=1h -o ControlPath=${SSH_CONTROL_PATH}"
REMOTE_DIR="/home/akhlaque.ak@gmail.com/kplex-ML/"

mkdir -p "$SSH_CONTROL_DIR"

ensure_ssh_master() {
  local ssh_start ssh_end
  ssh_start=$(date +%s)
  ssh $SSH_OPTS -O check "$ak" >/dev/null 2>&1 || \
    ssh -fN $SSH_OPTS "$ak"
  ssh_end=$(date +%s)
  echo "SSH ensure time: $((ssh_end - ssh_start)) seconds"
}

# initialize sync timestam
if [ ! -f "$STAMP_FILE" ]; then
  echo "First run: syncing everything..."
  date > "$STAMP_FILE"
  ensure_ssh_master
  rsync -az -e "ssh $SSH_OPTS" --exclude='.git/' . "$ak:$REMOTE_DIR"
  exit 0
fi

start=$(date +%s)
echo "Syncing changed files.."

# find changed files
changed_files=$(find . -type f -newer "$STAMP_FILE" ! -path "./.git/*")

count=0
tmpfile=$(mktemp)

for file in $changed_files; do
  [[ "$file" == "$STAMP_FILE" ]] && continue
  printf '%s\n' "${file#./}" >> "$tmpfile"
  count=$((count+1))
done

if [ "$count" -gt 0 ]; then
  ensure_ssh_master
  rsync -az -e "ssh $SSH_OPTS" --files-from="$tmpfile" ./ "$ak:$REMOTE_DIR"
fi

rm -f "$tmpfile"

# update sync timestamp
date > "$STAMP_FILE"

end=$(date +%s)

echo "--------------------------------------------"
echo "Synced $count files"
echo "Time taken: $((end - start)) seconds"
echo "--------------------------------------------"

# =========================
# 🔹 DAILY GIT COMMIT LOGIC
# =========================

# initialize git stamp if missing
if [ ! -f "$GIT_STAMP" ]; then
  date > "$GIT_STAMP"
fi

last_git_time=$(stat -f %m "$GIT_STAMP")
current_time=$(date +%s)

# 86400 seconds = 1 day
if (( current_time - last_git_time > 86400 )); then
  echo "Running daily git commit..."

  git add -A

  if ! git diff --cached --quiet; then
    git commit -am "auto commit $(date '+%Y-%m-%d %H:%M:%S')"
    git push
    echo "Git commit + push done"
  else
    echo "No changes to commit"
  fi

  # update git timestamp
  date > "$GIT_STAMP"
fi

# a430sysid

System Identification for aircraft A430.

## TODO

1. 增加评估脚本，执行某个csv中的动作序列，评估生成的轨迹与原轨迹相似度

## Develop

### Prepare python environment

```bash
cd a430sysid
uv sync
```

### Pre-commit

```bash
# Install pre-commit
pre-commit install

# Run
pre-commit run --all-files  # run all hooks on all files
pre-commit run <HOOK_ID> --all-files # run one hook on all files
pre-commit run --files <PATH_TO_FILE>  # run all hooks on a file
pre-commit run <HOOK_ID> --files <PATH_TO_FILE> # run one hook on a file

# Commit
git add .
git commit -m <MESSAGE>

# Commit without hooks
git commit -m <MESSAGE> --no-verify

# update pre-commit hook
pre-commit autoupdate
```

### Run scripts in offline environment

1. Copy `~/.cache` to offline computer
2. Execute `uv sync` on offline computer

#!/bin/bash
# Fail if the general tool names anything personal: a maintainer's own markers (names, e-mail, machines, private plugins and runs), one extended regular
# expression per line in tools/split_markers.local (not in git: the list itself would publish them). Without that file the check is skipped.
# Checks the core paths: src/, tests/, bench/, docs/ (except docs/personal/), tools/, README.md, pyproject.toml.
cd "$(dirname "$0")/.." || exit 2
M=tools/split_markers.local
if [ ! -s "$M" ]; then echo "check_split: skipped (no $M)"; exit 0; fi
PAT=$(grep -v '^\s*$' "$M" | paste -sd'|')
files=$(git ls-files 'src/*' 'tests/*' 'bench/*' 'docs/*' 'tools/*' README.md pyproject.toml | grep -v '^docs/personal/' | grep -v -E '^src/[a-z_]+_personal/')
# one allowed line: the entry point a private plugin package appends to pyproject.toml on its own branch
hits=$(grep -n -E "$PAT" $files 2>/dev/null | grep -v -E '^pyproject.toml:[0-9]+:[a-z_]+_personal = "[a-z_]+_personal"$')
if [ -n "$hits" ]; then echo "personal markers in the general part:"; echo "$hits" | head -40; exit 1; fi
echo "check_split: clean ($(echo "$files" | wc -l) files)"

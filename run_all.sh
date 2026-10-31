#!/usr/bin/env bash
# run_all.sh — 一键复现
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "[1/5] 提取 L2 普通话音频"
python3 src/prepare_l2.py                    | tee logs/01_l2.log

echo "[2/5] 下载母语者录音（同一批句子）"
python3 src/fetch_cmu.py --from-list data/l2_meta.csv | tee logs/02_cmu.log

echo "[3/5] 组织语料目录"
python3 src/run_pipeline.py --build-corpus   | tee logs/03_corpus.log

echo "[4/5] 音素级强制对齐（需要 conda env: mfa）"
conda run -n mfa mfa align data/corpus \
    models/dict/english_us_arpa.dict models/acoustic data/aligned \
    --clean -j 8                             | tee logs/04_align.log

echo "[5/5] 提取共振峰并分析"
python3 src/run_pipeline.py --extract        | tee logs/05_extract.log
python3 src/analyze.py                       | tee logs/06_analyze.log

echo "全部完成"

"""
Project configuration.
"""

# ======================================================
# Dataset
# ======================================================

DATASET_PATH = "data/processed/qwen_tokenized_dataset"
DATASET_NAME = "Trendyol Cybersecurity"

# ======================================================
# Models
# ======================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "outputs/lora/final"

# ======================================================
# Evaluation
# ======================================================

NUM_EVAL_SAMPLES = 10

# ======================================================
# Generation
# ======================================================

MAX_NEW_TOKENS = 150
DO_SAMPLE = False
TEMPERATURE = 0.0

# ======================================================
# Hardware
# ======================================================

CPU_NAME = "AMD Ryzen 9 5950X"
CPU_CORES = "16 Cores / 32 Threads"
RAM = "64 GB DDR4"
GPU = "CPU Only"
OPERATING_SYSTEM = "Windows 11"

# ======================================================
# Output Directories
# ======================================================

METRICS_DIR = "outputs/metrics"
REPORTS_DIR = "outputs/reports"
GRAPHS_DIR = "outputs/graphs"
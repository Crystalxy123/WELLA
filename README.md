
# WELLA: Workload Estimation with LLMs and Agents

[![GitHub Repo stars](https://img.shields.io/github/stars/your-org/wella?style=social)](#)
[![Last Commit](https://img.shields.io/github/last-commit/your-org/wella)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](#)
[![Model](https://img.shields.io/badge/Model-Qwen--7B-blue)](#)

> **TL;DR**：本仓库基于 **LLaMA-Factory**，使用私有领域数据对 **Qwen-7B** 进行 **SFT（支持 LoRA/QLoRA）**，用于核电/高风险行业场景的**操纵员工作负荷（workload）估计**与**多角色智能体协同（agents）**。提供零代码 Web UI、CLI、一键导出与推理脚本，满足可复现与落地部署需求。

## 目录

* [功能特性](#功能特性)
* [快速开始](#快速开始)

  * [环境与安装](#环境与安装)
  * [数据准备](#数据准备)
  * [一键训练（LoRA/QLoRA）](#一键训练loraqlora)
  * [推理与评测](#推理与评测)
  * [合并与导出](#合并与导出)
* [配置样例（可直接运行）](#配置样例可直接运行)
* [数据格式示例](#数据格式示例)
* [项目结构](#项目结构)
* [常见问题](#常见问题)
* [许可与引用](#许可与引用)

---

## 功能特性

* **面向工作负荷估计的指令微调**：用私有数据对 **Qwen-7B** 做 SFT，使模型学习从**任务上下文/交互记录**→**SART/NASA-TLX 等分数**或等级标签。
* **LLaMA-Factory 全流程能力**：CLI 与 Web UI（Gradio/LlamaBoard）、多卡/分布式、W\&B/SwanLab 记录、vLLM/SGLang 快速推理。
* **高效训练**：LoRA/QLoRA、FlashAttention-2、NEFTune、rsLoRA、GaLore、BAdam 等可选加速与省显存技巧。
* **数据安全**：默认**本地训练**，支持脱网环境；数据目录隔离，可控导出。
* **即插即用 Agents**：提供多角色模板（RO/SRO/Shift Leader 等）调用同一基座，实现**场景/角色条件化**推理。

---

## 快速开始

### 环境与安装

```bash
# 1) 克隆本仓库（或在你的项目里添加本 README）
git clone https://github.com/your-org/wella.git
cd wella

# 2) 安装 LLaMA-Factory（作为子模块或直接安装）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
pip install -e "LLaMA-Factory[torch,metrics]" --no-build-isolation

# 可选：FlashAttention-2 / bitsandbytes / vllm 等按需安装
```

**推荐环境**

* Python ≥ 3.10，CUDA 12.x，PyTorch ≥ 2.1
* 显存参考：Qwen-7B + QLoRA（4bit）单卡 24GB 基本可跑；多卡更优

### 数据准备

将你的**私有数据**整理为 **SFT 指令格式**（见下文示例），并在 `data/dataset_info.json` 中登记（下方提供最小可用示例）。

> 你的数据不会被上传；所有训练均在本地/你控制的机器上进行。

### 一键训练（LoRA/QLoRA）

```bash
# CLI 训练（LoRA/QLoRA 任选其一，对应的 YAML 见下文）
llamafactory-cli train configs/wella_qwen7b_lora_sft.yaml
# 或
llamafactory-cli train configs/wella_qwen7b_qlora_sft.yaml
```

Web UI（LLaMA Board）：

```bash
llamafactory-cli webui
# 浏览器访问 http://localhost:7860
```

### 推理与评测

```bash
# 聊天/推理
llamafactory-cli chat configs/infer_qwen7b_lora.yaml

# 也可启用 OpenAI-style API（便于前端或Agent框架接入）
API_PORT=8000 llamafactory-cli api configs/infer_qwen7b_lora.yaml infer_backend=vllm
```

### 合并与导出

```bash
# 将 LoRA 权重合并为全量权重（便于单文件部署）
llamafactory-cli export configs/merge_qwen7b_lora.yaml
```

---

## 配置样例（可直接运行）

> 放到项目内 `configs/` 目录即可。若你使用 **Qwen-7B** 系列，模板请设为 `template: qwen`。

**1) LoRA SFT（`configs/wella_qwen7b_lora_sft.yaml`）**

```yaml
# 基座模型（可替换为你具体的Qwen-7B路径或HF Hub ID）
model_name_or_path: Qwen/Qwen-7B
template: qwen
finetuning_type: lora

# 数据
dataset: wella_sft
dataset_dir: ./data
max_samples: 0  # 0=用全部
cutoff_len: 4096
packing: true

# 训练
output_dir: ./output/wella_qwen7b_lora
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.5e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.0
bf16: true
logging_steps: 10
save_steps: 200
evaluation_strategy: "no"

# LoRA 超参
lora_target: "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# 加速
flash_attn: fa2
gradient_checkpointing: true
report_to: "wandb"   # 可选：或 "swanlab"
run_name: "wella_qwen7b_lora_sft"
```

**2) QLoRA SFT（`configs/wella_qwen7b_qlora_sft.yaml`）**

```yaml
model_name_or_path: Qwen/Qwen-7B
template: qwen
finetuning_type: lora

# 量化
quantization_bit: 4
bnb_4bit_compute_dtype: bf16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true

dataset: wella_sft
dataset_dir: ./data
cutoff_len: 4096
packing: true

output_dir: ./output/wella_qwen7b_qlora
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
flash_attn: fa2
gradient_checkpointing: true

lora_target: "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05

report_to: "none"
```

**3) 推理（`configs/infer_qwen7b_lora.yaml`）**

```yaml
model_name_or_path: Qwen/Qwen-7B
template: qwen
infer_backend: vllm
adapter_name_or_path: ./output/wella_qwen7b_lora  # 换成你的输出目录
vllm_enforce_eager: true
max_new_tokens: 512
temperature: 0.2
top_p: 0.9
```

**4) 合并 LoRA（`configs/merge_qwen7b_lora.yaml`）**

```yaml
model_name_or_path: Qwen/Qwen-7B
template: qwen
adapter_name_or_path: ./output/wella_qwen7b_lora
export_dir: ./export/wella_qwen7b_merged
export_legacy_format: false
```

---

## 数据格式示例

**登记数据集**（`data/dataset_info.json`）：

```json
{
  "wella_sft": {
    "file": "wella_sft.jsonl",
    "format": "sharegpt",
    "columns": {
      "messages": "messages"
    }
  }
}
```

**SFT 样例**（`data/wella_sft.jsonl` 的单行示例；多行拼接）：

```json
{
  "messages": [
    {"role": "system", "content": "You are WELLA, an assistant for workload estimation in nuclear control rooms."},
    {"role": "user", "content": "<ROLE>RO</ROLE>\n<TASK>Feed-and-Bleed</TASK>\n<LOG>...（可放指令序列、界面操作、时间戳、重要参数）...</LOG>\n<QUESTION>Estimate operator workload on SART scale (10-80) and briefly justify.</QUESTION>"},
    {"role": "assistant", "content": "{\"sart\": 62, \"justification\": \"Frequent parameter search and valve toggling; time pressure moderate; comprehension load high due to multi-panel cross-check.\"}"}
  ]
}
```


---

## 项目结构

```
wella/
├─ configs/                      # 训练/推理/合并的 YAML
├─ data/
│  ├─ dataset_info.json
│  └─ wella_sft.jsonl           # 你的私有数据（不提交到公共仓库）
├─ output/                       # 训练产物（LoRA）
├─ export/                       # 合并后的全量权重
├─ scripts/
│  ├─ prepare_data.py            # 可选：数据清洗/打标/格式化
│  └─ eval_workload.py           # 可选：评测脚本（R²/MAE/分布一致性等）
└─ README.md
```

---

## 常见问题

* **Q：Qwen-7B 用哪个模板？**
  A：`template: qwen`（训练与推理需一致）。

* **Q：显存不够？**
  A：使用 **QLoRA（4bit）**、`gradient_checkpointing: true`、`per_device_train_batch_size` 减小；必要时多卡/梯度累积。

* **Q：如何做“角色/场景条件化”？**
  A：在 **prompt** 中显式加入 `<ROLE>`、`<TASK>`、`<SCENARIO>` 等标签，训练与推理保持一致。

* **Q：如何保证数据安全？**
  A：全流程本地；不上传到云。建议在 `data/` 做访问控制与脱敏；如需容器化，挂载只读卷。

---

## 许可与引用

本仓库代码遵循 **Apache-2.0**；模型权重请遵循相应上游许可（例如 Qwen 模型许可）。

如本项目对你有帮助，欢迎引用：

```bibtex
@article{xiao2025dynamic,
  title={A Dynamic and High-Precision Method for Scenario-Based HRA Synthetic Data Collection in Multi-Agent Collaborative Environments Driven by LLMs},
  author={Xiao, Xingyu and Chen, Peng and Jia, Qianqian and Tong, Jiejuan and Liang, Jingang and Wang, Haitao},
  journal={arXiv preprint arXiv:2502.00022},
  year={2025}
}
```

并致谢：

* **LLaMA-Factory**（统一高效的 100+ 模型微调框架）
* 相关开源组件（PEFT/TRL/vLLM/FlashAttention-2 等）

# INT8-Hunyuan-Prompt-Enhancer

ComfyUI向けの **Hunyuan Prompt Enhancer（INT8 + Triton）** カスタムノードです。  
入力プロンプトを画像生成向けに強化し、`enhanced_prompt` を出力します。

**本プロジェクトは現在 `reprompt-INT8-shiba` を既定の自動ダウンロード先として運用しています。**

## 現在のモデル運用方針

- 自動ダウンロード先:  
  `Shiba-2-shiba/HunyuanImage-2.1-reprompt-INT8-by-Shiba-2-shiba`
- 既定INT8重み:  
  `HunyuanImage-2.1-reprompt-INT8-optimized.safetensors`
- 量子化方針:  
  **INT8 + FP16 keep（層ごとの混在）** により、速度と精度のバランスを狙う構成

## 主要ポイント

- Triton INT8カーネルによる高速化
- 実測ベンチでは `triton_on` と `triton_off` 比較で **約1.3x（1.36〜1.37x）** の高速化を確認
- `fallback率`（Triton失敗率）を可視化し、実際にTriton経路で動いているか確認可能
- Windows環境の権限問題を避けるため、HFキャッシュをプロジェクト内へ固定

## ノード

- 表示名: `INT8 Hunyuan Prompt Enhancer`
- カテゴリ: `XX/Hunyuan`
- 主入力:
  - `text`
  - `style_policy` (`illustration (Tag List)` / `photography (Detailed)`)
  - `temperature`, `top_p`, `top_k`, `max_new_tokens`, `seed`
  - `enable_thinking`, `device_map`, `attn_backend`, `quant_backend`
- 任意入力:
  - `custom_sys_prompt`
  - `quantized_safetensors`
- 出力:
  - `enhanced_prompt`

## 自動ダウンロード設定

`ckpts.yaml`

```yaml
int8:
  repo_id: "Shiba-2-shiba/HunyuanImage-2.1-reprompt-INT8-by-Shiba-2-shiba"
  subfolder: ""
  local_dir: "models/reprompt-INT8-shiba"
```

## ベースモデルについて（重要）

`reprompt-INT8-shiba` 側にベース分割重み（`model-00001-of-00004.safetensors` など）が存在しない場合、  
実装は自動でベースモデルディレクトリへフォールバックします。

- 優先:
  1. `PE_BASE_MODEL_DIR`（指定されている場合）
  2. `../promptenhancer`（既定フォールバック）

これにより、INT8重みをShibaリポジトリから取得しつつ推論を継続できます。

## 高速化ベンチ

```powershell
cd C:\Users\inott\Downloads\test\INT8-Hunyuan-Prompt-Enhancer
$env:PE_MODEL_DIR="C:\Users\inott\Downloads\test\promptenhancer"
$env:PE_INT8_VARIANT="optimized"
$env:PE_COMPARE_TRITON="1"
$env:PE_ENABLE_THINKING="0"
$env:PE_BENCH_RUNS="5"
$env:PE_BENCH_WARMUP="1"
$env:PE_MAX_NEW_TOKENS="512"
python tools\benchmark_triton_int8.py
```

出力で確認する指標:

- `prefill` / `decode` / `total`
- `decode_tok/s` / `total_tok/s`
- `[int8:...] fallback_rate=...`
- `[compare] ... total_speedup=...x`

## 依存関係

`requirements.txt` を使用:

```bash
pip install -r requirements.txt
```

主な依存:

- `bitsandbytes`
- `omegaconf`
- `tiktoken`
- `transformers`
- `torch`

## ライセンス

本リポジトリは [Tencent Hunyuan Community License Agreement](LICENSE.txt) の条件に従います。  
詳細は [NOTICE.txt](NOTICE.txt) を参照してください。


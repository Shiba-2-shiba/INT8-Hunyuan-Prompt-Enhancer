# RUNBOOK

## 前提
- 量子化モデル配置: `C:\Users\inott\Downloads\test\promptenhancer`
- INT8重み: `HunyuanImage-2.1-reprompt-INT8-optimized.safetensors`
- 自動ダウンロード先（既定）: `Shiba-2-shiba/HunyuanImage-2.1-reprompt-INT8-by-Shiba-2-shiba`

## 重要コマンド（再現手順）

### 1. 直ロードでTriton INT8推論テスト
```powershell
cd C:\Users\inott\Downloads\test\INT8-Hunyuan-Prompt-Enhancer
python tools\test_triton_int8.py
```

### 2. ComfyUIノード経由で推論テスト
```powershell
cd C:\Users\inott\Downloads\test\INT8-Hunyuan-Prompt-Enhancer
python tools\test_triton_int8_node.py
```

### 3. パス指定（必要な場合）
```powershell
$env:PE_MODEL_DIR="C:\Users\inott\Downloads\test\promptenhancer"
$env:PE_INT8_WEIGHTS="C:\Users\inott\Downloads\test\promptenhancer\HunyuanImage-2.1-reprompt-INT8-optimized.safetensors"
python tools\test_triton_int8_node.py
```

### 4. `optimized` 重みを明示指定して実行（推奨）
```powershell
cd C:\Users\inott\Downloads\test\INT8-Hunyuan-Prompt-Enhancer
$env:PE_MODEL_DIR="C:\Users\inott\Downloads\test\promptenhancer"
$env:PE_INT8_VARIANT="optimized"
python tools\test_triton_int8.py
python tools\test_triton_int8_node.py
```

### 5. Triton高速化の比較ベンチ（ON/OFF比較）
```powershell
cd C:\Users\inott\Downloads\test\INT8-Hunyuan-Prompt-Enhancer
$env:PE_MODEL_DIR="C:\Users\inott\Downloads\test\promptenhancer"
$env:PE_INT8_VARIANT="optimized"
$env:PE_COMPARE_TRITON="1"
$env:PE_BENCH_RUNS="5"
$env:PE_BENCH_WARMUP="1"
$env:PE_MAX_NEW_TOKENS="128"
python tools\benchmark_triton_int8.py
```

## ComfyUIノード推奨設定（Triton INT8）
- `quant_backend=triton_int8`
- `device_map=cuda:0`（VRAM不足時は `auto`）
- `attn_backend=auto`（必要時のみ `sdpa` / `flash_attention_2`）
- `quantized_safetensors` は空欄可（`PE_INT8_VARIANT` または `PE_INT8_WEIGHTS` で解決）

## 期待される結果
- `=== Output ===` または `=== Node Output ===` が表示され、推論テキストが出力される。
- `=== Perf ===` が表示され、実行時間が確認できる。
- ベンチでは `[compare] triton_on_vs_off: ...` が出力され、速度差（x倍）が確認できる。
- `rope_parameters` 警告は出ることがあるが致命的ではない。

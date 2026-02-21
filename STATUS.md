# STATUS

## 目的
INT8量子化（convert_to_quant出力）＋Tritonカーネルで、Hunyuan Prompt Enhancerの推論を動作させる。

## フォルダ構造
- `INT8-Hunyuan-Prompt-Enhancer/`
- `INT8-Hunyuan-Prompt-Enhancer/core/`  
  - `model.py`: 推論本体（Triton INT8対応）
  - `int8_triton.py`: INT8ロード＆Triton実行ヘルパ
- `INT8-Hunyuan-Prompt-Enhancer/kernels/`  
  - `int8_kernels.py`: Triton INT8カーネル（ベンダリング済み）
- `INT8-Hunyuan-Prompt-Enhancer/tools/`  
  - `test_triton_int8.py`: 直ロードで推論テスト
  - `test_triton_int8_node.py`: ComfyUIノード経由の推論テスト
- `promptenhancer/`（リポジトリ外の兄弟フォルダ）  
  - 量子化モデル本体（`HunyuanImage-2.1-reprompt-INT8-optimized.safetensors` など）

## 発見事項
- `device_map="cuda:0"` を `{"":0}` に正規化するため、CUDA判定はdict対応が必須。
- TritonはWindowsでキャッシュ/一時ディレクトリ権限に敏感。  
  `TRITON_CACHE_DIR` と `TEMP`/`TMP` の明示設定が必要。
- `rope_parameters` 警告はTransformers側の仕様で致命的ではない。

## 決定事項
- Triton INT8カーネルは `INT8-Hunyuan-Prompt-Enhancer/kernels/` にベンダリング。
- `torch_dtype` 警告は `dtype` 引数へ置換済み。
- INT8ロード時の missing keys は DEBUG ログへ降格。
- INT8重みの解決は `core/model.py::resolve_int8_weights()` へ集約（`PE_INT8_WEIGHTS` / `PE_INT8_VARIANT` 対応）。

## 次にやること
- ComfyUI実運用時に `quant_backend=triton_int8` + `device_map=cuda:0` の常用設定で長時間テスト。
- 必要なら `tools/benchmark_triton_int8.py` でベンチ結果を蓄積。

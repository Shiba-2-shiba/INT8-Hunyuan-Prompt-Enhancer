# TODO

## 目的
INT8量子化モデルをTritonカーネルで推論し、ComfyUIノードでも安定動作させる。

## フォルダ構造
- `core/`（推論本体とINT8/Tritonヘルパ）
- `kernels/`（Triton INT8カーネル）
- `tools/`（テストスクリプト）
- `promptenhancer/`（量子化モデル配置）

## 発見事項
- TritonはWindows環境で `TRITON_CACHE_DIR` と `TEMP`/`TMP` を明示的に設定する必要がある。
- `device_map` の正規化（dict）に対応しないとCPU/ CUDAの混在が起きる。

## 決定事項
- Triton INT8カーネルはリポジトリ内にベンダリング済み。
- `torch_dtype` 警告は `dtype` 引数へ修正済み。

## 次にやること
- 実機で `PE_INT8_VARIANT=high` を指定した推論結果を記録。
- ComfyUI本体で `quant_backend=triton_int8` / `device_map=cuda:0` の連続実行を確認。
- `tools/benchmark_triton_int8.py` の結果をベースライン化（GPU別）。

## 今回実装済み
- INT8重みの解決ロジックを追加（`explicit path` -> `PE_INT8_WEIGHTS` -> `PE_INT8_VARIANT` の順）。
- `promptenhancer_int8_high.safetensors` を含む複数候補の自動解決に対応。
- `tools/test_triton_int8.py` / `tools/test_triton_int8_node.py` に重み解決統一と簡易Perf表示を追加。
- `tools/benchmark_triton_int8.py` を追加（ウォームアップ・複数run平均）。

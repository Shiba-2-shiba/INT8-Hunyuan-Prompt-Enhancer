# INT8-Hunyuan-Prompt-Enhancer

ComfyUI向けの **Hunyuan Prompt Enhancer（INT8 + Triton）** カスタムノードです。  
入力プロンプトを画像生成向けに強化し、`enhanced_prompt` を出力します。

初回推論時に、UI で選択したモデルが未配置であれば自動ダウンロードされます。


## 現在のモデル運用方針

- UI から選択可能なモデル:
  - `INT8 (Standard)`  
    `Shiba-2-shiba/HunyuanImage-2.1-reprompt-INT8-by-Shiba-2-shiba`
  - `INT8 (Heretic)`  
    `Shiba-2-shiba/HunyuanImage-2.1-reprompt-heretic-INT8-by-Shiba-2-shiba`
- 既定モデル:
  `INT8 (Standard)`
- ローカル保存先:
  - `models/reprompt-INT8-shiba`
  - `models/reprompt-heretic-INT8-shiba`
- 量子化方針:  
  **INT8 + FP16 keep（層ごとの混在）** により、速度と精度のバランスを狙う構成

## 主要ポイント

- UIはTriton固定で運用（bitsandbytes非対応）
- INT8重みは `model.safetensors` を自動採用（UIでのフルパス指定不要）
- Triton INT8カーネルにより高速化（実測で約1.3x）
- Windowsの権限問題回避のため、HFキャッシュをプロジェクト内に固定

## 量子化モデルについて

本モデルは `convert_to_quant` を用いて、以下の方針で量子化されています。

- fp16 keep + SVD による INT8 量子化
- 層ごとに fp16 / INT8 を混在させ、速度と品質のバランスを調整

## ノード

- 表示名: `INT8 Hunyuan Prompt Enhancer`
- カテゴリ: `Hunyuan`
- 主入力:
  - `text`
  - `model_variant` (`INT8 (Standard)` / `INT8 (Heretic)`)
  - `style_policy` (`illustration (Tag List)` / `photography (Detailed)`)
  - `temperature`, `top_p`, `top_k`, `max_new_tokens`, `seed`
  - `enable_thinking`, `device_map`, `attn_backend`
- 任意入力:
  - `custom_sys_prompt`
- 出力:
  - `enhanced_prompt`

## 自動ダウンロード設定

`ckpts.yaml`

```yaml
models:
  "INT8 (Standard)":
    default: true
    repo_id: "Shiba-2-shiba/HunyuanImage-2.1-reprompt-INT8-by-Shiba-2-shiba"
    subfolder: ""
    local_dir: "models/reprompt-INT8-shiba"
  "INT8 (Heretic)":
    repo_id: "Shiba-2-shiba/HunyuanImage-2.1-reprompt-heretic-INT8-by-Shiba-2-shiba"
    subfolder: ""
    local_dir: "models/reprompt-heretic-INT8-shiba"
```

## 依存関係

`requirements.txt` を使用:

```bash
pip install -r requirements.txt
```

主な依存:

triton/triton-windows
omegaconf
tiktoken
protobuf
sentencepiece
blobfile

## ライセンス

本リポジトリは [Tencent Hunyuan Community License Agreement](LICENSE.txt) の条件に従います。  
詳細は [NOTICE.txt](NOTICE.txt) を参照してください。

## 謝辞

INT8 + Triton の推論コードは以下を参考にしています（MITライセンス）。

```
https://github.com/silveroxides/ComfyUI-QuantOps
```


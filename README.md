# INT8-Hunyuan-Prompt-Enhancer

ComfyUI向けの**Hunyuan Prompt Enhancer（INT8版）**カスタムノードです。  
入力したプロンプトを、画像生成向けに整理・補強した `Reprompt: ...` 形式へ変換します。

このカスタムノードは、[leeooo001/comfyui-Hunyuan-PromptEnhancer](https://github.com/leeooo001/comfyui-Hunyuan-PromptEnhancer) の作者が公開されているINT8モデル（`leeooo001/Hunyuan-PromptEnhancer-INT8`）を利用しています。  
スクリプト構成は大幅に変更しつつ、本カスタムノードは同プロジェクトの内容を改良して作成しました。素晴らしいモデルと先行実装の公開に、心より感謝します。

---

## 主な特徴

- **INT8モデルを自動取得**  
  初回実行時に `ckpts.yaml` の設定を参照し、必要なモデルを自動ダウンロードします。
- **用途別のプロンプト最適化**
  - `illustration (Tag List)`：イラスト/アニメ向けにタグ列へ最適化
  - `photography (Detailed)`：写真調・高精細向けに詳細化
- **後処理によるスタイル保護**  
  illustrationモードでは、不要な写真系キーワードの混入を抑制します（`policy.yaml` で調整可能）。
- **推論パラメータの調整が可能**  
  `temperature`, `top_p`, `max_new_tokens`, `seed`, `enable_thinking` などをノード上で指定できます。
- **デバイス/Attention backend切り替え**  
  `cuda:0` / `auto` / `cpu`、`sdpa` / `flash_attention_2` などを選択できます。

---

## ノード情報

- **ノード名（表示）**: `INT8 Hunyuan Prompt Enhancer`
- **カテゴリ**: `XX/Hunyuan`
- **入力（主要）**:
  - `text`: 元プロンプト
  - `style_policy`: `illustration (Tag List)` or `photography (Detailed)`
  - `temperature`, `top_p`, `max_new_tokens`, `seed`
  - `enable_thinking`, `device_map`, `attn_backend`
- **入力（任意）**:
  - `custom_sys_prompt`: システムプロンプトを上書き
- **出力**:
  - `enhanced_prompt`（STRING）

---

## 使い方（基本）

1. ComfyUIの `custom_nodes` 配下に本リポジトリを配置します。
2. ComfyUIを起動（または再起動）します。
3. ノード一覧から `INT8 Hunyuan Prompt Enhancer` を追加します。
4. `text` に元プロンプトを入力し、`style_policy` を選択します。
5. 必要に応じて生成パラメータを調整して実行します。
6. 出力された `enhanced_prompt` を、画像生成ノードのプロンプト入力へ接続します。

---

## 設定ファイル

### `ckpts.yaml`
モデル取得先と保存先を定義します。

```yaml
int8:
  repo_id: "leeooo001/Hunyuan-PromptEnhancer-INT8"
  subfolder: ""
  local_dir: "models/reprompt-INT8"
```

### `policy.yaml`
illustrationモード時に除外したい語彙（禁止語）などのポリシーを調整できます。

---

## サンプルワークフロー

`ComfyUI-INT8-Hunyuanpromptenhancer-example.json` を参考に、ノード接続を確認できます。

---

## 謝辞

- INT8モデルおよび先行実装を公開された  
  [leeooo001/comfyui-Hunyuan-PromptEnhancer](https://github.com/leeooo001/comfyui-Hunyuan-PromptEnhancer) の作者に深く感謝いたします。
- 本プロジェクトは上記の知見をベースに、構成・処理・運用性を見直して再設計したものです。

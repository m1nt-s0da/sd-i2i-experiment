# SD Playground

Stable Diffusion 1.5を使った画像生成・画像変換（i2i）と、デノイズ中の潜在表現の変化を可視化する実験用ツールです。TOML設定ファイルで入力画像、モデル、生成条件、出力先などを指定します。

## 実行方法

Python 3.14以上と[uv](https://docs.astral.sh/uv/)を使用します。依存関係をインストールし、デモを実行します。

```powershell
uv sync
uv run i2i --config demo.toml
```

アニメ画像の例は次のとおりです。

```powershell
uv run i2i --config demo_anime.toml
```

`--config`を省略すると`demo.toml`が使われます。設定内の画像・出力パスは、コマンドを実行したディレクトリからの相対パスです。デモ設定はCUDAとfloat16を指定しています。モデルウェイトは初回実行時に必要に応じてHugging Faceから取得されます。

## 動作の概要

- 入力には既存画像、またはテキストから生成した画像を指定できます。
- `start_step`はデノイズ開始位置を表す0始まりのインデックスです。0より大きい場合、入力画像をVAEで潜在表現に変換してから、その位置のノイズを加えます。値が大きいほど入力画像を保ちやすくなります。
- `start_step = 0`かつ`[i2i.step_map]`なしの場合は純粋なガウスノイズから生成するため、入力画像は使われず、text-to-imageと同じ初期状態になります。
- `[i2i.step_map]`を指定すると、潜在画像の位置ごとにi2iのデノイズステップ数を変えます。試走中の予測x0の変化量からステップ数を決め、ヒートマップを保存できます。試走は、入力画像に(`steps` − `probe_steps`)時点のノイズを加えた潜在から最後のステップまで行います。この機能で使えるサンプラーは`euler`と`ddim`です。
- `[[i2i.controlnet]]`でControlNetによる条件付けを追加できます。現在は深度（`type = "depth"`）に対応し、深度画像を指定しない場合は入力画像から推定します。

## 設定

`demo.toml`と`demo_anime.toml`に実行可能な設定例があります。設定は`[i2i]`をルートにし、主な項目は次のとおりです。

| 項目 | 説明 |
| --- | --- |
| `prompt`, `negative_prompt` | 生成プロンプトとネガティブプロンプト |
| `guidance_scale` | classifier-free guidanceの強度 |
| `steps` | 推論ステップ数 |
| `start_step` | i2iのデノイズ開始位置 |
| `seed` | VAEサンプリングと初期ノイズなどに使うシード |
| `latent_pixel_count` | 距離グラフで追跡する潜在画像の座標数 |
| `output_path` | 生成画像の保存先 |
| `latent_distance_plot_path` | デノイズステップ間の潜在ベクトル距離グラフの保存先（省略可） |
| `x0_distance_plot_path` | 予測x0間の距離グラフの保存先（省略可） |

入力は`[i2i.input]`で指定します。画像を使う場合は`type = "image"`と`path`を、プロンプトから入力画像を生成する場合は`type = "generate"`と`prompt`を指定します。生成入力では`steps`、`guidance_scale`、`width`、`height`、`seed`、`save_path`も設定できます。

`[i2i.pipeline]`では、Stable Diffusion 1.5のモデル名、`dtype`、`device`、サンプラー（`euler`または`ddim`）、`enable_attention_slicing`、`enable_model_cpu_offload`を指定します。

### step map

```toml
[i2i.step_map]
probe_steps = 5
refine_steps = 1
threshold = 1.0
pool_kernel = 3
probe_mode = "trajectory"
metric = "step_delta"
heatmap_path = "output/step-map.png"
```

試走するステップ数（`probe_steps`。最後のステップから数えます）、各ピクセルに割り当てる最小ステップ数（`refine_steps`）、変化判定のしきい値（`threshold`）、変化量を平滑化するカーネルサイズ（`pool_kernel`）、試走の方式（`probe_mode`。`trajectory`は試走の軌跡を辿り、`renoise`は各ステップで入力にノイズを加え直してx0を推定）、変化量の指標（`metric`。`step_delta`は連続するステップ間のx0差、`input_distance`は入力潜在とx0の差）を設定できます。ヒートマップが不要な場合は`heatmap_path`を省略します。

### 深度ControlNet

```toml
[[i2i.controlnet]]
type = "depth"
model = "lllyasviel/sd-controlnet-depth"
conditioning_scale = 1.0
# 深度画像を自分で指定する場合
# condition_image = "input-depth.png"
# 推定した深度画像を保存する場合
# save_image = "output/depth.png"
# estimation_model = "Intel/dpt-hybrid-midas"
```

## 出力例

`demo.toml`は`output/i2i.png`と`output/i2i-heatmap.png`を、`demo_anime.toml`は`output/i2i_anime.png`と`output/i2i-anime-heatmap.png`を生成します。出力先の親ディレクトリは必要に応じて作成されます。

# latent pixelごとに深さを変えるi2i: 試走方式・指標の比較

## 目的

入力画像と大きく矛盾するプロンプトでi2iすると、矛盾部分が変わらない。矛盾する部分だけをfull noiseに近い深さから推論し、残りは保つ。そのための「pixelごとの深さ（step_map）」を、入力画像と目標プロンプトだけから決められるかを調べる。

## 条件

- 入力セットは4つ。`city`（夜景→昼、depth ControlNetあり）、`anime`（少女→少年）、`landscape`（夏の草原と小屋→雪景色）、`cat`（ソファの猫→犬）。後ろ2つは本スクリプトでSD1.5から生成した。
- シードは各3つ（42〜44）。steps=30、`start_step=0`（最大深さ30）、`probe_steps=15`、`refine_steps=5`、`pool_kernel=3`。
- 手法（入力画像と目標プロンプトのみ使用）:
  - `u6/u12/u18`: 一様i2i（`start_step`=6/12/18）。
  - step_map: `probe_mode`（`traj`=trajectory、`reno`=renoise）× `metric`（`step`=step_delta、`inp`=input_distance）× `threshold`（0.7/1.0/1.3）の12通り。
- 評価専用（手法には使えない）:
  - `t2i`: 入力を使わない純ノイズからの生成。
  - 参照信号: 元プロンプトと目標プロンプトで、入力に同じノイズを加えたx0予測の差（末尾15ステップの平均、3×3平滑化）。値が大きい所を「変わるべき領域」とみなす。
- 指標: `clip_edit`（CLIP ViT-B/32で目標キャプション−元キャプション。大きいほど目標寄り）、`mae`（入力との平均絶対差。`high`=参照信号の上位40%、`low`=下位40%）、`corr`（深さ地図と参照信号の相関）、平均深さ。
- 実装: [depth_study.py](experiments/depth_study.py)。結果は`output/experiments/study/`（`results.json`と各セットの`*_montage.png`）。

## 結果

### 一様i2iの限界

`clip_edit`の平均（t2i / u6 / u12 / u18）:

| セット | t2i | u6 | u12 | u18 |
|---|---|---|---|---|
| city | 3.5 | -3.4 | -3.8 | -3.1 |
| anime | 8.4 | 3.2 | -0.4 | 0.5 |
| landscape | 8.8 | 4.5 | -4.8 | -6.1 |
| cat | 4.1 | 5.8 | 5.9 | -4.6 |

cityは深さ6でも目標（昼）に寄らない。それ以外も、入力に近い深さ（u12以降）では目標に寄らないか、急に目標に寄れなくなる。

### 深さ地図と参照信号の相関

`corr` / 平均深さ。

| 条件 | city | anime | landscape | cat |
|---|---|---|---|---|
| traj_step_t0.7 | 0.42 / 26 | 0.64 / 22 | 0.47 / 25 | 0.60 / 21 |
| traj_step_t1.0 | 0.47 / 17 | 0.64 / 15 | 0.51 / 15 | 0.63 / 14 |
| traj_step_t1.3 | 0.47 / 8 | 0.61 / 10 | 0.61 / 8 | 0.64 / 9 |
| traj_inp_t0.7 | 0.34 / 26 | 0.49 / 23 | 0.26 / 27 | 0.28 / 26 |
| traj_inp_t1.0 | 0.34 / 17 | 0.57 / 17 | 0.36 / 17 | 0.58 / 16 |
| traj_inp_t1.3 | 0.26 / 9 | 0.56 / 11 | 0.38 / 8 | 0.66 / 8 |
| reno_step_t0.7 | 0.43 / 25 | 0.65 / 21 | 0.57 / 24 | 0.65 / 22 |
| reno_step_t1.0 | 0.48 / 17 | 0.68 / 15 | 0.64 / 15 | 0.77 / 14 |
| reno_step_t1.3 | 0.47 / 9 | 0.68 / 10 | 0.70 / 9 | 0.85 / 10 |
| reno_inp_t0.7 | 0.32 / 27 | 0.45 / 25 | 0.19 / 28 | 0.27 / 28 |
| reno_inp_t1.0 | 0.32 / 17 | 0.51 / 16 | 0.12 / 17 | 0.55 / 16 |
| reno_inp_t1.3 | 0.24 / 8 | 0.53 / 9 | 0.20 / 7 | 0.77 / 8 |

- 全セット・全条件で正の相関があり、目標プロンプトだけで決めた深さ地図は、元プロンプトとの差が大きい領域にある程度寄る。
- `step`系（`step_delta`）の方が`inp`系（`input_distance`）より相関が高いセットが多い（city、landscape、cat）。特に`reno_step`が最も安定して高い（0.43〜0.85）。animeでは`step`系が0.6台、`inp`系が0.5前後。
- `traj`と`reno`の差は、`step_delta`では`reno`がやや高い程度。`input_distance`では一貫した差はない。

### 変化量が同じ一様i2iとの比較

出力の`mae_all`が同じ一様i2iの`clip_edit`（u6/u12/u18/t2iを線形補間）との差を「gain」とした。

- **anime**: `inp`系が最も効いた。`reno_inp_t1.0`は`mae_all`33、`clip_edit`4.1で、補間した一様i2iは0.5（gain +3.6）。`traj_inp_t0.7`は+2.6、`reno_inp_t0.7`は+2.1。参照信号上位/下位のMAE比も、一様i2iの1.6〜2.2に対し、2.1〜4.4と偏る。
- **cat**: 低MAE側（`mae_all`15〜17）で、一様i2i（u18、-4.6）が目標に寄れないのに対し、`traj_step_t1.3`などは`clip_edit`が+3.2、`reno_inp_t1.3`が+1.6。ただし一様i2iのu12は5.9で、一様i2iが急に失敗する境目に当たっているため、gainの値（+6〜+7）は過大評価で、補間も粗い。中間のしきい値（1.0）では`inp`系はgainがマイナス。
- **landscape**: 多くの条件でgainは負か0付近で、一様i2iに勝てなかった。深さ地図は広く散らばる。
- **city**: gainは±1以内。どの設定でも`clip_edit`は負のまま（昼に寄れない）。`mae_high/low`もほぼ1.5〜1.8で、一様i2iと同程度。

montage（`*_montage.png`）で見ると、landscapeでは`inp`系の深さが空・木・小屋にまだらに散り、`reno_inp_t1.0`の生成画像は小屋まわりだけが雪で、草と木は夏のまま。
各セットのモンタージュ（シード42）。上段は入力・参照信号（評価専用）・t2i・一様i2i、以降は条件ごとの生成画像と深さ地図（明るいほど深い）。

### anime

![anime](output/experiments/study/anime_montage.png)

### landscape

![landscape](output/experiments/study/landscape_montage.png)

### cat

![cat](output/experiments/study/cat_montage.png)

### city

![city](output/experiments/study/city_montage.png)

## 考察

1. 末尾の試走だけでも、変わるべき領域の位置はある程度とれる（`corr` 0.3〜0.85）。ただし精度はセットに強く依存する。
2. 指標は、位置の相関では`step_delta`、変化の質（animeの`clip_edit`/gain）では`input_distance`が良く、一貫した勝者はいない。
3. `threshold`は平均深さと変化量を大きく決めるが、最適値はセットごとに違う。0.7では深すぎて入力を壊し、1.3では浅すぎて目標に寄らない傾向。
4. 一様i2iに対する明確な改善が見えたのはanimeと（過大評価を含めて）catのみ。cityとlandscapeでは改善が見えない。

## 限界

- セットは4つ、シードは3つ、サンプル数が少ない。差が小さい条件は区別できない。
- `clip_edit`は粗い代理指標で、目標への寄りと画像の破綻を区別しない。`mae`は変化量を測るだけで、品質は測らない。
- 参照信号は、元プロンプトの結果と目標プロンプトのx0予測の差で、真の「変わるべき領域」ではない。`landscape`と`cat`の元プロンプトは入力生成に使ったものだが、`city`と`anime`の入力は、元プロンプトに沿って生成したとは限らない（`demo.toml`のコメントにあるプロンプトを使った）。
- `landscape`・`cat`の入力は、`mae`・`clip_edit`・参照信号を算出するため、元プロンプトで生成した。手法側は元プロンプトを使っていない。
- 深さは`steps − start_step`が上限。最深を30にするには`start_step=0`が必要。
- gainは、一様i2iの3点とt2iの線形補間と比べているため、一様i2iが急に変化する場合は大きくずれる。

## 実装上の課題

- 一様i2iとstep_mapで、深さの上限が`steps − start_step`に固定されている。

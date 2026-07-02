# Streaming Source Separation EnCodec 模型结构与张量流

## 来源

- 数据 handoff：[../../DATA_HANDOFF_STREAMING_SEP_ENCODEC.md](../../DATA_HANDOFF_STREAMING_SEP_ENCODEC.md)
- 数据加载：[streaming_sep/data.py](streaming_sep/data.py)
- 模型结构：[streaming_sep/model.py](streaming_sep/model.py)
- handoff 训练入口：[streaming_sep/train_handoff_indexes.py](streaming_sep/train_handoff_indexes.py)
- 配置：[configs/encodec_sep_200m.yaml](configs/encodec_sep_200m.yaml)

## 一句话总结

当前模型是一个流式友好的 source-conditioned EnCodec token separator：输入 mixture 的 RVQ token 和目标 `stem_id`，主时间维 causal Transformer 预测目标 stem 的 RVQ codebook 0，再用同一帧 codebook 维 MTP 小 Transformer 预测 residual codebooks 1..7。训练是联训，loss 为 `coarse_loss + residual_loss_weight * residual_loss`。

## 当前默认配置

| 项 | 值 | 含义 |
|---|---:|---|
| `sample_rate` | 24000 | EnCodec 24 kHz |
| `codebook_size` / `V` | 1024 | 每个 RVQ codebook 的 token 类别数 |
| `num_codebooks` / `K` | 8 | EnCodec RVQ codebook 数 |
| `d_model` / `D` | 1024 | 主 Transformer hidden size |
| `num_layers` | 18 | 主时间维 Transformer 层数 |
| `num_heads` | 16 | 主时间维 attention heads |
| `ffn_dim` | 3072 | 主 Transformer FFN hidden size |
| `sliding_window` | 1024 | 每个时间步最多 attend 最近 1024 帧 |
| `crop_frames` | 1200 | 训练样本最大裁剪帧数 |
| `residual_predictor` | `mtp` | residual RVQ 使用 codebook 维小 Transformer |
| `mtp_layers` | 2 | MTP 小 Transformer 层数 |
| `mtp_heads` | 8 | MTP attention heads |
| `mtp_ffn_dim` | 2048 | MTP FFN hidden size |
| `residual_loss_weight` | 0.5 | residual loss 权重 |

## Stem 条件

`stem_id` 是一个离散类别 id：

```text
vocal = 0
bass  = 1
drums = 2
other = 3
```

在 batch 中它的 shape 是 `[B]`。当前 `num_stems=4`，`d_model=1024`，所以：

```text
stem_emb table: [4, 1024]
stem_id:        [B]
stem_emb(...):  [B, 1024]
unsqueeze(1):   [B, 1, 1024]
```

然后 broadcast 加到每个时间步。若当前 batch pad 后长度为 `T`，则 `[B,1,1024]` 会广播到 `[B,T,1024]`：

```text
mix_hidden:       [B, T, 1024]
src_prev_hidden:  [B, T, 1024]
stem_hidden:      [B, 1, 1024] -> broadcast -> [B, T, 1024]
x:                [B, T, 1024]
```

当前没有 prefix token / sink attention；stem 信息是每帧 additive conditioning。因为每个时间步都会重新加 stem embedding，所以它不依赖长上下文窗口是否保留起始 token。

## 总体 Pipeline

```text
1. 四 stem index 读取
   vocal/bass/drums/other index.json 或 index-mnt.json
   -> 按 relative_path 建索引

2. 曲目取交集
   四个 stem 都有 token 的 relative_path 才进入训练
   -> common_keys

3. 样本展开
   每个 relative_path 展开成四条训练样本:
   (track, vocal), (track, bass), (track, drums), (track, other)

4. shard 懒加载
   torch.load(shard_path)["records"][local_index]
   -> mixture_codes   [T_raw, K] = [T_raw, 8]
   -> separated_codes [T_raw, K] = [T_raw, 8]

5. 裁剪与 batch 整理
   crop -> pad -> collate
   -> mixture_codes [B, T, K] = [B, T, 8]
   -> source_codes  [B, T, K] = [B, T, 8]
   -> mask          [B, T]
   -> stem_id       [B]

6. 主模型输入 embedding
   mixture_codes -> mix_hidden      [B, T, D] = [B, T, 1024]
   source_codes  -> src_prev_hidden [B, T, D] = [B, T, 1024]
   stem_id       -> stem_hidden     [B, 1, D] = [B, 1, 1024]

7. 主时间维 causal Transformer
   x = mix_hidden + src_prev_hidden + stem_hidden
   -> hidden [B, T, D] = [B, T, 1024]

8. q0 coarse head
   hidden -> coarse_logits [B, T, V] = [B, T, 1024]
   target = source_codes[..., 0]     = [B, T]

9. residual MTP
   hidden + source_codes[..., :-1]
   -> residual_logits [B, T, K-1, V] = [B, T, 7, 1024]
   target = source_codes[..., 1:]     = [B, T, 7]

10. 联合 loss
   loss = coarse_loss + residual_loss_weight * residual_loss
```

## crop / pad / collate

这三个是数据整理操作，不是模型模块。

| 操作 | 位置 | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| `crop` | Dataset `__getitem__` | 长序列截取到最多 `crop_frames`。训练随机起点，验证固定开头 | `[T_raw,K]` | `[T_crop,K]` |
| `pad` | `collate_sep` | batch 内样本长度不同，补 0 到同一个 `T` | 多个 `[T_i,K]` | `[B,T,K]` |
| `collate` | DataLoader | 把多条 `SepItem` 组装成 batch dict | `list[SepItem]` | tensors + `utt_id` |

`mask [B,T]` 标记真实 token 位置。真实位置为 `True`，padding 位置为 `False`。loss 只统计 `mask=True` 的位置。

## 数据张量

| 名称 | Shape | dtype | 含义 |
|---|---:|---|---|
| `mixture_codes` | `[B,T,K] = [B,T,8]` | `long` | mixture EnCodec RVQ token |
| `source_codes` | `[B,T,K] = [B,T,8]` | `long` | 当前 stem 的目标 separated EnCodec RVQ token |
| `mask` | `[B,T]` | `bool` | 非 padding 位置 |
| `stem_id` | `[B]` | `long` | source condition 类别 |
| `stem_hidden` | `[B,1,D] = [B,1,1024]` | float | stem 条件 embedding |
| `mix_hidden` | `[B,T,D] = [B,T,1024]` | float | mixture RVQ embedding sum |
| `src_prev_hidden` | `[B,T,D] = [B,T,1024]` | float | 上一帧 source RVQ teacher-forcing embedding |
| `hidden` | `[B,T,D] = [B,T,1024]` | float | 主时间维 Transformer 输出 |
| `coarse_logits` | `[B,T,V] = [B,T,1024]` | float | q0 分类 logits |
| `residual_logits` | `[B,T,K-1,V] = [B,T,7,1024]` | float | q1..q7 分类 logits |

其中当前默认 `K=8`，`V=1024`，`D=1024`。

## Dataset 流程

`StreamingSepHandoffDataset` 读取四个 stem 的 index，并按 `relative_path` 做交集。

每条 index item 指向一个 `.pt` shard：

```python
payload = torch.load(shard_path, map_location="cpu", weights_only=False)
record = payload["records"][local_index]
mix = record["mixture_codes"].long()       # [T_raw, K] = [T_raw, 8]
src = record["separated_codes"].long()     # [T_raw, K] = [T_raw, 8]
```

然后按 stem 生成：

```text
utt_id = f"{relative_path}::{stem}"
mixture_codes = mix
source_codes = src
stem_id = STEM_TO_ID[stem]
```

训练集 `random_crop=True`，验证集 `random_crop=False`。DataLoader 训练集 `shuffle=True`。

## Collate 流程

对一个 batch 内的多条样本：

```text
sample_0: mixture [T0,K] = [T0,8], source [T0,K] = [T0,8]
sample_1: mixture [T1,K] = [T1,8], source [T1,K] = [T1,8]
...
T = max(T0, T1, ...)
```

生成：

```text
mixture_codes [B,T,K] = [B,T,8]  # padding 位置为 0
source_codes  [B,T,K] = [B,T,8]  # padding 位置为 0
mask          [B,T]    # 前 Ti 个位置 True，其余 False
stem_id       [B]
utt_id        list[str]
```

## 主模型结构

模型类是 `DualStreamEncodecSeparator`。

### 1. Mixture RVQ Embedding

每个 codebook 有一个独立 embedding table：

```text
mix_code_emb[k]: V -> D = 1024 -> 1024
共有 K=8 个 embedding table，每个 table 参数 shape 为 [1024, 1024]
```

对 `mixture_codes [B,T,K]`，逐 codebook 查表后求和，并除以 `sqrt(K)`：

```text
mixture_codes[..., k]:          [B,T]
mix_code_emb[k](...):           [B,T,1024]
sum over k=0..7 then / sqrt(8): [B,T,1024]
mix_hidden:                     [B,T,D] = [B,T,1024]
```

作用：把当前 mixture 的完整 RVQ stack 压成每帧一个 hidden condition。

### 2. Source Teacher Forcing Embedding

目标 source RVQ stack 也有独立 embedding：

```text
source_code_emb[k]: V -> D = 1024 -> 1024
共有 K=8 个 embedding table，每个 table 参数 shape 为 [1024, 1024]
```

训练时使用整帧右移：

```text
source_bos:                    [1024]
source_codes[:, :-1, :]:       [B,T-1,8]
embed(source_codes[:, :-1,:]): [B,T-1,1024]
src_prev_hidden:               [B,T,1024]
```

shape：

```text
src_prev_hidden: [B,T,D] = [B,T,1024]
```

作用：让第 `t` 帧预测时可以看到第 `t-1` 帧已生成/真值 source RVQ stack，符合 autoregressive streaming 生成形式。

### 3. Stem Embedding

```text
stem_emb table: [4,1024]
stem_id:        [B]
stem_hidden = stem_emb(stem_id).unsqueeze(1)
stem_hidden:    [B,1,D] = [B,1,1024]
```

作用：告诉模型当前要输出哪个 stem。没有这个条件时，同一条 mixture 会对应四个不同 target，训练监督冲突。

### 4. 主时间维 Transformer

输入：

```text
mix_hidden:      [B,T,1024]
src_prev_hidden: [B,T,1024]
stem_hidden:     [B,1,1024] -> broadcast [B,T,1024]
x = RMSNorm(mix_hidden + src_prev_hidden + stem_hidden)
x:               [B,T,D] = [B,T,1024]
```

经过 `num_layers=18` 层 causal Transformer：

```text
hidden = Transformer(x, mask, sliding_window=1024)
hidden: [B,T,D] = [B,T,1024]
```

attention 约束：

- causal：第 `t` 帧不看未来帧。
- sliding window：最多看最近 `1024` 帧。
- padding mask：不 attend padding token。

## 输出头

### 1. Coarse Head

```text
coarse_head weight: [D,V] = [1024,1024]  # written conceptually; PyTorch Linear stores [V,D]
hidden:             [B,T,1024]
coarse_logits:      [B,T,V] = [B,T,1024]
target:             source_codes[..., 0] = [B,T]
```

作用：预测目标 source 的第 0 个 RVQ codebook，也就是 q0。

### 2. Residual MTP Predictor

当前默认 `residual_predictor: mtp`。

MTP 是一个小 causal Transformer，沿同一帧的 codebook 维度工作，不沿时间维工作。训练输入：

```text
hidden:     [B,T,D] = [B,T,1024]
prev_codes: source_codes[..., :-1] = q0..q6  # [B,T,K-1] = [B,T,7]
target:     source_codes[..., 1:]  = q1..q7  # [B,T,K-1] = [B,T,7]
```

内部把 `[B,T]` 展平成 `B*T` 个 frame：

```text
hidden          [B,T,D]     = [B,T,1024]   -> [B*T,1,1024]
prev_codes      [B,T,K-1]   = [B,T,7]      -> [B*T,7]
prev_code_emb   V -> D      = 1024 -> 1024
codebook_pos    [K-1]       = [7]
codebook_pos_emb table      = [7,1024]
```

每个 residual 位置的输入是：

```text
hidden_frame:                    [B*T,1,1024] -> broadcast [B*T,7,1024]
prev_code_emb(prev_codes):        [B*T,7,1024]
codebook_pos_emb:                 [7,1024] -> broadcast [B*T,7,1024]
mtp_x:                            [B*T,7,1024]
```

然后小 Transformer 做 codebook 维 causal attention：

```text
q0      -> predict q1
q0,q1   -> predict q2
...
q0..q6  -> predict q7
```

输出：

```text
MTP hidden after 2 layers: [B*T,7,1024]
MTP head logits:          [B*T,7,1024]
residual_logits:          [B,T,K-1,V] = [B,T,7,1024]
```

### 3. Parallel 兼容模式

也可以设置：

```yaml
residual_predictor: parallel
```

这时 residual 不走 MTP，而是 `K-1` 个独立 linear heads：

```text
linear_i(hidden): [B,T,V] = [B,T,1024]
stack 7 heads:    [B,T,K-1,V] = [B,T,7,1024]
```

当前默认不是这个模式。

## Loss

当前训练是联训，没有 detach hidden，也没有 staged schedule。

```text
coarse_loss = CE(coarse_logits [B,T,1024], source_codes[..., 0] [B,T])
residual_loss = CE(residual_logits [B,T,7,1024], source_codes[..., 1:] [B,T,7])
loss = coarse_loss + residual_loss_weight * residual_loss
```

mask 处理：

- `coarse_loss` 只统计 `mask=True` 的 `[B,T]` 位置。
- `residual_loss` 只统计 `mask=True` 的 `[B,T,K-1]` 位置。
- padding 位置不参与 loss。

默认：

```text
residual_loss_weight = 0.5
```

梯度路径：

```text
coarse_loss
  -> coarse_head
  -> main Transformer
  -> mixture/source/stem embeddings

residual_loss
  -> residual MTP
  -> hidden
  -> main Transformer
  -> mixture/source/stem embeddings
```

所以 MTP loss 会正常更新主干，这就是当前“联训”。

## 训练 Pipeline

```text
batch from DataLoader
  mixture_codes [B,T,K] = [B,T,8]
  source_codes  [B,T,K] = [B,T,8]
  mask          [B,T]
  stem_id       [B]

move to device
  -> autocast bf16/fp16 if CUDA amp enabled

model forward
  -> coarse_logits   [B,T,V]     = [B,T,1024]
  -> residual_logits [B,T,K-1,V] = [B,T,7,1024]
  -> hidden          [B,T,D]     = [B,T,1024]

separation_loss
  -> coarse_loss
  -> residual_loss
  -> total loss

backward
  -> gradient clipping
  -> AdamW step
  -> checkpoint per epoch
```

当前 optimizer：

```text
AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
```

checkpoint 包含：

```text
model state_dict
optimizer state_dict
config
epoch
step
param_count
index_paths
stems
```

## 推理 / 流式生成张量流

推理入口会先把 mixture waveform 编码成：

```text
mixture_codes [T,K] = [T,8]
```

用户指定：

```text
--stem vocal|bass|drums|other
stem_id [1]
```

生成循环逐时间帧执行：

```text
source_prefix: [1,t,K] = [1,t,8]

for step in 0..T-1:
  dummy_current_frame: [1,1,K] = [1,1,8]
  source_in = concat(source_prefix, dummy_current_frame)  # [1,step+1,8]
  model(
    mixture_codes[:, :step+1],  # [1,step+1,8]
    source_codes=source_in,     # [1,step+1,8]
    stem_id=stem_id             # [1]
  )
  hidden_last_frame = hidden[:, -1]             # [1,1024]
  coarse_logits[:, -1]                          # [1,1024]
  coarse = argmax/sample(coarse_logits[:, -1])  # q0, [1]

  if MTP:
    generated = [q0]
    for residual position:
      prev_codes = stack(generated)             # [1,n], n from 1 to 7
      MTP(hidden_last_frame, prev_codes)         # [1,n,1024]
      next_q = MTP(... )[:, -1]                  # [1]
      generated.append(next_q)
    next_codes = [q0, q1, ..., q7]               # [1,8]

  source_prefix = concat(source_prefix, next_codes)
```

输出：

```text
source_codes [T,K] = [T,8]
```

再用 EnCodec decode 得到目标 stem waveform。

当前 `generate_stream` 是逻辑流式：每一步不看未来帧，但工程上会重算 prefix，还没有 KV-cache。

## 和 Qwen3-TTS 的对应关系

当前结构参考 Qwen3-TTS-12Hz 的思想：

| Qwen3-TTS-12Hz | 当前模型 |
|---|---|
| 主 Talker 沿时间维预测 zeroth codebook | 主 Transformer 沿时间维预测 q0 |
| MTP / code predictor 生成 residual codebooks | `ResidualMTPPredictor` 生成 q1..q7 |
| 多码本 acoustic tokens | EnCodec RVQ tokens |
| text + codec dual-track | mixture RVQ + source feedback + stem condition |

不同点：

- Qwen3-TTS 是 TTS，条件是 text / speaker / language。
- 当前任务是 source separation，条件是 mixture token 和 stem id。
- Qwen3-TTS 工程推理有 cache；当前实现还没有 KV-cache。

## 关键结论

- 当前模型不是单码本模型；它训练和生成完整 EnCodec RVQ stack。
- 当前 residual RVQ 默认使用 MTP，不是 parallel heads。
- 当前训练是主干和 MTP 联合训练，不做 hidden detach。
- 当前结构满足因果流式约束，但推理实现还不是高效 KV-cache 流式。
- `stem_id` 是必需条件，否则四 stem 目标会互相冲突。

# Qwen3-TTS 结构与张量流对照

## 来源

- 论文：[../../papers/Qwen3-TTS_Technical_Report_2601.15621.pdf](../../papers/Qwen3-TTS_Technical_Report_2601.15621.pdf)
- 官方代码：[../../third_party/Qwen3-TTS](../../third_party/Qwen3-TTS)
- 核心模型：[modeling_qwen3_tts.py](../../third_party/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py)
- 配置：[configuration_qwen3_tts.py](../../third_party/Qwen3-TTS/qwen_tts/core/models/configuration_qwen3_tts.py)
- Finetune 数据：[dataset.py](../../third_party/Qwen3-TTS/finetuning/dataset.py)
- Finetune 训练：[sft_12hz.py](../../third_party/Qwen3-TTS/finetuning/sft_12hz.py)

## 一句话总结

Qwen3-TTS-12Hz 是一个 dual-track multi-codebook speech LM：主 Talker 沿时间维 autoregressive 预测第 0 个 codec codebook，CodePredictor/MTP 沿同一帧 codebook 维 autoregressive 预测剩余 codebooks。Finetune 代码中主干和 MTP 联合训练，loss 是 `q0_loss + 0.3 * sub_talker_loss`。

## Embedding 初始化

Qwen3-TTS 的 `Qwen3TTSPreTrainedModel._init_weights` 使用 `initializer_range`，默认 `0.02`：

```text
Linear / Conv weight: Normal(0, initializer_range)
Embedding weight:    Normal(0, initializer_range)
padding_idx row:     zero
LayerNorm/RMSNorm:   weight = 1
```

对应代码位置：[modeling_qwen3_tts.py:479](../../third_party/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py#L479)。

这和我们当前 separator 代码不同：我们现在没有自定义初始化，所以 `nn.Embedding` 走 PyTorch 默认 `Normal(0,1)`。如果对齐 Qwen/LLM 风格，建议我们改成显式 `Normal(0,0.02)` 或 `Normal(0,d_model^-0.5)`。

## 关键符号

| 符号 | 含义 | Qwen3-TTS-12Hz 实际/代码值 |
|---|---|---|
| `B` | batch size | 训练参数决定 |
| `T_text` | text token 长度 | 样本决定 |
| `T_codec` | codec frame 长度 | 12.5 Hz token frame，样本决定 |
| `C` | codec codebook/group 数 | 12Hz finetune 代码按 `16` 使用 |
| `D_text` | text embedding hidden | 代码默认 `2048` |
| `D` | Talker hidden | 代码默认 `1024` |
| `D_mtp` | CodePredictor hidden | 代码默认 `1024` |
| `V_talker` | Talker codec/text mixed vocab | checkpoint config 决定 |
| `V_mtp` | CodePredictor codec vocab | 代码默认 `2048`，checkpoint 可覆盖 |

注意：`configuration_qwen3_tts.py` 里的 `num_code_groups` 默认是 `32`，但官方 12Hz finetune 数据和训练代码固定使用 `audio_codes [T,16]`，因此下面按 12Hz 的 `C=16` 描述。

## 总体 Pipeline

```text
1. 数据准备
   text -> Qwen text tokenizer
   audio -> Qwen3-TTS-Tokenizer-12Hz
   -> text_ids    [1, T_text]
   -> audio_codes [T_codec, 16]
   -> ref_mel     [1, T_mel, 128]

2. collate 成 dual-track 序列
   input_ids [B, T_total, 2]
     channel 0: text channel
     channel 1: codec channel, mainly q0 and special codec tokens
   codec_ids [B, T_total, 16]
   codec_0_labels [B, T_total]
   codec_mask [B, T_total]
   attention_mask [B, T_total]

3. speaker encoder
   ref_mel [B, T_mel, 128]
   -> speaker_embedding [B, D] = [B, 1024] in default code

4. dual-track embedding sum
   text_embedding(input_ids[...,0])      -> [B,T_total,D_text]
   text_projection                       -> [B,T_total,D]
   codec_embedding(input_ids[...,1])     -> [B,T_total,D]
   high-codebook embeddings q1..q15      -> [B,T_total,D]
   sum -> input_embeddings [B,T_total,D]

5. 主 Talker 时间维 Transformer
   input_embeddings[:, :-1, :] + attention_mask[:, :-1]
   -> hidden_states [B,T_total-1,D]
   -> codec_head -> q0 logits [B,T_total-1,V_talker]

6. MTP / CodePredictor
   select codec positions:
     talker_hidden_states [N,D]
     talker_codec_ids     [N,16]
   build MTP sequence:
     [h_t, q0, q1, ..., q14] -> [N,16,D]
   CodePredictor Transformer
   -> logits for q1..q15 [N,15,V_mtp]

7. 联合 loss
   q0_loss = CE(q0 logits, codec_0_labels)
   sub_talker_loss = CE(MTP logits, codec_ids[:,1:])
   loss = q0_loss + 0.3 * sub_talker_loss
```

## 数据与 Collate

官方 finetune dataset 每条样本返回：

```text
text_ids:    [1, T_text]
audio_codes: [T_codec, 16]
ref_mel:     [1, T_mel, 128]
```

collate 后：

| 名称 | Shape | 含义 |
|---|---:|---|
| `input_ids` | `[B,T_total,2]` | dual-track token ids，0 通道 text，1 通道 codec |
| `codec_ids` | `[B,T_total,16]` | 完整 codec codebook stack |
| `text_embedding_mask` | `[B,T_total,1]` | text embedding 是否有效 |
| `codec_embedding_mask` | `[B,T_total,1]` | codec embedding 是否有效 |
| `codec_mask` | `[B,T_total]` | 哪些位置是真实 codec frame |
| `attention_mask` | `[B,T_total]` | 主 Talker attention mask |
| `codec_0_labels` | `[B,T_total]` | q0 监督；非监督位置为 `-100` |

结构来源：[dataset.py:153](../../third_party/Qwen3-TTS/finetuning/dataset.py#L153)。

## 主 Talker 输入 Embedding

Finetune 代码中，输入 embedding 是多个来源相加：

```text
input_text_ids = input_ids[:, :, 0]     # [B,T_total]
input_codec_ids = input_ids[:, :, 1]    # [B,T_total]

text_embedding:
  text_embedding(input_text_ids)        # [B,T_total,D_text]
  text_projection                       # [B,T_total,D]

codec q0/special embedding:
  codec_embedding(input_codec_ids)      # [B,T_total,D]

speaker slot:
  input_codec_embedding[:, 6, :] = speaker_embedding

high codebook embeddings:
  for i in 1..15:
    code_predictor.embedding[i-1](codec_ids[:,:,i]) # [B,T_total,D]
    mask by codec_mask

input_embeddings = text + codec + sum(high_codebook_embeddings)
input_embeddings: [B,T_total,D] = [B,T_total,1024] by default code
```

代码位置：[sft_12hz.py:86](../../third_party/Qwen3-TTS/finetuning/sft_12hz.py#L86)。

### 维度细节

| 模块 | 参数/输出 shape |
|---|---:|
| `talker.model.text_embedding` | `[text_vocab_size, D_text]` |
| `talker.text_projection` | `[B,T,D_text] -> [B,T,D]` |
| `talker.model.codec_embedding` | `[V_talker, D]` |
| `speaker_encoder(ref_mel)` | `[B,D]` |
| `code_predictor.get_input_embeddings()[i]` | `[V_mtp, D]` |
| final `input_embeddings` | `[B,T_total,D]` |

## 主 Talker

主 Talker 类是 `Qwen3TTSTalkerForConditionalGeneration`，内部包含：

```text
Qwen3TTSTalkerModel
  - codec_embedding
  - text_embedding
  - 20-layer causal Transformer by default code
  - RMSNorm

codec_head
  - Linear(D, V_talker)

code_predictor
  - MTP / sub-talker
```

默认代码配置：

```text
D = 1024
num_hidden_layers = 20
num_attention_heads = 16
num_key_value_heads = 2
intermediate_size = 2048
max_position_embeddings = 32768
sliding_window = 4096 if enabled
```

主 forward：

```text
inputs_embeds:  [B,T,D]
attention_mask: [B,T]
-> hidden_states [B,T,D]
-> codec_head(hidden_states) [B,T,V_talker]
```

代码位置：[modeling_qwen3_tts.py:1564](../../third_party/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py#L1564)。

## MTP / CodePredictor

CodePredictor 是独立小 Transformer：

```text
Qwen3TTSTalkerCodePredictorModel
  - codec_embedding ModuleList, one table per residual codebook
  - 5-layer Transformer by default code
  - RMSNorm

Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
  - lm_head ModuleList, one Linear per residual codebook
```

默认代码配置：

```text
D_mtp = 1024
num_hidden_layers = 5
num_attention_heads = 16
num_key_value_heads = 8
intermediate_size = 3072
vocab_size = 2048 by default code
```

代码位置：[configuration_qwen3_tts.py:187](../../third_party/Qwen3-TTS/qwen_tts/core/models/configuration_qwen3_tts.py#L187)。

### MTP 训练输入

在 `forward_sub_talker_finetune` 中，输入是：

```text
talker_hidden_states: [N,D]
codec_ids:            [N,16]
```

其中 `N` 是 batch 内所有 codec frame 的总数：

```text
N = sum(codec_mask)
```

构造 MTP sequence：

```text
position 0: talker_hidden_states.unsqueeze(1)  # [N,1,D]
position 1: embedding(q0)                      # [N,1,D]
position 2: embedding(q1)                      # [N,1,D]
...
position 15: embedding(q14)                    # [N,1,D]

sub_talker_inputs_embeds: [N,16,D]
labels: codec_ids[:,1:] = q1..q15              # [N,15]
```

CodePredictor 输出：

```text
hidden_states: [N,16,D_mtp]
lm_head[i-1](hidden_states[:, i]) for i=1..15
logits: [N,15,V_mtp]
```

代码位置：
- 构造输入：[modeling_qwen3_tts.py:1619](../../third_party/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py#L1619)
- 输出 logits：[modeling_qwen3_tts.py:1235](../../third_party/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py#L1235)

### MTP 语义

```text
h_t + q0         -> predict q1
h_t + q0,q1      -> predict q2
...
h_t + q0..q14    -> predict q15
```

这和我们当前 separator 的 `ResidualMTPPredictor` 是同一类结构，只是 Qwen3-TTS 使用 `C=16`，我们使用 EnCodec `K=8`。

## Loss

Finetune 代码中：

```text
q0_loss = outputs.loss
sub_talker_loss = CE(q1..q15)
loss = q0_loss + 0.3 * sub_talker_loss
```

代码位置：[sft_12hz.py:111](../../third_party/Qwen3-TTS/finetuning/sft_12hz.py#L111)。

梯度路径：

```text
q0_loss
  -> codec_head
  -> main Talker

sub_talker_loss
  -> CodePredictor / MTP
  -> talker_hidden_states
  -> main Talker
```

公开 finetune 代码没有对 `talker_hidden_states` 做 detach，所以 MTP loss 会回传到主 Talker。

## 推理流程

主 Talker 每个时间步先生成 q0：

```text
past_hidden: [B,1,D]
input_ids: q0 [B,1]
```

然后 CodePredictor 生成 q1..q15：

```text
inputs_embeds = concat(past_hidden, embedding(q0))  # [B,2,D]
max_new_tokens = num_code_groups - 1                # 15
predictor_result.sequences -> [B,15]
codec_ids = concat(q0, q1..q15)                     # [B,16]
```

再把完整 codebook stack 的 embedding 求和，作为当前时间步的 codec hidden 喂回主 Talker：

```text
last_id_hidden = embedding(q0)                      # [B,1,D]
q1..q15 embeddings                                  # each [B,1,D]
codec_hiddens = concat along codebook dim           # [B,16,D]
inputs_embeds = codec_hiddens.sum(1, keepdim=True)  # [B,1,D]
```

代码位置：[modeling_qwen3_tts.py:1670](../../third_party/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py#L1670)。

## 和当前 separator 的逐项对照

| Qwen3-TTS-12Hz | 当前 separator |
|---|---|
| TTS text/speaker/language 条件 | mixture RVQ + stem_id 条件 |
| audio_codes `[T,16]` | EnCodec codes `[T,8]` |
| 主 Talker 预测 q0 | 主 Transformer 预测 q0 |
| CodePredictor/MTP 预测 q1..q15 | ResidualMTPPredictor 预测 q1..q7 |
| q0 loss + `0.3 * sub_talker_loss` | coarse loss + `0.5 * residual_loss` |
| `initializer_range=0.02` 初始化 | 当前 separator 仍是 PyTorch 默认 embedding 初始化 |
| 推理使用 cache / HF generate | 当前 separator 逻辑流式但重算 prefix |

## 对我们模型的直接建议

1. Embedding 初始化建议改成 Qwen 风格：

```text
Embedding / Linear: Normal(0, 0.02)
RMSNorm weight: 1
Linear bias: 0
```

2. MTP 结构方向已经对齐 Qwen3-TTS：

```text
main hidden + q0..qK-2 -> q1..qK-1
```

3. 如果继续对齐 Qwen3-TTS，下一步更值得做的是推理缓存，而不是再改 MTP 训练方式：

```text
main Transformer KV-cache
MTP codebook-step cache or cheap per-frame recompute
```


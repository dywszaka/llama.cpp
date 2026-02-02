# llama-server 解码逻辑梳理（源码导向）

面向准备开发的同学，聚焦 **llama-server 的解码（decode）流程**、模块分层与职责、关键参数。仅讨论通用逻辑，不涉及 CPU/CUDA 等后端差异。

## 1. 代码入口与模块分层

### 1.1 主要入口与路由
- `tools/server/server.cpp`
  - `main()`：启动 HTTP server、加载模型、注册任务回调、进入主循环。
  - HTTP 路由与 JSON 解析：`handle_*` 系列（completion/chat/embeddings/rerank 等）。

### 1.2 任务与调度层
- `server_queue`（`tools/server/server.cpp`）
  - 负责任务队列、defer 机制、主循环 `start_loop()`。
  - 通过回调把“新任务”和“更新槽位（slots）”分发给 `server_context`。

- `server_response`（`tools/server/server.cpp`）
  - 结果队列与等待逻辑（同步或流式）。
  - 支持按 task id 拉取结果、错误处理、取消请求。

### 1.3 执行上下文与槽位
- `server_context`（`tools/server/server.cpp`）
  - 持有模型与上下文（`llama_model`, `llama_context`）、批处理（`llama_batch`）、slots、metrics。
  - 关键逻辑：`process_single_task()`、`update_slots()`。

- `server_slot`（`tools/server/server.cpp`）
  - 每个并发请求使用一个 slot。
  - 管理状态机、KV 缓存索引、采样器、生成统计、缓存 tokens。

### 1.4 输入与采样
- `tools/server/utils.hpp`
  - `server_tokens`：封装 prompt tokens/多模态 chunk，支持缓存与 LCS 复用。
  - `tokenize_input_prompts()`：把 JSON prompt 解析为 tokens（支持 mixed / 多 prompt）。

- `common/sampling.h`
  - `common_sampler_*`：统一采样链（top-k/top-p/grammar/penalties…）。

### 1.5 关键基础结构
- `common/common.h`
  - `common_params`：全局参数（n_ctx/n_batch/n_parallel/ctx_shift/…）。
  - `common_params_sampling` / `common_params_speculative`。

- `llama.h` / `src/llama.cpp`
  - `llama_decode()`、KV 缓存、batch 机制。


## 2. 核心对象与状态机

### 2.1 Slot 状态机（`slot_state`）
```
IDLE
  -> STARTED (接到任务并初始化)
  -> PROCESSING_PROMPT (prompt 正在入 batch)
  -> DONE_PROMPT (prompt eval 完成)
  -> GENERATING (开始采样生成)
  -> IDLE (完成/被取消)
```

### 2.2 任务类型（`server_task_type`）
- Completion / Infill / Embedding / Rerank：会占用 slot。
- Cancel / Metrics / Slot save/restore/erase / Set Lora：不进入 decode 流或只改变状态。


## 3. 解码主流程（从请求到 token）

下面是 **completion/chat** 的通用路径（infill 类似，embeddings/rerank 只 eval prompt）。

### 3.1 HTTP 层到任务队列
1. 请求进入 `handle_completions_impl()`。
2. 解析 JSON：
   - prompt → `tokenize_input_prompts()` → `server_tokens`。
   - 采样参数 → `server_task::params_from_json_cmpl()`。
3. 每个 prompt 生成一个 `server_task`，写入 `server_queue`。
4. `server_response` 记录 task id，用于阻塞 / 流式拉取。

### 3.2 任务调度（`server_queue.start_loop()`）
循环执行：
1. 弹出队列任务 → `server_context.process_single_task()`。
2. 任务处理完后 → `server_context.update_slots()`。
3. 阻塞等待新任务。

### 3.3 分配 Slot（`process_single_task()`）
- 根据 `id_slot` 或 `get_available_slot()` 选择 slot：
  - 优先使用 prompt 相似度（LCS）命中缓存（`slot_prompt_similarity`）。
  - 否则 LRU。
- slot 进入 `SLOT_STATE_STARTED`，初始化 sampler 与参数。
- 如果 slot 忙：任务被 defer。

### 3.4 update_slots：一次“批处理循环”
**这是 decode 逻辑的核心：**

#### (1) 全部 idle 时清理
- 若所有 slot idle 且 `clean_kv_cache`，清空 KV cache。

#### (2) Context shift（如果需要）
- 当 `n_past + 1 >= n_ctx`：
  - 若 `ctx_shift=false` → 直接停止并报错。
  - 否则：保留 `n_keep`、丢弃 `n_discard`，KV cache 通过 `llama_memory_seq_add/rm` 右移。

#### (3) 构建 batch
- `common_batch_clear(batch)`。
- **优先加入正在生成的 slot 的“已采样 token”**（保证继续生成）。
- **再加入正在处理 prompt 的 slot 的 prompt tokens**：
  - 支持 prompt truncation / cache reuse / chunk reuse（`n_cache_reuse`）。
  - 如果 prompt 处理完：标记 `SLOT_STATE_DONE_PROMPT`，设置 logits 只取最后一个 token。

#### (4) 调用 llama_decode
- 可能分块执行（`n_batch`），每块调用一次 `llama_decode()`。
- 返回错误时：尝试减小 batch size 重试。

#### (5) 解析结果 & 采样
- 对于 `DONE_PROMPT`：
  - Embedding / Rerank：直接产生结果并释放 slot。
  - Completion：转为 `GENERATING`。
- 对 `GENERATING`：
  - `common_sampler_sample()` 采样一个 token。
  - `common_sampler_accept()` 更新采样状态。
  - `process_token()` 处理 stop 规则、indent、time limit、ctx limit。
  - 发送 partial（stream）或 final（非 stream）。

#### (6) 可选的 speculative decoding
- 若启用 draft model，slot 可以进入 speculative：
  - 生成 draft tokens → batch decode → sampler 比对接受。
  - 被接受 token 也走 `process_token()`。

### 3.5 llama_decode 过程细节（拆解一次循环）
以下拆解对应 `server_context.update_slots()` 内部，对 **一次 batch decode** 的精细流程：

#### (A) 构造 llama_batch（输入级别）
- 每个 batch token 都包含：
  - `token`：要喂给模型的 token。
  - `pos`：KV cache position（对应 `n_past`）。
  - `seq_id`：slot id（区分不同并发序列）。
  - `logits`：是否需要返回该 token 的 logits。
- 规则（核心区别）：
  - **生成阶段**（`SLOT_STATE_GENERATING`）：加入“上一步采样出的 token”（`slot.sampled`）。这些 token 的 `logits=true`，用于得到下一步采样的分布。
  - **prefill 阶段**（`SLOT_STATE_PROCESSING_PROMPT` / `STARTED`）：加入 prompt tokens。只对 **prompt 的最后一个 token** 设置 `logits=true`（避免计算无用 logits）。

#### (B) llama_decode 执行
- `llama_decode(ctx, batch_view)` 对 batch 中所有 token 执行前向：
  - 更新 KV cache（pos/seq_id 驱动）。
  - 仅对 `logits=true` 的 token 生成 logits 缓存。
- 若返回错误（KV 空间不足等），会缩小 `n_batch` 重试。

#### (C) 结果消费（采样）
- 对于 `DONE_PROMPT` 的 slot：
  - 如果是 completion：转为 `GENERATING`，使用 prompt 的最后一个 token 的 logits 进行**首次采样**。
  - 如果是 embedding / rerank：直接读 embedding 输出并结束。
- 对于 `GENERATING` 的 slot：
  - 使用 `common_sampler_sample()` 从 logits 采样 token。
  - `common_sampler_accept()` 更新采样器状态。
  - `process_token()` 判断 stop/ctx/indent/time 等条件。

### 3.6 如何区分 prefill 与 generate？
在 llama.cpp 中 **没有单独的函数区分**，它们都通过 `llama_decode()` 完成。区分点来自 **batch 构造和 slot 状态**：

**判定维度 1：slot 状态机**
- `SLOT_STATE_PROCESSING_PROMPT` / `SLOT_STATE_STARTED`：属于 **prefill**。
- `SLOT_STATE_GENERATING`：属于 **generate**。

**判定维度 2：token 来源**
- **prefill**：batch 的 token 来源于 `prompt_tokens`。
- **generate**：batch 的 token 来源于 `slot.sampled`（上一次采样出来的 token）。

**判定维度 3：logits 规则**
- **prefill**：只对 prompt 的最后一个 token 设置 `logits=true`。
- **generate**：每一步生成 token 都设置 `logits=true`，用于下一步采样。

**判定维度 4：统计字段**
- `n_prompt_tokens_processed` / `t_prompt_processing` 只在 prefill 阶段增长。
- `n_decoded` / `t_token_generation` 只在 generate 阶段增长。

简化理解：
- **prefill = prompt eval**（把输入 tokens 喂入模型，构建 KV cache）。
- **generate = iterative decode**（每轮用 logits 采样新 token，再喂回模型）。

### 3.7 结果返回
- `server_response.recv()`（阻塞）或 SSE 流式返回。
- partial results 通过 `send_partial_response()`。
- final results 通过 `send_final_response()`。


## 4. 关键参数（开发常改）

### 4.1 全局参数（`common_params`）
文件：`common/common.h`
- **并发 / batch**
  - `n_parallel`: slot 数量（并发请求数上限）。
  - `n_batch`: 逻辑 batch size，影响 prompt 处理效率。
  - `n_ubatch`: 物理 batch size（必须 >=32）。
  - `cont_batching`: 是否允许插入新序列。
- **上下文 / KV**
  - `n_ctx`: context size。
  - `ctx_shift`: 超出上下文时是否 shift。
  - `n_cache_reuse`: KV cache 复用的最小 chunk size。
  - `n_swa_checkpoints`: SWA checkpoint 数量。
- **生成长度**
  - `n_predict`: 默认最大生成 token 数。
- **server 参数**
  - `port`, `hostname`, `api_prefix`。
  - `timeout_read/timeout_write`。

### 4.2 per-request 参数（`slot_params`）
文件：`tools/server/server.cpp`
- **生成控制**
  - `n_predict`, `n_keep`, `n_discard`。
  - `stream`：是否流式返回。
  - `cache_prompt`：是否启用 prompt cache。
  - `t_max_predict_ms`：生成超时（按新行触发）。
  - `n_indent`：缩进规则。
- **采样参数**（映射到 `common_params_sampling`）
  - `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`。
  - `repeat_last_n`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`。
  - `mirostat*`, `dry_*`。
  - `logit_bias`, `grammar`, `grammar_triggers`。
- **停止条件**
  - `antiprompt` / `stop`（字符串）。
  - `ignore_eos`。
- **Speculative**
  - `speculative.n_min / n_max / p_min`。

### 4.3 slot 选择与复用
- `slot_prompt_similarity`：LCS 相似度阈值（用于复用 slot KV cache）。
- `cache_prompt`: 是否允许复用 cache。


## 5. 数据流示意（简化）

```
HTTP Request
   -> handle_completions_impl()
   -> server_task(s)
   -> server_queue
   -> process_single_task()
      -> slot (STARTED)
   -> update_slots()
      -> build batch (prompt + sampled tokens)
      -> llama_decode()
      -> sample next token
      -> process_token()
      -> send_partial/final
```


## 6. 开发关注点

- **decode loop 入口**：`server_context.update_slots()`。
- **stop 逻辑**：`process_token()`（stop word、ctx limit、budget、indent、time）。
- **KV 复用**：`slot.cache_tokens` + `n_cache_reuse` + `ctx_shift`。
- **sampling**：`common_sampler_*`。
- **streaming**：partial 通过 `server_task_result_cmpl_partial` 发送。
- **speculative**：draft model 只在通用层可启用（非多模态）。


## 7. 建议的阅读顺序
1. `tools/server/server.cpp`：`process_single_task()` → `update_slots()` → `process_token()`。
2. `tools/server/utils.hpp`：`server_tokens` 与 prompt tokenization。
3. `common/common.h`：参数定义。
4. `common/sampling.h`：采样链与 grammar。
5. `llama.h` / `src/llama.cpp`：`llama_decode` 与 KV cache 实现。

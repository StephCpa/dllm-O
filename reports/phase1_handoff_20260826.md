# Phase 1 研究交接说明（2026-08-26 快照）

## 1. 当前研究问题与范围

本阶段研究考察 masked-diffusion language model 中不同 token commitment schedule 是否改变数学推理的高阶覆盖，以及一个不依赖正确性标签的 Boundary Closure Index（BCI）能否在未见 query 上预测 AR 相对 arbitrary-order decoding 的 CoverageAUC 优势。

本阶段只检验 decoder schedule、局部 uncertainty closure 与最终正确解覆盖之间的关系。它目前不检验 differential privacy，不建立 causal fork，不训练模型，也不允许把 entropy reduction 一般化为有害机制。

独立统计单位始终是 query。Rollout、token 和 denoising step 都是 query 内重复测量，不能当作独立样本扩大显著性。

## 2. 已完成工作

### Phase 0：机制校准

- 完成 LLaDA-8B-Instruct 的 trace-equivalence smoke test 与 GSM8K signal experiment。
- 完成 strict AR (`ar_b1`) 与 confidence-based arbitrary order (`ao_b32`) 的多 rollout 对比和 token-level trace 记录。
- Phase 0 显示 delayed commitment 与更强 entropy closure 同时出现，并提示 AR 与 AO 可能在 one-shot accuracy 和 high-k coverage 之间存在差异。
- Phase 0 的全部结果已被重新归类为 calibration evidence，不能作为 Phase 1 held-out 证据。

### Phase 1 冻结设计与实现

- 冻结模型：`GSAI-ML/LLaDA-8B-Instruct`，revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`。
- 冻结外部 grader/code：JustGRPO commit `1a2fddb5c6655597e63081c0af5ebb718a849f39`。
- 冻结任务：GSM8K 与 MATH-500；每个任务 50 个 calibration query、100 个 held-out query，split 与 prompt hash 均写入 manifest。
- 冻结五个 schedule：`ar_b1`、`ao_b8`、`ao_b16`、`ao_b32`、`random_b32`。
- 完成可恢复、原子写入、trace hash 校验、split/freeze/commit 门禁和 shard completion manifest 的 Phase 1 runner。
- 当前 runner 和 freeze 所绑定的 Git commit 是 `eaf4b4f9214eec6066010fad7dba5c5319d85122`。
- 本地测试为 `50 passed`。

### Phase 1A：calibration inference

- GSM8K 复用 Phase 0 中 `ar_b1`/`ao_b32` 的前 16 个 rollout，并补齐 `ao_b8`/`ao_b16`/`random_b32`。
- MATH-500 完成 50 queries × 5 policies × 16 rollouts。
- Phase 1A 共完成 6,400 个新 generation；加上 GSM8K 复用的 1,600 个 primary-policy generation，freeze fitting 使用的审计体系已完整闭合。
- 所有 calibration shard completion markers、result JSON 和 trace artifacts 均已完成并校验。

### BCI calibration、匹配修复与最终 freeze

- Candidate score 冻结为任务内稳健标准化的“高 eligible entropy + 低 eligible logit margin”等权分数。
- Candidate threshold 为各任务 calibration token 的第 75 百分位；规则不读取正确性或最终答案。
- 最终 matched control 限定在同一 task/query/rollout/block 内，并同时匹配 block position、commitment rank、eligible entropy 和 negative-log eligible margin。
- 第一个 freeze 因未平衡 entropy/margin、导致 closure endpoint 存在机械耦合而在 held-out 前作废。
- 第二个 freeze 保留平衡匹配，但缺少 zero-match query 和最低覆盖率规则，也在 held-out 前作废。
- 两个作废 SHA 及原因均写入最终 freeze 的 revision history，不能删除或隐藏。
- 最终 freeze SHA-256：`8f2b6df3f98abdb0b9060b704b02fcc428090d01d2d279514c7c108e2fb6ef17`。
- 最终 calibration 中共有 409,600 个 eligible tokens、102,400 个 candidates 和 13,558 个 matched pairs。
- 匹配后 candidate-minus-control entropy closure 为 `+0.1603`；GSM8K 为 `+0.2077`，MATH-500 为 `+0.1193`。
- Margin closure 是 secondary endpoint，整体为 `-0.0692`，不能省略或改写为支持性结果。
- 匹配覆盖 GSM8K `50/50` queries、MATH-500 `49/50` queries；每任务 held-out 匹配覆盖低于 90% 时，必须把该任务 RQ1 判为 unresolved，不得事后放宽 caliper。

### Calibration 中已经出现的不利信号

- 16-rollout ScreeningCoverageAUC proxy 有 60% query 的 AR-minus-AO advantage 恰为零，目标较稀疏且噪声较大。
- BCI 与 calibration target 的 pooled Spearman 为 `-0.1969`，不是预期的正方向。
- Baseline selected-CV MAE 为 `0.08959`，加入 BCI 后为 `0.09033`，校准阶段没有显示增量预测收益。
- 这些结果已完整记录，BCI 公式没有按相关性方向重新选择。后续 held-out 是证伪性测试，不应预设成功。

## 3. 正在运行的实验

### Phase 1B held-out screening，shard 0/8

- 当前只启动了 `shard 0-of-8`，运行在 GPU 0；GPU 5 明确保留给其他任务，不能使用。
- 快照时间：服务器时间 2026-08-27 05:43 CST。
- 快照进度：`1189/2000`，即 59.45%；result 与 trace 数量均为 1189，另有一个正常写入中的 `.tmp` trace。
- 当前没有 Traceback、OOM、RuntimeError 或 parser-level launch failure。
- 当前只做健康度与进度检查，尚未汇总或查看 held-out correctness、Pass@k、BCI association 或任何主要结果。
- 日志：`dllm_order_transmission/logs/phase1b_heldout_screening_shard0_gpu0.log`。
- 输出：`dllm_order_transmission/outputs/dllm_order_phase1/phase1-v1/heldout/screening/`。
- 预计 shard 0 仍需约 7 小时，但外部 GPU 进程可能改变吞吐或引发 OOM。

### 当前 GPU 风险

- GPU 0 上我们的 runner 当前约占 23.3 GiB device memory。
- 同卡还有 embedding service、Ollama 和后来启动的外部 JEPA 任务；快照时 GPU 总剩余显存约 4.6 GiB。
- 不得停止或修改其他合作者的进程。若发生 OOM，应先暂停并协调释放资源，再用完全相同的 shard 命令恢复；不得减少 generation length、改 batch 语义或替换模型。

## 4. 尚未完成但协议要求必须执行的实验

### 完成 held-out screening

- `shard 0/8` 完成后还需执行 shards `1..7`。
- Held-out screening 总规模为 16,000 generations：200 queries × 5 policies × 16 rollouts。
- 每个 8-way shard 恰有 2,000 generations；不得删除不利 policy、跳过 MATH-500 或按前 16 个结果停止。
- 所有 shard 必须使用同一个最终 freeze、split manifest、model revision、code commit 和 seed derivation。

### Primary extension（强制）

- Screening 完成后，`ar_b1` 和 `ao_b32` 必须从 16 rollouts 扩展到 64 rollouts。
- 新增规模为 200 queries × 2 policies × 48 rollouts = 19,200 generations。
- Extension 不能依据 screening 是否有利而取消；runner 会要求前 16 个 primary results 已存在且 hash/schema 合法。
- 正式 CoverageAUC 使用 `Pass@4/8/16/32/64` 的均值，不能用 calibration 的三点 proxy 替代。

### Held-out analyzer 与 Gate 1 报告

- 在读取 held-out correctness 统计之前，应完成并测试 held-out analyzer；它必须直接消费最终 freeze，而不是重新拟合。
- RQ1：按 query 聚合 candidate-minus-control entropy closure，并按任务报告 query-clustered uncertainty。
- RQ2：报告 task-stratified BCI 与 `ARCoverageAdvantage` 的 held-out Spearman bootstrap interval。
- Predictive value：使用 calibration-fitted baseline/augmented coefficients，报告 `MAE_baseline - MAE_augmented` 的 paired query bootstrap interval；禁止 held-out refit。
- Secondary：完整报告五个 schedule、block-size dose response、random control、margin closure、parser failure 和 malformed-output sensitivity。
- Gate 1 只有在 closure 条件和 BCI predictive-value 条件同时满足时才通过。

## 5. Gate 1 之后的条件性实验

### 如果 Gate 1 通过

- 进入 Phase 2 forced-token intervention，验证高 closure candidate 是否真正改变后续路径和最终答案。
- 选择约 50 个 query，每个 query 最多三个 candidate positions；比较强制 top-1/top-2 token 后的答案/路径 divergence。
- Controls 必须匹配初始 entropy、position、token frequency 等，不能再次用未平衡 control。
- 只有 intervention 显示高-closure candidate 比 controls 产生更大 downstream divergence，才可称为 causal fork。

### 如果 Gate 1 失败

- 不得重新选择 k、改用 per-rollout accuracy、删除 MATH-500、改变 candidate rule 或利用 lexical Phase 0 结果“救回”假设。
- 如果 closure 为正但 BCI 不预测 coverage，应将项目收缩为 dLLM decoder diagnostic study。
- 如果 candidate-control closure 也失败，应停止 causal intervention 和 private aggregation 路线。

### Phase 3 private aggregation（只有前置 gate 支持时）

- 使用固定标签域任务，如 MMLU-Pro、ARC-Challenge 或经许可审计后的 GPQA。
- GSM8K/MATH-500 是 mechanism tasks，不应直接包装成 fixed-domain DP benchmark。
- 研究层级应包括 partition、demonstration order、decoder randomness 与 private release channel。
- 任何跨 epsilon 的重放只能称为 counterfactual mechanism analysis，不能称为多个免费的真实部署。
- 第二个 dLLM checkpoint 也只在 Gate 1 后考虑，不能在当前阶段用额外模型搜索有利结果。

## 6. 值得补充的理论工作

- 明确区分“candidate boundary token”“高 closure token”和“causal fork”；Phase 1 只能使用前两个术语。
- 形式化 schedule-induced local closure 到 query-level solution coverage 之间需要哪些额外假设，并解释局部 closure 为何可能无法预测全局正确答案多样性。
- 讨论 16-rollout calibration proxy 的离散化与 measurement error；这可解释 statistical attenuation，但不能被当作改 endpoint 或放宽 Gate 1 的理由。
- 将 BCI 定位为 task/model/schedule-specific diagnostic，而不是 universal scalar；只有跨 held-out task 和后续 intervention 支持后才能提高理论强度。
- 若 held-out 结果为负，应把“local uncertainty closure 与 global coverage 可解耦”作为理论结论候选，而不是继续假设 BCI 必然有效。
- Private aggregation 理论应在 Gate 1/2 后再连接，避免把 decoder-level结果提前解释为 DP amplification、masking 或 privacy-utility law。

## 7. 合作者接手时的硬性注意事项

- 不要读取或汇总 held-out correctness，直到 held-out analyzer、bootstrap seed、missingness 和输出表结构全部冻结并经过测试。
- 不要修改最终 freeze。唯一允许使用的 SHA 是 `8f2b6df3f98abdb0b9060b704b02fcc428090d01d2d279514c7c108e2fb6ef17`。
- 不要使用两个历史 freeze；它们只作为审计链保存。
- Runner 仓库必须保持 clean，并与 freeze 中的 commit `eaf4b4f...` 完全一致；提交新分析代码后不要让 inference runner 跟随到新 HEAD。
- 不要改变 query、rollout count、policy、seed、parser、generation length、temperature、block size 或模型 checkpoint。
- Parser failure、truncation、timeout 和 degenerate output 必须保留在 intention-to-decode denominator；任一 task-policy malformed rate 超过 1% 时补 valid-output sensitivity。
- 不要把 token 数量当样本量；所有主要 uncertainty 都按 query 聚类或 query bootstrap。
- Resume 时使用相同 shard 命令，已完成 JSON/trace 会被校验后跳过。若进程异常死亡留下 `.tmp`，先确认 PID 已不存在、对应 final JSON 不存在，再隔离 stale temp；不要在运行中删除 `.tmp`。
- Outputs 与 traces 在 `.gitignore` 中，Git push 不会备份实验结果；应单独做 checksummed storage backup。
- 不要提交服务器路径、用户名、主机、密码、私有缓存路径或进程信息到公开 artifact。
- GPU 5 不得用于本项目。GPU 0 当前存在明显外部显存竞争，新增 shard 前必须重新检查每卡的显存、利用率和进程归属。
- 不要混合提交论文工作区的无关修改。代码提交仅应包含 `dllm_order_transmission` 下本任务相关文件。

## 8. 关键文件

- 冻结协议：`dllm_order_transmission/protocols/phase1_protocol_v1_20260823.md`
- Split manifest：`dllm_order_transmission/protocols/phase1_split_manifest_v1.json`
- Artifact schema：`dllm_order_transmission/protocols/phase1_artifact_schema_v1.json`
- Phase 1 runner：`dllm_order_transmission/src/dllm_order_transmission/phase1_runner.py`
- Calibration/freezer：`dllm_order_transmission/src/dllm_order_transmission/phase1_calibration.py`
- 最终 freeze：`dllm_order_transmission/outputs/dllm_order_phase1/phase1-v1/calibration-analysis/phase1_bci_freeze_v1.json`
- Calibration report：`dllm_order_transmission/outputs/dllm_order_phase1/phase1-v1/calibration-analysis/report.json`
- 总体研究方案：`dp_llm_research/analysis/order_transmission_dllm_dp_icl_research_plan_20260820.md`


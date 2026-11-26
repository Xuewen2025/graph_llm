# SAGE-Drive reproduction: conversation log

Conversation between the user and Claude Code (Claude Opus 5.5) while reproducing the SAGE-Drive paper.
Only messages are kept (tool calls and command outputs are omitted). API keys are redacted.

### **User**  <sub>2026-11-25 22:34 UTC</sub>

根据这个paper复现代码：/home/ubuntu/test/graph_llm/SAGE_Drive_IROS (1).pdf
不明确的你可以问我，你也可以修改和完善不合理的部分

---

### **Claude**  <sub>2026-11-25 22:34 UTC</sub>

I'll start by reading the paper.

---

### **Claude**  <sub>2026-11-25 22:34 UTC</sub>

Paper read. Let me check the environment before deciding on the design.

---

### **User (answers to Claude's questions)**  <sub>2026-11-25 22:37 UTC</sub>

The user answered: "There is no nuPlan data on this machine, and nuPlan needs a registered account to download. Which data should the code target?"="你换一个数据集，方便下载的", "The paper uses GPT-4o as the teacher to generate CoT reasoning labels (Eq. 13). No API keys are set here. Which teacher should I use?"="用openrouter的，先用便宜或者free的模型", "Should closed-loop evaluation (nuPlan simulator, Val14 NR/R scores) be in scope?"="Open-loop now, planner adapter stub (Recommended)". Read the answers carefully — they may request clarification, changes, or that you not proceed — and follow what they actually say.

---

### **Claude**  <sub>2026-11-25 22:39 UTC</sub>

4× A100 available. Writing the AV2 downloader now.

---

### **User**  <sub>2026-11-25 22:39 UTC</sub>

openrouter的key：sk-or-v1-<REDACTED>

---

### **Claude**  <sub>2026-11-25 22:39 UTC</sub>

Got the key. I'll keep it in a local `.env` file (permissions 600) and won't hardcode it in the code. Starting the full downloads in the background.

---

### **Claude**  <sub>2026-11-25 22:39 UTC</sub>

Meanwhile, I'm checking the AV2 file format and the available OpenRouter models.

---

### **Claude**  <sub>2026-11-25 22:44 UTC</sub>

Now preprocessing (multi-process) and the graph-predictor training script.

---

### **Claude**  <sub>2026-11-25 22:45 UTC</sub>

Train download is done (20k scenarios). Preprocessing train, then training the graph predictor in the background on GPU 0.

---

### **Claude**  <sub>2026-11-25 22:50 UTC</sub>

Fixing a bad assignment line in the training script, then writing the evaluation script.

---

### **Claude**  <sub>2026-11-25 22:52 UTC</sub>

Predictor finished: val ADE@3s 0.59 m and FDE@3s 1.08 m. Running future-graph inference and record building for both splits.

---

### **Claude**  <sub>2026-11-25 22:53 UTC</sub>

Starting teacher labeling for the 20k train samples in the background, with a $15 spending cap. Meanwhile I'll smoke-test LLM training on the variants that don't need labels.

---

### **Claude**  <sub>2026-11-25 22:57 UTC</sub>

At eval time no CoT labels are needed. I'll make the label file optional for non-train splits.

---

### **Claude**  <sub>2026-11-25 23:08 UTC</sub>

All 5 variants are training. While they run, I'm writing the inference planner wrapper, the nuPlan closed-loop adapter stub, the results-table script, and the README.

---

### **Claude**  <sub>2026-11-25 23:10 UTC</sub>

GPU 0 is at 80.8 GB with raw and full sharing it, so it could run out of memory. Checking the logs for errors.

---

### **Claude**  <sub>2026-11-25 23:11 UTC</sub>

The full model ran out of memory because it shares GPU 0 with the raw baseline. I'll add a gradient-checkpointing option, which trades a little speed for a large memory saving, and relaunch.

---

### **Claude**  <sub>2026-11-25 23:14 UTC</sub>

All five are running. Estimated times: raw ~1 h, hist ~1.4 h, fut ~2.3 h, cot ~2.6 h, full ~4 h+. Writing the README and a one-command pipeline script meanwhile.

---

### **Claude**  <sub>2026-11-25 23:15 UTC</sub>

The data setup is reasonable: median ego speed 5 m/s, about 17 agents and 3 crosswalks per scene. I'll wait for the raw baseline to finish and then evaluate it.

---

### **Claude**  <sub>2026-11-25 23:15 UTC</sub>

While waiting, the README with the paper-to-code mapping and the list of changes I made:

---

### **Claude**  <sub>2026-11-25 23:15 UTC</sub>

The README is written; I'll fill in results as each run finishes. Waiting for the raw baseline.

---

### **Claude**  <sub>2026-11-25 23:15 UTC</sub>

The full model is the slowest because of gradient checkpointing and the shared GPU. It should speed up once the raw baseline releases GPU 0. Waiting for the background notification.

---

### **User**  <sub>2026-11-25 23:42 UTC</sub>

现在进展到哪里了

---

### **Claude**  <sub>2026-11-26 00:14 UTC</sub>

No response requested.

---

### **User**  <sub>2026-11-26 00:14 UTC</sub>

现在是什么进度了

---

### **Claude**  <sub>2026-11-26 00:14 UTC</sub>

所有任务都在跑，上次会话结束没有中断训练。截至 00:14 (UTC) 的进度：

| 对照组 | 训练进度 | 当前文本损失 | 预计还要 |
|---|---|---|---|
| raw（基线，不用图） | 3000 / 3750 | 0.189 | 约 20 分钟 |
| hist（+历史/当前场景图） | 3275 / 3750 | 0.185 | 约 10 分钟 |
| fut（+预测的未来图） | 2075 / 3750 | 0.201 | 约 1 小时 |
| cot（+CoT 推理） | 1550 / 3750 | 0.42 | 约 1.5–2 小时 |
| full（完整模型） | 525 / 3750，还在第一阶段 | 0.775 | 约 4–5 小时 |

完整模型的轨迹回归损失从 16.7 降到 1.56，收敛正常。它最慢，因为开了梯度检查点，又和 raw 共用一张卡；raw 训完后会快一些。

刚加了一个后台脚本 [scripts/auto_eval.sh](scripts/auto_eval.sh)，不受会话断开影响：
- 每个对照组训练完，自动在 2000 个验证场景上评测，指标是 3 s 和 6 s 的 ADE/FDE、FVR 和碰撞率。
- raw 训完后还会额外评测一次没微调的 Qwen3-0.6B 作参考。
- 每完成一个就更新汇总表 `outputs/results.md`，格式同论文表 IV。

最先出结果的是 hist 和 raw，大约半小时后。全部跑完（包括完整模型评测）要 5 小时左右。你随时问我，我就读结果表整理给你。

---

### **User**  <sub>2026-11-26 00:16 UTC</sub>

你是不是用的四张卡跑，别浪费

---

### **Claude**  <sub>2026-11-26 00:16 UTC</sub>

对，现在 4 张卡都在用：GPU0 跑 raw 和 full，GPU1 跑 hist，GPU2 跑 fut，GPU3 跑 cot。

但你说得对，接下来会浪费。hist 和 raw 大约 10–20 分钟后训完，GPU0 和 GPU1 会空出来，而最慢的完整模型还是单卡加梯度检查点，要跑 4–5 小时。所以我给训练脚本加上多卡并行（DDP），等这两张卡空出来，就把完整模型改成双卡重跑，不再开梯度检查点。预计 1.5–2 小时就能跑完，比现在快一倍多。

---

### **Claude**  <sub>2026-11-26 00:16 UTC</sub>

DDP is added. Before switching over, a quick 2-GPU smoke test with a tiny batch, so the switch doesn't fail partway:

---

### **Claude**  <sub>2026-11-26 00:18 UTC</sub>

The switcher is waiting in the background. I'll wait for raw and hist to finish and then confirm the switch went through.

---

### **Claude**  <sub>2026-11-26 00:28 UTC</sub>

hist has finished training and its evaluation is running. raw has about 275 steps left (~10 min), after which the switch happens.

---

### **Claude**  <sub>2026-11-26 00:37 UTC</sub>

对，4 张卡都在用。完整模型已经从单卡改成双卡并行，速度快了大约 3 倍。

- 新增了多卡训练（DDP）：用 `torch.distributed.run` 启动就行，单卡用法不变。2 卡测试跑通了。
- raw 和 hist 训完后，后台脚本 [scripts/switch_full_ddp.sh](scripts/switch_full_ddp.sh) 自动停掉了原来单卡的完整模型（才到 625/3750 步），改成 GPU0+1 双卡重新训练，梯度检查点也关了。
- 速度从约 8 步/分钟提到约 23 步/分钟。GPU0 上 raw 的评测还在跑，评测完会再快一点。完整模型预计约 2.5 小时训完，原来要 6 小时以上。

现在四张卡的分工：

| GPU | 任务 |
|---|---|
| 0 + 1 | 完整模型双卡训练；raw 的评测也在 GPU0 上 |
| 2 | fut 训练，约 1 小时后完成，之后自动评测 |
| 3 | cot 训练，约 1.5 小时后完成，之后自动评测 |

**第一个结果：hist 对照组（+历史/当前场景图），2000 个验证场景**

| | ADE@3s | FDE@3s | ADE@6s | FDE@6s | FVR | 碰撞率 |
|---|---|---|---|---|---|---|
| hist | 0.547 | 1.32 | 2.03 | 5.30 | 8.7% | 6.1% |
| 匀速外推（参考） | 1.11 | 2.57 | 3.61 | 8.98 | 0% | 10.8% |

hist 输出的轨迹全部能解析，6 s 误差比匀速外推低了 44%。它的 FVR 是 8.7%，比真实轨迹本身的 11.0% 还低，因为阈值取的是真实轨迹的第 95 百分位。

还有一个问题：fut 和 cot 训完后，GPU2 和 GPU3 大约 1–1.5 小时后会空出来。完整模型已经在双卡上跑，中途没法再加卡。这两张卡可以用来跑论文表 I 的模型规模实验，即用 Qwen3-1.7B 或 4B 跑完整模型。要不要跑？

---

### **User**  <sub>2026-11-26 01:09 UTC</sub>

好，空出来的卡跑 Qwen3-1.7B 和 4B

---

### **Claude**  <sub>2026-11-26 01:09 UTC</sub>

好。表 I 是在每个规模上比较基线（raw，纯文本）和完整模型，所以要加 4 组：1.7B 和 4B 各跑一个 raw、一个 full。

算力上，1.7B 每步大约是 0.6B 的 3 倍，4B 大约 7 倍。如果照 0.6B 那样各跑 3 个 epoch，总共要 17 小时以上。所以规模实验改成每组 **1 个 epoch**：完整模型两阶段按 1/3 + 2/3 分配，同一规模下基线和完整模型的训练量相同。先预下载模型，再写一个排队脚本，哪张卡空了就接着跑下一个。

---

### **Claude**  <sub>2026-11-26 01:10 UTC</sub>

排队脚本 [scripts/scale_jobs.sh](scripts/scale_jobs.sh) 已经在后台跑了，Qwen3-1.7B 已下载完，4B 正在下载。各卡安排如下：

| GPU | 什么时候开始 | 依次跑 |
|---|---|---|
| 2 | fut 训完、评测完以后（约 1 小时） | 1.7B 基线 → 4B 基线 |
| 3 | cot 训完、评测完以后（约 1.5 小时） | 1.7B 完整模型 |
| 0+1 | 0.6B 完整模型训完、评测完以后（约 2.5 小时） | 4B 完整模型，双卡并行 |

每组训完会自动评测，并把结果加进 `outputs/results.md`，按论文表 I 的格式汇总。全部跑完预计要 6–7 小时。

**目前出了两个 0.6B 的结果，需要先说明一个情况：**

| 对照组 | ADE@3s | ADE@6s | FDE@6s | FVR | 碰撞率 |
|---|---|---|---|---|---|
| 未微调（zero-shot） | 11.19 | 16.38 | 26.21 | 21.2% | 36.3% |
| 基线：纯文本描述，有微调 | 0.544 | **1.998** | **5.16** | 8.5% | **5.2%** |
| +历史/当前场景图 | 0.547 | 2.028 | 5.30 | 8.7% | 6.1% |
| 匀速外推（参考） | 1.11 | 3.61 | 8.98 | 0% | 10.8% |

- **场景图版本目前并没有比基线好，两者基本持平。** 论文里加图后 ADE 从 1.22 降到 0.47，在这里还没有复现出来。
- 论文的基线 FVR 高达 99.85%，看起来是一个没训练好、甚至没微调的基线。我们这里的基线用同样的数据和 LoRA 认真微调过，本身就已经很强了。
- 我的推测是：在 AV2 上，6 s 的误差主要由自车自身的运动决定（历史轨迹加导航指令），换成哪种描述方式影响不大。这只是推测，还没验证。

真正要看的是后面三组：加预测的未来图、加 CoT 推理、加图 token 对齐的完整模型。如果到时候整体差距仍然很小，我建议再按交互强度把场景分组评测，比如前方近距离有车、匀速外推误差大的那些场景。场景图带来的提升应该主要出现在这类场景里，不会平均摊到所有场景上。

我会继续盯着，后面三组的结果一出来就告诉你。

---

### **User**  <sub>2026-11-26 04:19 UTC</sub>

现在有结果了吗

---

### **Claude**  <sub>2026-11-26 04:19 UTC</sub>

Most results are in. Since the full model has a 26.9% parse-failure rate, I'll check why, and also look at the 1.7B/4B progress.

---

### **Claude**  <sub>2026-11-26 04:20 UTC</sub>

Training targets always end with "4. Intent decision: ...". The fix: stop at the end of that line, insert [PLAN], then generate the trajectory.

---

### **Claude**  <sub>2026-11-26 04:22 UTC</sub>

After inserting [PLAN], the text output is still garbage. I'll look at what is generated right after [PLAN].

---

### **Claude**  <sub>2026-11-26 04:24 UTC</sub>

Parse failures are down from 27% to 0. For comparison, I'll compute the other variants on the same 64 scenes.

---

### **Claude**  <sub>2026-11-26 04:24 UTC</sub>

0.6B 的消融结果已经出齐，1.7B 和 4B 的基线也跑完了。完整模型这一组之前的数字作废：评测代码有个 bug，我已经修好，现在三个尺寸的完整模型正在用修好的代码评测。

**0.6B 消融（2000 个验证场景，ADE@6s 越低越好）**

| 对照组 | ADE@3s | ADE@6s | FDE@6s | FVR | 碰撞率 |
|---|---|---|---|---|---|
| 基线：纯文本描述，有微调 | 0.544 | 1.998 | 5.16 | 8.5% | 5.2% |
| + 历史/当前场景图 | 0.547 | 2.028 | 5.30 | 8.7% | 6.1% |
| **+ 预测的未来场景图** | **0.497** | **1.776** | **4.53** | 9.0% | **4.2%** |
| + CoT 推理 | 0.547 | 2.062 | 5.40 | 8.2% | 4.9% |
| 完整模型 | 作废，正在重测 | | | | |
| 匀速外推（参考） | 1.112 | 3.613 | 8.98 | 0% | 10.8% |

- **只有加预测的未来场景图这一步明显有效**：ADE@6s 降了 11%，FDE@6s 降了 12%，碰撞率从 5.2% 降到 4.2%。这正是论文的核心思路。
- 加历史图和加 CoT 都没有带来提升。

**完整模型出了什么问题**

训练时的序列顺序是：推理文字 → [PLAN] 向量 → 轨迹文本。[PLAN] 是我加的一个学习向量，不是词表里的 token，所以模型生成时自己写不出它。旧的推理代码直接让模型往下生成，跳过了 [PLAN]，和训练时的上下文对不上，结果有 27% 的输出写成了 "Answer:" 之类，轨迹解析不出来。

修复后的推理分三步：先生成推理文字，再插入 [PLAN]，然后固定写上 "Trajectory:" 前缀，让模型只续写数字。之所以要固定前缀：我用训练数据检查过，[PLAN] 之后的第一个词模型几乎总是预测错，因为那个位置同时被回归头占用了；而后面的数字基本都能预测对。在 64 个场景上试跑，解析失败率从 27% 降到 0。

**但修好以后，完整模型依然比基线差。** 同样 64 个场景上的 ADE@6s：

| 对照组 | ADE@6s |
|---|---|
| 基线 | 1.73 |
| + 未来图 | 1.59 |
| 完整模型（文本轨迹） | 2.36 |
| 完整模型（回归头轨迹） | 2.25 |

训练日志里也能看出来：完整模型的文本损失在 0.65 左右，CoT 那组是 0.36。我判断这是设计和超参的问题。第二阶段文本损失的权重只有 0.1，回归损失占了主导，LoRA 被拉去服务回归头，语言建模学得不够。

**模型规模（每组只训了 1 个 epoch）**
- 1.7B 基线 ADE@6s 2.116，4B 基线 2.035，都没有比训了 3 个 epoch 的 0.6B（1.998）更好。这一组主要受训练量限制。
- 1.7B 和 4B 的完整模型 4B 正在评测（已经用上修好的代码）；1.7B 等它旧代码那次评测跑完会重测。

**下一步，需要你拍板：** 三组完整模型的评测大约 1 小时内出结果。之后我想改两处，重训 0.6B 完整模型，双卡大约 2.5 小时：
- 把回归头读取的位置挪到轨迹文本之后，不再和生成第一个词抢同一个位置；
- 第二阶段文本损失的权重不再降到 0.1，保持 1.0。

这两处是对论文没写清楚部分的补全。要不要等评测完就开跑？

---

### **User**  <sub>2026-11-26 04:26 UTC</sub>

所以你觉得是为什么，是把image转为graph这步本身就不合理吗

---

### **Claude**  <sub>2026-11-26 04:26 UTC</sub>

我先用现有的预测结果做个分组统计，用数据回答你，不只靠推测。

---

### **Claude**  <sub>2026-11-26 04:27 UTC</sub>

不是。这里其实没有图像转成图这一步：AV2 给的直接就是结构化数据（每个物体的位置、速度、车道线坐标），论文用的 nuPlan 也一样。真正要问的是：把同一份结构化信息改写成图的形式，有没有用。

我按场景类型拆开看了 2000 个验证场景，数字是 ADE@6s，越低越好：

| 场景 | 数量 | 基线（纯文本） | + 历史图 | + 未来图 | + CoT |
|---|---|---|---|---|---|
| 全部 | 2000 | 1.998 | 2.028 | **1.776**（−11%） | 2.062 |
| 前方 25 m 内有车 | 600 | 1.752 | 1.813 | **1.445**（−18%） | 1.757 |
| 前方没车 | 1400 | 2.104 | 2.121 | **1.918**（−9%） | 2.192 |
| 匀速外推误差最大的 20%（变速、转弯多） | 400 | 3.224 | 3.251 | **2.819**（−13%） | 3.264 |

**我的结论**

1. **只改写表示形式没有用，加了新信息才有用。** 我们的基线文本已经列出了所有物体的坐标、速度、1 秒前的位置和车道线。历史图只是把这些重新写成"A 在 EGO 前方、距离多少"这样的关系三元组。信息量没变，微调过的 LLM 自己就能算出这些关系，所以没有提升。论文里加历史图后提升很大，我认为主要是因为它的基线太弱：FVR 高达 99.85%，像是没训练好。
2. **场景图真正起作用的是"预测未来"这部分。** 未来图来自单独训练的时空图预测器，它给出其他车 3 秒内会走到哪、会不会冲突，这是 LLM 从文本里自己推不出来的。前方有车的场景提升 18%，前方没车只有 9%。交互越多提升越大，说明图里的交互信息确实被用上了。
3. **CoT 没用，问题出在标签本身。** 推理标签是教师模型在看到真实轨迹之后写的"事后解释"。测试时模型得先自己生成这段推理，生成错了会带偏后面的轨迹，而这些推理也没有提供新信息。对 0.6B 这样的小模型，它更多是在占用容量。
4. **完整模型变差，是我这边的实现和超参问题，不能说明场景图不行。** 回归头和生成第一个词共用 [PLAN] 那个位置；第二阶段文本损失的权重又降到 0.1，语言建模学得不够。
5. **误差的大头在纵向。** 基线的纵向误差平均 1.85 m，横向只有 0.36 m，也就是说难点主要在什么时候加速、什么时候减速。能提供"前车接下来会怎样"的信息就有用，只描述当前空间关系的帮助有限。

所以论文的思路本身没问题，但它的增益主要应该来自预测器带来的未来信息，而不是"用图来描述场景"这件事。这也提示一个更省事的做法：把预测器的输出直接加进纯文本提示里，很可能就能拿到大部分收益。我可以加一组"纯文本 + 预测轨迹、不用图结构"的对照来验证，0.6B 单卡大约 2 小时。要不要加？

---

### **User**  <sub>2026-11-26 04:29 UTC</sub>

但这样就失去了论文的motivation，论文的motivation是相比输入文本/imgae token来说，graph的表现形式能得到更好的planning结果，因为graph适合表征物体之间的关系（物理世界）

---

### **Claude**  <sub>2026-11-26 04:30 UTC</sub>

你说得对。我上一条的建议实际上是绕开了图，这偏离了论文要验证的问题。不过需要分清：目前的结果并没有否定"图更适合表达关系"这个动机，因为我们还没有真正测到它。原因有三点。

**1. "加历史图"那组测的不是图表示，只是换了一种文本写法。** 那组是把图展开成"A 在前方、距离多少"这样的三元组文字喂给 LLM，所以它测的是"关系描述文本 vs 坐标文本"。论文里真正属于图表示的是图 token：用 GNN 把整张图编码成向量，直接作为 token 输入 LLM，物体之间的关系都在里面。这个东西我们只在完整模型里用过，而完整模型同时混进了回归头、CoT，还有我前面说的 bug。所以图 token 本身到底有没有用，现在还不知道。

**2. 我们的文本基线比论文里的对照强得多。** 论文把图和图像 token、自然语言描述做对比：图像要 LLM 自己从像素里推几何关系，图结构的优势很明显。我们的基线文本是以自车为原点的精确坐标和速度，这已经是半结构化的输入了，又用 2 万条样本微调，LLM 能自己学会算相对关系。在这种强基线下，图的优势本来就会被压小。

**3. 有一个结果其实是支持论文动机的。** 未来图是 GATv2 在物体关系图上做消息传递预测出来的，它带来了 11% 的提升，在前方有车、交互多的场景里是 18%。还需要确认的是：这个收益到底来自关系建模，还是只要有任何一个轨迹预测器就够了。

为了在"图表示 vs 文本"这个层面直接检验论文的动机，我建议加三组对照，每组只改一个因素：

| 实验 | 对比 | 检验什么 |
|---|---|---|
| A. 图 token 消融 | 未来图文本 vs 未来图文本 + 图 token（都不带 CoT 和回归头） | 输入 LLM 的图向量本身是否有增益 |
| B. 预测器去掉边 | GATv2 预测器 vs 去掉所有边、每辆车独立预测的同结构预测器，下游 LLM 设置相同 | 增益是来自物体之间的关系建模，还是只来自"有预测" |
| C. 少数据 | 只用 2000 条训练样本，再比一次基线 vs 历史图 vs 未来图 | 数据少、LLM 学不出几何关系时，图的结构先验是否更有价值 |

- **A** 最直接对应论文的核心主张，代码已经支持，只需在 [sage/llm.py](sage/llm.py) 的变体配置里加一项，单卡约 2.5 小时。
- **B** 要重训一次预测器（约 10 分钟），再重新生成未来图和对应的训练数据，然后训一次 LLM，大约 3 小时。如果有边的版本明显更好，就能直接说明"关系建模"是收益来源，这是论文动机最有力的证据。
- **C** 每组不到半小时。

另外，完整模型那两处修复（回归头不再和生成第一个词共用 [PLAN] 的位置，第二阶段文本损失权重保持 1.0）也照常重训。这样最后一张表能把"图结构""图 token""CoT""回归头"各自的贡献分开。

现在评测还占着几张卡，大约 1 小时后空出来，四张卡正好够同时跑 A、B、C 和修复后的完整模型。要不要全部跑？还是先跑 A 和 B？

---

### **User**  <sub>2026-11-26 04:31 UTC</sub>

全部跑，卡空出来就开始

---

### **Claude**  <sub>2026-11-26 04:31 UTC</sub>

好，我先把代码改好，再写一个调度脚本：哪张卡空出来，就自动把排队的任务放上去。

---

### **Claude**  <sub>2026-11-26 04:32 UTC</sub>

Let me verify the edge-feature layout really is "8-dim geometry + 4 relations + 9 edge types".

---

### **User**  <sub>2026-11-26 04:32 UTC</sub>

现在的模型架构是什么样的，input和output分别是什么

---

### **Claude**  <sub>2026-11-26 04:33 UTC</sub>

两个新改动的冒烟测试都通过了（模型还没训练，输出乱是正常的）。先把调度脚本挂上，然后回答你的架构问题。

---

### **Claude**  <sub>2026-11-26 04:33 UTC</sub>

A 已经在 GPU1 上开跑，其余几组等卡空出来会自动接上。最后把汇总脚本补上新实验的行：

---

### **Claude**  <sub>2026-11-26 04:33 UTC</sub>

四组新实验已经挂上了：A 在 GPU1 上开跑，D、B、C 会在 GPU0、2、3 的评测跑完后自动依次接上。全部跑完大约要 5 小时，结果会自动汇总到 `outputs/results.md`。

下面回答架构问题。

## 整体结构

```
AV2 场景（过去 2 s 每 0.5 s 一帧共 5 帧，以 t0 时刻的自车为原点）
 所有物体的位置/速度/朝向 + 地图（车道中心线、人行横道）+ 导航指令
        │
        ▼
① 构建场景图 G_t（5 帧各一张）
   节点：自车 + ≤32 个物体、≤48 条车道、≤8 个人行横道
   边：9 种（物体-物体 <30 m、车道前后左右、物体在某车道上、物体靠近人行横道）
       每条边带 21 维特征（相对位置/速度、碰撞时间、前/后/左/右关系）
        │
        ▼
② 时空图预测器（单独预训练，训练 LLM 时冻结）
   GATv2 三层，在图上做消息传递 → GRU 按时间汇总 5 帧 → 逐步预测
   输出：每个物体未来 3 s（6 步）的位置、速度、朝向
   → 用同样的规则重建 6 张未来场景图 Ĝ
        │
        ├──────────────────────────────┐
        ▼                              ▼
③ 把图写成文本                    ④ 图 token（只有 full 和 futg 用）
   节点表、t0 时刻的关系三元组、      GATv2 的节点向量 → 注意力池化
   历史变化，                         → 8 个历史 token + 8 个未来 token
   预测的 +1/+2/+3 s 状态（冲突      （映射到 LLM 的词向量维度）
   会标 "!"）、导航指令、任务说明
        │                              │
        └──────────────┬───────────────┘
                       ▼
⑤ Qwen3（0.6B / 1.7B / 4B）+ LoRA，r=16
   输入序列：[图 token] [文本提示] → 生成
```

## 输入和输出

**输入**
- 一个场景过去 2 s 的状态和地图，以 t0 时刻的自车为原点：x 朝前、y 朝左，单位米。
- 导航指令：直行、左转、右转、停车四选一，由真实路线推出来。
- 预测器不在 LLM 里面：它先离线跑一遍，把预测结果写进文本提示。

**输出**
- 文本：自车未来 6 s 的 12 个路点（每 0.5 s 一个），格式是 `Trajectory: [(x1,y1),...,(x12,y12)]`。
- 带 CoT 的变体会先写四步推理：趋势分析、交互评估、冲突预判、意图决策，然后再给轨迹。
- 完整模型多一个回归输出：推理之后插入一个学习得到的 [PLAN] 向量，用一个 MLP 读它的隐藏状态，预测 12 步的位移，累加后得到路点。

**训练损失**
- 只输出文本的变体：只用文本交叉熵。
- 完整模型：轨迹回归损失（SmoothL1）+ α × 文本交叉熵。
  - 原版分两段：先 α=1 训 1 个 epoch，再 α=0.1 训 2 个 epoch。
  - 新的 full v2 两段都用 α=1，而且 [PLAN] 那个位置只给回归头用，不再参与预测下一个词。

## 各变体分别加了什么

| 变体 | 文本提示 | 预测的未来图 | CoT | 图 token | 回归头 |
|---|---|---|---|---|---|
| raw（基线） | 坐标列表 | – | – | – | – |
| hist | 图三元组（历史和当前） | – | – | – | – |
| fut | 图三元组 + 预测的未来 | ✓ | – | – | – |
| cot | 同 fut | ✓ | ✓ | – | – |
| full / full v2 | 同 fut | ✓ | ✓ | ✓ | ✓ |
| **[A] futg** | 同 fut | ✓ | – | ✓ | – |
| **[B] fut_noaa** | 同 fut，但预测器去掉了物体-物体之间的边 | ✓ | – | – | – |
| **[C] *_2k** | raw / hist / fut，只用 2000 条训练数据 | | | | |

几处具体实现：
- 实验 B 只删掉 GATv2 里物体-物体的边，物体和地图之间的边保留，参数量完全一样。这样能单独检验"物体之间的关系建模"是不是增益来源。
- 新增的 futg 变体在 [sage/llm.py](sage/llm.py)。
- 去掉某类边的开关在 [sage/models.py](sage/models.py)，用 `--drop_etypes` 参数控制。
- 调度脚本是 [scripts/exp2_jobs.sh](scripts/exp2_jobs.sh)。

---

### **User**  <sub>2026-11-26 04:35 UTC</sub>

对于这个模型架构来说，你觉得体现出graph的作用了吗

---

### **Claude**  <sub>2026-11-26 04:35 UTC</sub>

说实话，现在这个架构里 graph 只体现了一部分，而且主要不在 LLM 内部。下面按 graph 出现的三个地方逐一看。

## Graph 在哪几个地方起作用

**1. 场景图写成文本：基本没体现**

图被展开成"A1 在 EGO 前方 12 m"这样的三元组文字，LLM 看到的只是一串文字，图的结构没了。它学到的，和直接给它坐标让它自己算差不多。hist 和基线几乎一样（2.028 vs 1.998），已经说明了这一点。

**2. 时空图预测器：真正在做关系计算的只有这里**

GATv2 在物体和车道之间做消息传递，预测的是"别的车接下来会怎样"。加未来图之后 ADE@6s 降了 11%，交互多的场景降了 18%，收益主要来自这里。

但这样一来，整个结构更像"一个 GNN 预测器 + 一个读预测结果的 LLM"。graph 的作用发生在 LLM 之前，LLM 读到的仍然是文字。这个收益到底是不是来自关系建模，正是实验 B 要回答的。

**3. 图 token：唯一把图直接输入 LLM 的通道，但设计上很弱**

- **压缩太狠。** 几十个节点、5 帧，被注意力池化成 8 个 token。哪辆车和哪条车道是什么关系，这种节点级的信息在池化里基本丢了。LLM 拿到的只是一个全局摘要，没法去关注某一辆具体的车。
- **和文本对不上号。** 文本里写的是 "A1"，图 token 里不知道哪部分对应 A1，LLM 没办法把两边的信息连起来。
- **编码器是冻结的。** 它是为预测其他车的运动训练的，不是为自车规划训练的。
- **和文本内容重复。** 文本里已经有全部信息，LLM 完全可以忽略这些 token。

**4. 输出端没用到图**

规划结果直接是 xy 坐标，和车道图结构没有关系。

另外还有评测本身的问题：开环 L2 误差主要由自车自身的运动决定，误差的大头是纵向（1.85 m 对横向 0.36 m）。关系推理真正关键的只有少数交互场景，平均下来会被稀释。

## 怎样让 graph 真正发挥作用

按我认为的价值从高到低：

1. **一个节点一个 token，并和文本对齐。** 不再池化成 8 个 token，而是给自车、最相关的约 16 个物体、约 16 条相关车道各输出一个 GNN 节点向量。每个向量直接放在文本中对应名字的后面，比如 `A1 <node_emb>`。这样 LLM 读到 "A1" 时就能拿到它在图上的关系特征。这是图结构 LLM 的常见做法（GraphGPT、LLaGA 都是这个思路）。
2. **一起微调图编码器。** 至少解冻最后一层 GATv2，让节点向量为规划服务。代码里已经有 `--train_graph` 开关。
3. **在输出端用上图。** 先在车道节点里选一条目标车道（分类），再预测沿这条车道的速度曲线，最后生成轨迹。横向走哪条道由图结构决定，纵向快慢由交互决定，正好对应误差的大头。
4. **评测加上按交互强度分组。** 比如前方有车、匀速外推误差大的场景单独报，这才是图应该发挥作用的地方。

## 和正在跑的实验的关系

正在跑的实验 A（futg）测的正是现在这种"池化成 8 个 token"的版本。我预计它的提升会很小，如果真是这样，就印证了上面第 3 点的判断。

我的建议是：等 A 和 B 的结果出来后，实现第 1、2 点，也就是逐节点的图 token、和文本对齐、并微调图编码器，作为一组新的变体。

- 如果它在交互场景上明显好于"只有文本"的版本，就是对论文动机最直接的支持。
- 如果还是不行，就说明在有精确坐标文本的前提下，图对 LLM 规划的增益有限。这本身也是一个有价值的结论。

第 3 点改动比较大，相当于换一种规划输出方式。可以等前面的结果出来之后再决定做不做。

---

### **User**  <sub>2026-11-26 04:37 UTC</sub>

好，按照你的建议来

---

### **Claude**  <sub>2026-11-26 04:37 UTC</sub>

这次要做的是让图真正进入 LLM：每个节点生成一个 token，放在文本里该节点名字的旁边，并且图编码器一起参与训练。我先看文本里的节点名（A1、L1 这些）和图里的节点下标是怎么对应的。

---

### **Claude**  <sub>2026-11-26 04:38 UTC</sub>

Start with the serialization side: pull the node-selection logic out into a helper and generate the name → node index map.

---

### **Claude**  <sub>2026-11-26 04:38 UTC</sub>

The refactored text is byte-for-byte identical to the cached version. Generating the node maps in the background while I update the model:

---

### **Claude**  <sub>2026-11-26 04:40 UTC</sub>

最后把按场景类型分组的评测也做成脚本，以后每次汇总都能直接看交互场景：

---

### **Claude**  <sub>2026-11-26 04:40 UTC</sub>

The full model's prediction file is being rewritten by the in-progress re-evaluation, so it's incomplete. I'll skip runs whose files aren't complete:

---

### **Claude**  <sub>2026-11-26 04:40 UTC</sub>

已经按建议实现了第 1、2 点：每个节点在提示里有自己的图 token，并且图编码器一起训练。这组实验叫 N，已经排进队列，等前面四组都拿到卡之后，下一张空出来的卡就会开跑。测试的是一个很小的批次，训练和推理都能跑通，旧变体也没受影响；完整训练还没开始。

**改了什么**

- **一个节点一个图 token，和文本对齐。** 原来是把整张图压成 8 个 token；现在提示里每个节点定义的名字后面，比如 `EGO`、`A1`、`L1`、`X1`，都紧跟一个该节点自己的图 token：
  ```
  EGO<node>: speed 6.5 m/s, ...
  A1<node> vehicle: pos (12.3,0.4), speed 7.9, ...
  L1<node>: heading 0 deg, length 66 m; left=L2, next=L3
  ```
  每个节点 token 由三部分拼起来：
  - 当前帧的 GATv2 向量，带着它和周围物体、车道的关系；
  - GRU 汇总的过去 2 s 历史状态；
  - 预测的未来 3 s 各帧向量的平均。
  
  这样 LLM 读到 "A1" 时，能直接拿到 A1 在图上的关系特征。
- **图编码器一起微调**，学习率 1e-4，比 LoRA 的小，避免把预训练好的预测能力冲掉。
- **按场景类型分组的评测**：每次汇总结果时自动生成 `outputs/results_by_scene.md`，分开报前方有车、匀速外推难预测等场景，因为图主要应该在这些场景里起作用。

我还检查过：生成节点和名字对应关系时改动了提示的生成代码，抽查 50 个场景，提示文字和之前完全一样。所以 N 和已有的 fut、futg 用的是同一份文本，只差图 token 的给法。

**实验排队情况**

| 组 | 检验什么 | 状态 |
|---|---|---|
| A futg：池化成 8 个图 token | 粗粒度的图 token 是否有用 | GPU1 上训练中 |
| D 完整模型修复版 | 修掉 [PLAN] 冲突后，完整模型是否变好 | 等卡 |
| B 预测器去掉物体之间的边 | 收益是否来自物体之间的关系建模 | 等卡 |
| C 只用 2000 条训练数据 | 数据少时图是否更有用 | 等卡 |
| **N 逐节点图 token + 微调编码器** | **图表示进入 LLM 后是否带来增益** | 排在 A–D 之后 |

**怎么判断结果**：重点比较 N、futg、fut 三组，它们的文本完全相同，只有图 token 的给法不同。
- 如果 N 在交互场景上明显好于 fut，就能说明图表示本身对规划有用，支持论文的动机。
- 如果 B 明显比 fut 差，就能说明增益来自物体之间的关系建模。

目前四张卡都在用：GPU0、2、3 还在跑上一轮的评测，GPU1 在训 A。估计 A–D 大约 5 小时后出结果，N 再晚 2–3 小时。

新增和修改的文件：
- [sage/llm.py](sage/llm.py)：`node` 变体，以及节点 token 的插入
- [sage/serialize.py](sage/serialize.py)：`node_map`，名字到节点下标的对应
- [scripts/build_nodemap.py](scripts/build_nodemap.py)
- [scripts/stratify.py](scripts/stratify.py)
- [scripts/node_job.sh](scripts/node_job.sh)

---

### **User**  <sub>2026-11-26 04:49 UTC</sub>

你把我和你的对话记录下来，存入/home/ubuntu/test/graph_llm/sage_drive中，然后把整个文件上传github（先设为私密）

---

### **Claude**  <sub>2026-11-26 04:49 UTC</sub>

`gh` isn't installed. Let me check for any existing GitHub credentials (without printing their contents):

---

### **Claude**  <sub>2026-11-26 04:50 UTC</sub>

去掉系统通知和压缩摘要这类不属于对话的内容，并把通过选项框给出的回答和中途插入的问题补进来：

---

### **Claude**  <sub>2026-11-26 04:50 UTC</sub>

前半段对话的一部分没有出现在当前这个会话文件里。我在其他会话文件里找找：

---

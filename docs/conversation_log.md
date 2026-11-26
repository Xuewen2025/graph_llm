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

### **Claude**  <sub>2026-11-26 04:51 UTC</sub>

正在补全对话记录：中途插入的消息存放在另一种记录类型里（queued_command），我把它加进导出脚本。

---

### **Claude**  <sub>2026-11-26 04:51 UTC</sub>

所有用户消息都在，API key 已经打码。接下来在本地建 git 仓库：只提交代码、文档和结果汇总，不提交数据、模型权重和 `.env`。

---

### **Claude**  <sub>2026-11-26 04:51 UTC</sub>

Secret scan came back clean. Committing locally with a repo-local author identity:

---

### **Claude**  <sub>2026-11-26 04:51 UTC</sub>

对话记录已经存到项目里，本地仓库也已经提交好了，但还没上传到 GitHub：这台机器没有装 `gh`，我也拿不到你的 GitHub 凭据。

**对话记录**
- 存在 [docs/conversation_log.md](docs/conversation_log.md)，按时间排列，从第一条复现请求到这次的上传请求都在。你的 16 条消息都收进去了，包括选项框里的回答和中途插进来的消息。
- 只保留双方的文字消息，工具调用和命令输出都去掉了。
- 文件里的 OpenRouter key 已打码为 `sk-or-v1-<REDACTED>`。
- 导出脚本是 [scripts/export_conversation.py](scripts/export_conversation.py)，以后可以重新生成。README 末尾加了指向记录的链接。

**本地仓库**
- 已提交一次 `c4f1683`，共 36 个文件，232 KB：代码、脚本、README、对话记录，以及结果汇总 `outputs/results.md` 和 `outputs/results_by_scene.md`。
- 排除了 `.env`、`data/`、模型权重和日志。提交前扫过一遍，没有发现 key 或邮箱。
- 提交作者设成了你的名字和公司邮箱，只在这个仓库生效。推送后，能访问这个仓库的人都能看到这个邮箱；如果想换成别的邮箱，告诉我就行。

**上传需要你提供认证，二选一：**

1. **你在终端登录 `gh`（推荐）**：
   ```bash
   # 安装 gh（需要 sudo）后：
   gh auth login
   ```
   告诉我登录好了，我会执行 `gh repo create sage_drive --private --source . --push`。
2. **你自己建一个空的私有仓库**：在 GitHub 网页上建好后把地址发给我，我来配置远程仓库地址并推送。前提是这台机器有能推送的 SSH key 或凭据。

另外说明一点：刚才排查对话记录格式时，有一条本地命令的输出里出现了 OpenRouter key 的前半段。它只出现在这次会话里，没有写进任何文件，也没有发到外部。如果你介意，可以在 OpenRouter 后台换一个新 key。

后台实验还在跑（A 在 GPU1 训练中，D、B、C、N 等卡），结果出来后我再提交一次更新。

---

### **User**  <sub>2026-11-26 04:55 UTC</sub>

我已经创建：https://github.com/Xuewen2025/graph_llm.git

---

### **Claude**  <sub>2026-11-26 04:55 UTC</sub>

HTTPS 推送需要账号密码，这台机器上没有。换 SSH 地址再试一次：

---

### **Claude**  <sub>2026-11-26 04:55 UTC</sub>

两种方式都推送失败了，所以代码还没传上去：
- HTTPS：这台机器上没有保存 GitHub 的用户名和 token。
- SSH：没有能访问这个仓库的 SSH key。

远程地址已经配置好了（`origin` 指向你的 `graph_llm` 仓库），代码也已在本地提交。最简单的办法是你在 VSCode 的终端里自己推一次，系统会提示你输入凭据：

```bash
cd /home/ubuntu/test/graph_llm/sage_drive
git push -u origin main
```

- **Username**：填 `Xuewen2025`。
- **Password**：不能填 GitHub 登录密码，要填 Personal Access Token。生成路径是 GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens，只勾选 `graph_llm` 这个仓库，权限给 Contents: Read and write。

想让以后的推送不用每次都输：先运行 `git config --global credential.helper store`，再推一次，凭据会以明文保存在 `~/.git-credentials`。之后实验结果出来，我就能直接帮你提交并推送。

如果你更愿意用 SSH，我可以在这台机器上生成一个专用 key。你把公钥加到仓库的 Settings → Deploy keys，并勾选 "Allow write access"，之后我就能直接推送。

---

### **User**  <sub>2026-11-26 04:56 UTC</sub>

<pasted_content id="0282">
git push -u origin main
Missing or invalid credentials.
Error: connect ECONNREFUSED /run/user/1000/vscode-git-331b1bb4f8.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:2083:16) {
  errno: -111,
  code: 'ECONNREFUSED',
  syscall: 'connect',
  address: '/run/user/1000/vscode-git-331b1bb4f8.sock'
}
Missing or invalid credentials.
Error: connect ECONNREFUSED /run/user/1000/vscode-git-331b1bb4f8.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:2083:16) {
  errno: -111,
  code: 'ECONNREFUSED',
  syscall: 'connect',
  address: '/run/user/1000/vscode-git-331b1bb4f8.sock'
}
remote: Repository not found.
fatal: Authentication failed for 'https://github.com/Xuewen2025/graph_llm.git/'
</pasted_content id="0282">

---

### **Claude**  <sub>2026-11-26 04:56 UTC</sub>

推送失败不是凭据填错了：VSCode 用来弹窗收账号密码的进程已经断开（报错里的 `ECONNREFUSED ... vscode-git-*.sock`），git 根本没机会问你要 token。没有凭据时，GitHub 对私有仓库一律返回 "Repository not found"，所以那句报错不代表地址不对。

先让 git 绕开 VSCode 的弹窗，直接在终端里问凭据。在同一个终端依次运行：

```bash
cd /home/ubuntu/test/graph_llm/sage_drive
unset GIT_ASKPASS SSH_ASKPASS VSCODE_GIT_ASKPASS_MAIN VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_EXTRA_ARGS VSCODE_GIT_IPC_HANDLE
git config --global credential.helper store
git push -u https://<email>/Xuewen2025/graph_llm.git main
```

- 提示 `Password for 'https://<email>':` 时，粘贴你的 Personal Access Token（需要对 `graph_llm` 有 Contents: Read and write 权限）。不能用 GitHub 登录密码。粘贴时屏幕上不会显示字符，直接回车就行。
- 推送成功后再运行一次 `git push -u origin main`，把本地分支和 `origin` 关联上。这次会直接用刚保存的 token，不会再问。
- 第三行的作用是把 token 明文存进 `~/.git-credentials`。之后实验结果出来，我就能直接帮你提交并推送。不想存的话，跳过这一行即可，代价是每次推送都要再输一次 token。

另一个办法是在 VSCode 里按 `Ctrl+Shift+P`，执行 "Developer: Reload Window"，等弹窗功能恢复后再用 `git push -u origin main`。

推完告诉我一声，我会确认远程仓库里的内容是否完整。

---

### **User**  <sub>2026-11-26 05:00 UTC</sub>

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBF+gP/jG0ukeIFsjQWjrSXx++yAkCJ2sdxo6o7YdTl6 graph_llm-deploy
我的key就是这个

---

### **Claude**  <sub>2026-11-26 05:00 UTC</sub>

代码还没推上去：你发的那串公钥，配对的私钥不在这台服务器上（应该在你自己电脑上），服务器用不了它来推送。

所以我在服务器上另生成了一对专用 key。私钥是 `~/.ssh/graph_llm_deploy`，只留在服务器上。仓库也已经设置好，推送时走这把 key，远程地址改成了 SSH。

请把下面这个**新公钥**加到仓库里：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPQ3SwG6gGuW/Gxb1zmdc8yQNIjDF3dNHvoo+ZbSTjai graph_llm-deploy-server
```

1. 打开 https://github.com/Xuewen2025/graph_llm/settings/keys
2. 点 **Add deploy key**，Title 随便填，比如 `server`，Key 粘贴上面这一行。
3. **勾选 "Allow write access"**，不勾就只能读、不能推。
4. 点 Add key。

加好后告诉我，我来推送，并检查远程仓库里的内容是否完整。这把 key 只对 `graph_llm` 这一个仓库有效，以后实验结果出来，我可以直接帮你提交并推送。

你之前那把 `graph_llm-deploy` 如果已经加到了仓库里，留着也行；不用的话可以删掉。

---

### **User**  <sub>2026-11-26 05:01 UTC</sub>

<pasted_content id="0282">
<PRIVATE KEY REDACTED>
</pasted_content id="0282">

---

### **Claude**  <sub>2026-11-26 05:01 UTC</sub>

你发的这把私钥应该和你之前那把公钥 `graph_llm-deploy` 是一对，我用它来推送。先把它保存成仅当前用户可读的文件：

---

### **Claude**  <sub>2026-11-26 05:01 UTC</sub>

推送成功，再确认一下远程和本地是否一致：

---

### **Claude**  <sub>2026-11-26 05:01 UTC</sub>

项目已经推到 https://github.com/Xuewen2025/graph_llm 的 `main` 分支，远程和本地都是 `c4f1683` 这次提交，完全一致。

- **上传了什么**：代码、脚本、README、对话记录 `docs/conversation_log.md`，以及结果汇总。数据、模型权重、日志和 `.env` 都没有上传。
- **私钥的处理**：你发的私钥存在服务器的 `~/.ssh/graph_llm_user`，只有你的账号可读。我核对过，它和你之前那把公钥是一对。仓库已设置为推送时只用这把 key，以后我可以直接帮你推结果。
- **安全提醒**：这把私钥已经以明文出现在聊天记录里。它只是一个仓库的 deploy key，风险范围有限，但还是建议实验做完后在仓库 Settings → Deploy keys 里删掉它，换一把新的。
- **可以删掉的 key**：我之前在服务器上生成的 `graph_llm-deploy-server` 没有用上，GitHub 那边不用添加。需要的话我可以把它从服务器上删掉。

本地的两份结果汇总 `outputs/results.md` 和 `outputs/results_by_scene.md` 又被更新过了，因为后台还在跑评测。等这一轮实验（A、B、C、D、N）全部出结果后，我会一起提交并推送一次。

---

### **User**  <sub>2026-11-26 05:01 UTC</sub>

好，大概多久可以跑完

---

### **Claude**  <sub>2026-11-26 05:02 UTC</sub>

按现在的速度，全部跑完大约还要 **5 小时**，预计北京时间 **18:00 左右**（UTC 10:00）。

| 时间（北京） | 什么完成 | 用卡 |
|---|---|---|
| ~13:20 | 1.7B 完整模型重新评测完，GPU3 空出来 | GPU3 |
| ~13:40 | 4B 完整模型评测完，GPU0 空出来 | GPU0 |
| ~14:30 | **C** 只用 2000 条数据训练的三组 | 先空出来的那张 |
| ~15:00 | **A** 池化图 token（目前 750/3750 步） | GPU1 |
| ~16:10 | **D** 修复版完整模型（目前 200/3750 步，带 CoT，评测比较慢） | GPU2 |
| ~16:30 | **B** 去掉物体之间边的预测器（重训预测器 + 训练 LLM） | 另一张空出来的卡 |
| **~17:30–18:00** | **N** 逐节点图 token（C 跑完后接上那张卡） | 同 C |

这些时间是按目前每分钟跑的步数推算的，前后可能差半小时左右。N 等的是 C 空出来的卡，所以最晚出结果。

0.6B 完整模型用修好的推理代码重新评测完了：
- 解析失败从 27% 降到 0。
- ADE@6s：回归头输出 2.169，文本输出 2.432。修复前分别是 2.600 和 2.881，但仍然比基线 1.998 差。
- 这和之前的判断一致：问题出在训练设计，所以 D 的修复版重训有必要。

1.7B 和 4B 的完整模型半小时内也会评测完，到时会补进规模对比表。全部结果出来后，我会把消融表、规模表和按场景类型的分组表一起整理好，提交并推送到 GitHub。

---

### **User**  <sub>2026-11-26 05:06 UTC</sub>

跑完了告诉我，结果推到github

---

### **Claude**  <sub>2026-11-26 05:07 UTC</sub>

等全部跑完我会收到通知，然后汇总结果并推送到 GitHub。先挂一个后台等待：

---

### **Claude**  <sub>2026-11-26 05:07 UTC</sub>

好的。后台已经在等，全部实验跑完我会收到通知，预计北京时间 18:00 左右。届时我会：

1. 汇总消融表、模型规模表和按场景类型的分组表，用中文给你结论，重点回答 graph 到底有没有用：
   - 实验 N（逐节点图 token）有没有比只有文本的版本好；
   - 实验 B（预测器去掉物体之间的边）有没有变差。
2. 把结果写进 README 的 Results 部分。
3. 重新导出对话记录，然后提交并推送到 https://github.com/Xuewen2025/graph_llm 。

这段时间你随时可以问进度。

---

### **User**  <sub>2026-11-26 06:19 UTC</sub>

现在进度怎么样了

---

### **Claude**  <sub>2026-11-26 06:19 UTC</sub>

比预计的快，**预计北京时间 16:40 左右全部跑完**（UTC 08:40）。四张卡都在用：

| 实验 | 进度 | 预计完成（北京时间） |
|---|---|---|
| **C** 只用 2000 条数据训练 | ✅ 已完成 | – |
| **A** 池化图 token | 2825/3750 步 | ~15:05 |
| **D** 完整模型修复版 | 2000/3750 步 | ~16:10 |
| **B** 预测器去掉物体之间的边 | 预测器已重训好，LLM 875/3750 步 | ~16:15 |
| **N** 逐节点图 token | 350/3750 步 | ~16:40 |

**已出结果的部分（ADE@6s，越低越好）**

实验 C，只用 2000 条数据训练：

| 对照组 | 2000 条数据 | 全部 2 万条数据（对照） |
|---|---|---|
| 基线（纯文本） | 2.434 | 1.998 |
| + 历史图 | 2.388（−1.9%） | 2.028（+1.5%） |
| + 未来图 | **2.166（−11%）** | **1.776（−11%）** |

- 数据少时，历史图有一点提升（−1.9%），数据多了就没有了（+1.5%）。这和"图的结构先验在数据少时更有价值"的预期方向一致，但幅度很小，可能只是噪声。
- 未来图在两种数据量下都稳定降低 11%。

模型规模对比，用的是旧版完整模型，1.7B 和 4B 都只训了 1 个 epoch：

| 模型 | 基线 | 旧版完整模型 |
|---|---|---|
| 0.6B | 1.998 | 2.169（差 9%） |
| 1.7B | 2.116 | 2.855（差 35%） |
| 4B | 2.035 | 2.549（差 25%） |

在三个尺寸上，旧版完整模型都比基线差，问题出在完整模型的训练设计本身，不是模型大小。等 D 的修复版出来，如果效果变好，我建议用修复版在 1.7B 和 4B 上再各跑一次规模对比。

全部跑完后，我会按之前说的汇总结论，并提交推送到 GitHub。

---

### **User**  <sub>2026-11-26 06:20 UTC</sub>

大概还需要多长时间

---

### **Claude**  <sub>2026-11-26 06:20 UTC</sub>

上次报进度才过去一分钟左右，预估没变：**还要大约 2 小时 20 分钟，北京时间 16:40 左右全部完成。**

| 实验 | 当前进度 | 还需约多久 |
|---|---|---|
| A 池化图 token | 2875/3750 步 | 45 分钟（含评测） |
| D 完整模型修复版 | 2025/3750 步 | 1 小时 50 分钟（带 CoT，评测比较慢） |
| B 预测器去掉物体之间的边 | 900/3750 步 | 1 小时 55 分钟 |
| N 逐节点图 token | 375/3750 步 | 2 小时 20 分钟 |

N 最晚完成，所以总时间由它决定。全部跑完后我会把结果汇总给你，并推送到 GitHub。

---

### **User**  <sub>2026-11-26 06:41 UTC</sub>

现在什么进度

---

### **Claude**  <sub>2026-11-26 06:41 UTC</sub>

四组都还在训练，没有一组开始评测，所以暂时还没有新结果。全部完成的时间没变，仍是北京时间 16:40 左右。

| 实验 | 训练进度 | 预计完成（北京时间，含评测） |
|---|---|---|
| A 池化图 token | 3450/3750 步（92%） | ~15:05 |
| D 完整模型修复版 | 2525/3750 步（67%） | ~16:10 |
| B 预测器去掉物体之间的边 | 1500/3750 步（40%） | ~16:15 |
| N 逐节点图 token | 925/3750 步（25%） | ~16:40 |

A 大约 25 分钟后出结果。全部跑完后我会一次性汇总，并推送到 GitHub。

---

### **Claude**  <sub>2026-11-26 07:05 UTC</sub>

A 已经出结果，其余三组还在训练，全部完成仍预计在北京时间 16:40 左右。

**A 的结果：把整张图池化成 8 个 token 再输入 LLM，基本没有带来提升**（ADE@6s，越低越好）

| 场景 | 只有未来图文本 | + 池化图 token（A） |
|---|---|---|
| 全部 | 1.776 | 1.794 |
| 前方有车 | 1.445 | 1.438 |
| 匀速外推难预测的场景 | 2.819 | 2.906 |

碰撞率从 4.20% 降到 3.75%，但其他指标基本持平。这和之前的判断一致：池化把节点之间的关系信息压掉了，文本里又已经有全部内容，LLM 用不上这 8 个 token。真正检验"图表示进入 LLM 有没有用"的是 N（逐节点图 token），它还在训练。

**剩余进度**

| 实验 | 训练进度 | 预计完成（北京时间） |
|---|---|---|
| D 完整模型修复版 | 3075/3750 步 | ~16:10 |
| B 预测器去掉物体之间的边 | 2150/3750 步 | ~16:15 |
| N 逐节点图 token | 1550/3750 步 | ~16:40 |

---

### **Claude**  <sub>2026-11-26 07:12 UTC</sub>

和 7 分钟前那次查看相比，没有新结果。剩下三组都还在训练，全部完成时间不变，仍是北京时间 16:40 左右。

| 实验 | 训练进度 | 预计完成（北京时间，含评测） |
|---|---|---|
| D 完整模型修复版 | 3250/3750 步（87%） | ~16:10 |
| B 预测器去掉物体之间的边 | 2350/3750 步（63%） | ~16:15 |
| N 逐节点图 token | 1725/3750 步（46%） | ~16:40 |

你不用一直来看。全部跑完后，后台任务会通知我，我会把结论汇总发给你，并推送到 GitHub。

---

### **Claude**  <sub>2026-11-26 07:22 UTC</sub>

和 10 分钟前相比没有新结果，三组都还在训练。全部跑完仍预计在北京时间 16:40 左右。

| 实验 | 训练进度 | 预计完成（北京时间，含评测） |
|---|---|---|
| D 完整模型修复版 | 3475/3750 步（93%） | ~16:10 |
| B 预测器去掉物体之间的边 | 2625/3750 步（70%） | ~16:15 |
| N 逐节点图 token | 2000/3750 步（53%） | ~16:40 |

D 大约 10 分钟后训完，但它要生成推理文字，评测要再花半小时左右。

---

### **User**  <sub>2026-11-26 07:35 UTC</sub>

现在的进度

---

### **Claude**  <sub>2026-11-26 07:35 UTC</sub>

刚才会话重启，我用来等实验结束的后台监听也跟着断了，已经重新挂上。实验本身是独立在后台跑的，没有受影响。

| 实验 | 状态 | 预计完成（北京时间） |
|---|---|---|
| A 池化图 token | ✅ 已完成 | – |
| C 只用 2000 条数据 | ✅ 已完成 | – |
| D 完整模型修复版 | 训练完成，评测中（4/63 批） | ~16:05 |
| B 预测器去掉物体之间的边 | 训练 3000/3750 步 | ~16:10 |
| N 逐节点图 token | 训练 2350/3750 步 | ~16:40 |

全部完成仍预计在北京时间 16:40 左右。GPU1 在 A 跑完后已经空出来了。

全部跑完后我会把结论汇总发给你，并推送到 GitHub。

---

### **Claude**  <sub>2026-11-26 07:48 UTC</sub>

13 分钟前那次查看之后还没有新结果，进度都在正常推进，全部完成仍预计在北京时间 16:40 左右。

| 实验 | 状态 | 预计完成（北京时间） |
|---|---|---|
| D 完整模型修复版 | 评测中，33/63 批 | ~16:05 |
| B 预测器去掉物体之间的边 | 训练 3350/3750 步 | ~16:10 |
| N 逐节点图 token | 训练 2675/3750 步 | ~16:40 |

---

### **Claude**  <sub>2026-11-26 07:55 UTC</sub>

还没有新结果，D 是最接近出结果的一组。全部完成仍预计在北京时间 16:40 左右。

| 实验 | 状态 | 预计出结果（北京时间） |
|---|---|---|
| D 完整模型修复版 | 评测中，50/63 批 | **约 6 分钟后（~16:01）** |
| B 预测器去掉物体之间的边 | 训练 3550/3750 步，之后还要评测 | ~16:15 |
| N 逐节点图 token | 训练 2875/3750 步，之后还要评测 | ~16:45 |

---

### **Claude**  <sub>2026-11-26 08:01 UTC</sub>

D（完整模型修复版）出结果了：**两处修复只带来很小的改善，完整模型依然比只用未来图文本的版本差很多。**

ADE@6s，越低越好：

| 对照组 | 全部场景 | 前方有车 | 难预测场景 |
|---|---|---|---|
| 基线（纯文本） | 1.998 | 1.752 | 3.224 |
| + 未来图 | **1.776** | **1.445** | **2.819** |
| + CoT 推理 | 2.062 | 1.757 | 3.264 |
| 完整模型（旧版，回归头输出） | 2.170 | 1.627 | 4.045 |
| 完整模型修复版 D（回归头输出） | 2.136 | 1.696 | 3.820 |
| 完整模型修复版 D（文本输出） | 2.310 | – | – |

- **完整模型变差的主要原因是 CoT，不是回归头。** 只加 CoT 这一步，误差就从 1.776 升到 2.062，完整模型是在 CoT 版本基础上搭起来的，所以先吃了这部分亏。D 修好了回归头那两处问题，但只把 2.170 降到 2.136。
- **CoT 拖后腿，我判断是推理标签的问题。** 这些标签是教师模型看了真实轨迹之后写的"事后解释"。测试时小模型要自己先生成这段推理，推理一错就会把后面的轨迹带偏。前面说"数据多了以后历史图就没有提升"，也说明这些推理没有带来文本里没有的信息。
- **回归头输出的 FVR 很高（21.5%）。** 也就是平滑性、动力学约束方面的违规比较多，因为回归头没有被约束成平滑的轨迹。

剩下两组：
- B（预测器去掉物体之间的边）：训练快结束，大约 15 分钟后出结果。
- N（逐节点图 token）：训练 3025/3750 步，大约 45 分钟后出结果。

N 是回答"图表示进入 LLM 有没有用"的关键实验。全部出来后我会给出完整结论，并推送到 GitHub。

---

### **Claude**  <sub>2026-11-26 08:42 UTC</sub>

五组实验全部跑完了。先对两组关键对比做配对 bootstrap 显著性检验（同一批场景上比较），看差异是不是噪声：

---

### **Claude**  <sub>2026-11-26 08:42 UTC</sub>

Results are clear. I'll save the significance test as a reusable script, then update the README results section:

---

### **Claude**  <sub>2026-11-26 08:42 UTC</sub>

Now update the README results section:

---

### **Claude**  <sub>2026-11-26 08:43 UTC</sub>

Re-exporting the conversation log, scanning for secrets, then committing and pushing:

---

### **Claude**  <sub>2026-11-26 08:43 UTC</sub>

The scan caught a problem: the SSH private key you pasted earlier ended up in the exported conversation log. I'll add private-key redaction to the exporter and re-export:

---

### **Claude**  <sub>2026-11-26 08:43 UTC</sub>

已经推到 GitHub 的那次提交里没有这把私钥：它是在那次提交之后才发进对话的。现在提交并推送：

---

### **Claude**  <sub>2026-11-26 08:43 UTC</sub>

全部实验都跑完了，结果和结论已推到 https://github.com/Xuewen2025/graph_llm ，提交是 `dc56770`，远程和本地已核对一致。

## 主要结果

数字是 ADE@6s，越低越好。括号里是相对于"+ 未来图"那一行的变化，星号表示 95% 置信区间不含 0（统计上显著）。

| 对照组 | 全部场景 | 前方有车 | 难预测场景 |
|---|---|---|---|
| 基线：纯文本 | 1.998 | 1.752 | 3.224 |
| + 历史图写成文本 | 2.028 | 1.813 | 3.251 |
| **+ 未来图写成文本** | **1.776** | **1.445** | **2.819** |
| B：预测器去掉物体之间的边 | 1.870（+5.3%\*） | 1.626（+12.5%\*） | 2.972 |
| A：再加 8 个池化图 token | 1.794（+1.0%） | 1.438 | 2.906 |
| N：再加逐节点图 token，并微调编码器 | **1.767**（−0.5%） | 1.443 | **2.753** |
| + CoT | 2.062（+16%\*） | 1.757 | 3.264 |
| 完整模型修复版 | 2.136 | 1.696 | 3.820 |

## 结论：graph 到底有没有用

1. **有用的是物体之间的关系建模，论文动机的这一半得到了支持。** 预测器只去掉物体之间的边，其余参数和地图边都不变，就丢掉了大约一半的收益：前方有车的场景误差升高 12.5%，统计上显著。这说明收益来自 GNN 在相互影响的物体之间传递信息，而不只是"有一个轨迹预测"。
2. **把图直接输入 LLM，在文本基础上没有额外增益，论文动机的另一半没能复现。** 两种图 token 都试过了：
   - 池化成 8 个 token：比只有文本还略差 1.0%，不显著。
   - 逐节点 token，和文本里的节点名对齐，并且一起微调编码器：好 0.5%，置信区间是 −2.4% 到 +1.5%，等于没有差别。
   
   原因是文本里已经有精确坐标，加上 2 万条数据微调，LLM 自己就能把几何关系算出来。
3. **CoT 蒸馏是负作用，完整模型变差主要就是因为它。** 推理标签是教师模型看了真实轨迹之后写的。测试时小模型要先自己生成这段推理，推理一错，后面的轨迹也跟着偏。修复回归头那两处问题只把完整模型从 2.169 降到 2.136，帮助有限。
4. **论文里基线和完整模型之间的大差距没有复现。** 论文里基线的 FVR（违反动力学约束的比例）高达 99.85%，很像是没有微调过。我们的基线认真微调过，本身就很强。

**还要注意的局限**：只做了开环评测，用的是 AV2 而不是 nuPlan，每个配置只跑了一个随机种子，1.7B 和 4B 只训了 1 个 epoch。

## 推送的内容
- README 新增了结果和结论部分。
- 结果汇总：
  - `outputs/results.md`：全部指标
  - `outputs/results_by_scene.md`：按场景类型分组
  - `outputs/significance.md`：显著性检验，新脚本 `scripts/significance.py`
- 对话记录已经更新。推送前的安全扫描发现，你发过的 SSH 私钥被导出进了对话记录，我已经在导出脚本里加了私钥打码，重新导出后才推送。之前那次推送比你发私钥更早，所以 GitHub 上从来没有出现过这把私钥。

两条建议：
- 这把私钥在聊天里出现过，做完后最好在仓库 Settings → Deploy keys 里删掉它，换一把新的。
- 如果还想继续往下做，我认为最值得的是：去掉 CoT，用"文本 + 节点图 token"的版本，在 1.7B 和 4B 上按完整的 3 个 epoch 重新做规模对比。

---

### **User**  <sub>2026-11-26 08:44 UTC</sub>

对于现在的结果，你建议如何建模和改进，graph是否有用，如果有用，如何做

---

### **Claude**  <sub>2026-11-26 08:45 UTC</sub>

我的结论是：**graph 有用，但它的价值在于"做计算"，而不在于"给 LLM 看"。** 建模上应该让图模型承担交互推理和轨迹生成，让 LLM 只做离散的语义决策，并且决策要落到图上的具体节点。下面是依据和具体做法。

## 一、实验已经说明的事

| 证据 | 说明了什么 |
|---|---|
| 预测器去掉物体之间的边后，前方有车的场景误差升高 12.5%（显著） | 图的关系归纳偏置有用：在相互影响的物体之间传递信息，能算出 LLM 自己算不出的东西 |
| 历史图写成文本、池化图 token、逐节点图 token，都没有超过纯文本 | 文本里有精确坐标时，图作为输入形式对 LLM 是冗余的 |
| 纵向误差 1.85 m，横向只有 0.36 m | 难点在速度决策，比如跟车、让行、何时减速，本质上是交互问题 |
| CoT 让误差升高 16% | 先自己生成"事后解释"式的推理，错误会传到轨迹上 |
| 回归头的 FVR 约 20% | 让 LLM 直接回归坐标，轨迹质量差 |

也就是说，现在的 SAGE-Drive 结构里，真正起作用的计算（GNN 预测器）放在了 LLM 之外，而 LLM 承担的又是它最不擅长的部分：逐个数字地写坐标。

## 二、建议的建模方式

```
场景图 G ──► 时空 GNN（交互建模）──► 节点向量 + 多模态未来预测
                 │                              │
                 │          ┌───────────────────┘
                 ▼          ▼
           LLM 决策层：输出"语义决策"，指向图上的节点
           例："沿 L1 → L3 行驶，跟随 A2，目标速度 6 m/s，在 X1 前让行"
                 │
                 ▼
           图条件轨迹解码器（GNN）：根据决策 + 交互，生成平滑轨迹
```

**1. 输出改成"沿车道图走的路径 + 速度曲线"**
- 横向：在车道图里选一条车道序列，也就是对车道节点做分类。这直接用上了车道图的拓扑结构。
- 纵向：根据被选中的前车或让行对象的预测轨迹，生成速度曲线。
- 优点：针对误差的大头（纵向），轨迹天然平滑，FVR 会降下来。

**2. LLM 只做离散决策，并且决策必须引用图节点**
- 决策内容包括：目标车道 `L_k`、交互对象 `A_k`、行为（跟随、让行、超车、停车）、目标速度档位。
- 这些标签可以从真实轨迹自动推出来：最终停在哪条车道、速度怎么变化、前方最近的车是谁。不再需要教师模型写推理。
- 这样一来，逐节点图 token 才真正有意义：LLM 要"指向"某个节点，就必须读懂那个节点的图特征。现在的设定下，它只是在文本旁边多放了一份重复信息。

**3. CoT 要么去掉，要么基于图来写**
- 最简单：去掉 CoT，或者把推理放到轨迹后面，作为解释输出，这样推理错了也不会影响轨迹。
- 如果要保留推理链，就从图中可验证的事实自动生成，比如"A2 在 L1 上，2 s 后与 EGO 距离 6 m，有冲突"，而不是让教师模型事后编一段解释。

**4. 预测和规划联合建模**
- 现在的预测器在预测其他车时并不知道自车打算怎么开，但交互是双向的：自车的选择也会影响别人。
- 做法：先生成几条候选自车轨迹，让 GNN 分别预测其他车在每种情况下的反应，再由 LLM 或打分器挑出最好的一条。这是图最擅长的"假如这样开会怎样"式推理。

## 三、建议的实验顺序

| 优先级 | 实验 | 目的 | 成本 |
|---|---|---|---|
| **P1** | **纯 GNN 规划器**：预测器原本就能预测自车 3 s，把它扩展到 6 s 规划，不用 LLM | 先确定"只用图"能做到多好。如果接近或超过 1.776，说明 LLM 目前的贡献很小 | 约 1 小时，单卡 |
| **P2** | **LLM 决策 + 图解码器**：用自动推出的决策标签训练 LLM，GNN 解码器按决策生成轨迹 | 本方案的核心：让 LLM 和图各做擅长的事，并检验逐节点图 token 在"指向节点"时是否有用 | 约 1 天开发 + 3 小时训练 |
| P3 | **用图 token 替代长文本**：去掉坐标文本，只保留节点名 + 节点 token | 检验精度不变的前提下，输入 token 能减少多少（现在提示约 1500 个 token） | 约 3 小时 |
| P4 | **评测改进**：多模态输出（minADE）、交互场景子集、闭环评测（nuPlan 或 interPlan） | 开环 L2 主要奖励"模仿"，体现不出交互推理的价值 | 闭环需要另申请 nuPlan 数据 |

我建议先跑 P1。只要 1 小时，它的结果会决定 P2 怎么设计：
- 如果纯 GNN 规划器就已经很强，P2 的重点放在"LLM 决策能不能在难预测和交互场景里再进一步"；
- 如果它比较弱，说明 LLM 在融合信息上确实有价值，P2 就保留 LLM 作为主体。

要不要现在开始跑 P1？

---

### **User**  <sub>2026-11-26 08:48 UTC</sub>

你把结论和建议保存在log中，push to github，我明天再来跑，推出后服务器会被清理，所以你需要上传这些

---

### **Claude**  <sub>2026-11-26 08:48 UTC</sub>

服务器会被清空，所以除了结论和建议，明天还要用的东西也得一起传上去。先看看有哪些文件、各多大：

---

### **User**  <sub>2026-11-26 08:49 UTC</sub>

你把结论和建议保存在log中，push to github，我明天再来跑，推出后服务器会被清理，所以你需要上传这些（数据不用上传，写好下载提示/代码即可，checkpoint挑选有用未来可能需要的上传）

---

### **Claude**  <sub>2026-11-26 08:49 UTC</sub>

The CoT labels are keyed by scenario ID, so tomorrow's download must fetch exactly the same 22,000 scenarios. I'll check whether the download script picks them deterministically:

---

### **Claude**  <sub>2026-11-26 08:49 UTC</sub>

S3 lists keys in lexicographic order, so in principle the selection is deterministic. To be safe, I'll still save the exact scenario ID lists and add an `--ids` option to the download script:

---

### **Claude**  <sub>2026-11-26 08:50 UTC</sub>

The S3 listing reproduces the saved ID list exactly. Now the restore script, which rebuilds the environment and all derived data on a fresh server:

---

### **Claude**  <sub>2026-11-26 08:50 UTC</sub>

Before writing the handoff doc, one cheap check that informs P1: the existing GNN predictor already predicts the ego vehicle's next 3 s. How does its ADE@3s compare with the LLM's?

---

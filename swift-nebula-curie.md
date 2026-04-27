# Base Placement Optimization (BPO) 方案 v5

> **最新方案**：段内 base 可动的 Jacobian SQP 优化，通过 L1/Huber 稀疏惩罚实现"base 尽量不动，除非有巨大增益"。
>
> v4 → v5 核心变化：base 从"段内固定一个点"变为"段内逐点可动的轨迹"，通过稀疏惩罚保证大部分时间 base 不动。一个 λ_base 参数统一固定/可动/在线全谱系场景。
>
> **v5 修订（QP 可行性修正）**：
> 1. EE 等式约束软化为松弛形式 (J·δ + s = Δx)，避免初始大残差下 QP 不可行
> 2. 引入 λ_slack 自适应调度，前期小（保证可行）后期大（收敛到零残差）
> 3. arm 连续性约束作用在最终轨迹 (qᵢ+δqᵢ) 而非更新步 (δqᵢ)
> 4. Stage 1 IK 容差从 5cm/10° 收紧到 1cm/1°，允许部分失败
> 5. 显式归一化权重矩阵 Wx 处理平移/旋转的量纲差异
>
> **v5 修订（李群修正）**：
> 6. SE(2)/SE(3) 相邻 base 差使用李代数差 ξ = log(bᵢ₋₁⁻¹·bᵢ)，不用欧氏减法
> 7. 所有 Huber 惩罚 / base 速度 / base 加速度约束作用在 ξ 上（含 wrap_to_pi）
> 8. bᵢ 作为李群元素存储，δbᵢ 作为李代数向量，更新用 retraction (exp)
> 9. Stage 1 相邻 waypoint 插值用测地线，不用欧氏线性插值

> **v5 修订（参考系约定锁定 / 方案 A）**：
> 10. 全程采用 body-frame twist + right-retraction：δb 在 base body frame，
>     Δx 在 EE body frame（见 2.1.1 节约定表）
> 11. J_base = Ad(T_eb) = Ad(T_be⁻¹)，而非原 v4 暗示的 Ad(T_be)（方向反）
> 12. Δxᵢ 计算锁定为 log(T_we,current⁻¹ · T_we,target)
> 13. pinocchio jacobian 统一用 `ReferenceFrame.LOCAL`

> **v5 修订（架构分层，为 v6-online 留接口）**：
> 14. 明确 v5 定位为"离线全局 QP"，流式/MPC 作为 v6-online 独立分支（第 14 节）
> 15. 引入三层架构：底层（FK/Jacobian/李群）、中层（LocalQPBuilder + cost 插件）、
>     上层（Solver 外循环），前两层 v5/v6 共用（见 14.0.2 节）
> 16. `qp_formulator` 接口接受 `waypoint_indices`/`frozen_indices`，不假设全局

> **v5 修订（Huber QP 展开）**：
> 17. Huber 采用**分量级**形式（重构 B），每分量独立 δⱼ，纯 QP 可解（见 2.4.1）
> 18. 引入辅助变量 u/v/t 把 Huber 展开为"目标二次 + 约束线性"标准 QP
> 19. 诚实标注 v5 覆盖范围：**仅稀疏档 + 固定档**，平滑档/自由档交给 v6-online
> 20. YAML 新增 `base_sparsity` 配置块，预留 `target`/`norm_type` 接口

---

## 方案演进记录

- **v1**: 全量 IRM 离散交集 + 逐点 IK 验证
- **v2**: 轨迹条件化 IRM + coarse-to-fine + 连续 IK 验证（采纳 Codex 反馈）
- **v3**: 增加 Stage 0 预处理、NMS 去重、多分支 q₀、自适应加密、strict/best_effort 语义（采纳 Codex+Gemini 反馈）
- **v4**: 引入联合 Jacobian QP 迭代作为核心求解器，假设段内 base 固定
- **v5**: base 段内可动（逐点变量），L1/Huber 稀疏惩罚，统一量纲处理，λ_base 统一场景谱系
- **v5 修订（QP 可行性）**: λ_slack 松弛变量解决 QP 可行性问题，归一化权重矩阵精细化
- **v5 修订（李群）**: SE(2)/SE(3) 李群操作规范化，相邻差用李代数 ξ，retraction 更新
- **v5 修订（参考系约定）**: 锁定 body-frame twist + right-retraction（方案 A），
  修正 J_base = Ad(T_eb)，Δx 用 EE body frame twist
- **v5 修订（架构分层）**: 三层架构（底层 Jacobian/李群 + 中层 LocalQPBuilder
  + 上层 Solver），v5 与未来 v6-online 共享底两层（见 14 节）
- **v5 修订（Huber QP 展开，当前）**: Huber 改为分量级（重构 B），纯 QP 可解；
  诚实标注 v5 仅覆盖稀疏档 + 固定档，平滑/自由档交给 v6-online

---

## 1. v5 核心问题定义

**v5 的定位（边界声明）**：
- **离线、全局 QP**。输入是**完整 EE 轨迹** {T_ee_1, ..., T_ee_N}（一次性给齐）
- 一次 SQP 求解，每次 QP 是 N-waypoint 联合问题
- **不处理流式输入 / 滚动 horizon / MPC 场景**（见第 14 节"未来目标"）
- Huber 稀疏性能成立**依赖全局视野**，流式/单点 QP 下会退化

这个定位在方案的数学假设和 Stage 流程中是基础前提。如果未来需要在线跟踪，
应另开分支（v6-online / MPC 范式），不要强行在 v5 上改造。

给定 EE 轨迹 {T_ee_1, ..., T_ee_N} 和机器人 URDF，求解：
- **arm 关节轨迹** {q₁, ..., qₙ}
- **base 轨迹** {b₁, ..., bₙ}

使 EE 精度满足，同时 **base 尽量不动**，除非移动 base 能显著改善关节状态（远离限位、远离奇异）。

### 1.1 为什么选择 Jacobian SQP 而非统一链 IK

曾经对比过两种路径：
- 路径 A（Jacobian SQP）：base/arm 区分，联合优化
- 路径 B（统一链 IK + 后处理）：base 当虚拟关节，逐点 IK，关节空间后处理

结论：**对于"base 尽量不动"的需求，Jacobian SQP 天然更合适**，因为：

1. **base 平滑性是 Jacobian 方法的内在性质**：SQP 更新是沿梯度的连续小步，不需后处理
2. **L1/Huber 稀疏惩罚可以直接在 QP 中建模**：IK 采样天然无法产生稀疏结构
3. **多 waypoint 全局耦合**：SQP 一次优化所有 waypoint，IK 逐点独立
4. **EE 精度硬约束**：SQP 中的等式约束严格保证，IK 后处理平滑会破坏 EE 精度
5. **速度/加速度约束**：QP 线性不等式精确满足，防抖动天然有保障

IK 并行方案仍可用于 Stage 1 的初始点生成（快速筛选可行 base 区域），但核心优化用 SQP。

### 1.2 λ_base 场景谱系与 v5 覆盖范围

| λ_base 值 | 行为 | 对应场景 | v5 覆盖 |
|-----------|------|---------|---------|
| ∞ | base 完全不动 | 固定底盘操作（退化为 v4） | ✅ |
| 大（~100） | 只在关节接近极限时动 base | **v5 默认模式**（稀疏档） | ✅ |
| 中（~10） | base 平滑跟随 | 在线跟踪 | ❌ 交给 v6-online |
| 小（~1） | base 自由移动 | 全身协调运动 | ⚠️ 名义覆盖但未验证 |
| 0 | base 和 arm 无区别 | 纯运动链 IK | ⚠️ 名义覆盖但退化 |

**v5 的诚实边界**：
- v5 主打**稀疏档（大 λ_base）和固定档**，算法设计与之匹配
- **平滑档（中 λ_base）在 v5 下会退化**：原因是 Huber 作用在相邻差 `ξᵢ` 上，
  产生的是"跳变事件稀疏"而非"平滑缓变"。想要平滑跟随应把 Huber 放在
  `ξᵢ - ξᵢ₋₁`（base 加速度）上，这需要不同的 cost 插件配置
- **平滑档/自由档的完整支持属于 v6-online 的职责**（见 14.0 / 14.1 节）
- v5 的 cost 插件 `BaseSparsityCost` 预留 `target="xi" | "delta_xi"` 接口，
  但 v5 默认固定为 `target="xi"`，不承诺其他模式的数值稳定性

"一个 λ_base 参数统一所有场景"是 v5 设计**灵感**，但工程上 v5 仅实现稀疏档。

---

## 2. 核心数学

### 2.1 运动学方程（逐点）

```
Δxᵢ = J_arm(qᵢ) · δqᵢ + J_base(qᵢ, bᵢ) · δbᵢ
```

与 v4 相同，但 δbᵢ 为逐点独立变量。

#### 2.1.1 参考系约定（全局锁定：body frame / 方案 A）

v5 全程采用 **body-frame twist + right-retraction** 约定。记账上统一到一处，
避免 J_base、retraction、Δx、pinocchio jacobian 之间错配：

| 对象 | 含义 | frame |
|------|------|-------|
| T_wb | base 位姿（world→base 的齐次变换） | — (是矩阵, 不是 twist) |
| T_be | EE 相对 base 的位姿（由 arm FK 给出） | — |
| T_we = T_wb · T_be | EE 相对 world 的位姿 | — |
| δbᵢ ∈ 𝔰𝔢(3)/𝔰𝔢(2) | base 的扰动 twist | **base body frame** |
| Δxᵢ ∈ 𝔰𝔢(3) | EE 当前→目标的残差 twist | **EE body frame** |
| ξᵢ = log(bᵢ₋₁⁻¹·bᵢ) | 相邻 base 的李代数差 | **base body frame** |
| J_arm | arm 对 EE 的 jacobian | **LOCAL** (pinocchio) |
| J_base | base 对 EE 的 jacobian | body→body |

**更新规则（retraction）**：

```
T_wb ← T_wb · Exp(α · δb)            # 右乘，body-frame twist
q    ← q + α · δq                     # arm 关节在 ℝⁿ（连续 revolute 需 wrap）
```

**残差计算**：

```
Δxᵢ = log_SE3(T_we,current⁻¹ · T_we,target)       # EE body-frame twist
      （注意是 current⁻¹·target，得到 "从 current 看 target" 的 body twist）
```

**pinocchio 对应接口**：

```python
pin.computeJointJacobians(model, data, q)
J_arm_body = pin.getFrameJacobian(model, data, ee_frame_id,
                                  pin.ReferenceFrame.LOCAL)   # 6×n_arm
```

**物理不变量**：无论选 body 还是 spatial，最优 `T_wb^*` 和 `q^*` 轨迹相同；
选 body 纯粹是工程理由（底盘控制接口、非完整约束的线性性、与 retraction 一致）。

### 2.2 QP 子问题（SQP 每次迭代求解）

#### 2.2.1 问题公式（含松弛变量 + 李群规范化）

**先定义相邻 base 的李代数差**（在当前工作点）：

```
b̂ᵢ = bᵢ · Exp(δbᵢ)                                    # 下一步的 base (retraction)
ξᵢ = log(b̂ᵢ₋₁⁻¹ · b̂ᵢ)                                  # 李代数中的相邻差

在当前迭代点做一阶 Taylor 线性化:
ξᵢ ≈ ξᵢ^{(k)} + (δbᵢ - δbᵢ₋₁)                         # (简化形式)

其中 ξᵢ^{(k)} = log(bᵢ₋₁^{(k)}⁻¹ · bᵢ^{(k)}) 是当前工作点上的"静态"李代数差
```

(严格形式带右/左雅可比 Jr⁻¹，线性化后仍为 δb 的仿射函数，见 2.2.4 节)

**QP 公式**：

```
变量: {δqᵢ}, {δbᵢ}, {sᵢ} (松弛变量, 对每个 waypoint 的 EE 等式)

min  Σᵢ [ λ_margin   · margin_cost(qᵢ + δqᵢ)                    // 关节余量
        + λ_smooth_q · ‖δqᵢ - δqᵢ₋₁‖²_Wq                        // arm 平滑
        + λ_base     · Huber_δ(ξᵢ)_Wb                            // base 稀疏移动
                                                                 // (作用于李代数差)
        + λ_slack    · ‖sᵢ‖²_Wx ]                                // EE 残差松弛

s.t. J_arm_i · δqᵢ + J_base_i · δbᵢ + sᵢ = Δxᵢ                  ∀i  (EE 软约束)
     q_min ≤ qᵢ + δqᵢ ≤ q_max                                    ∀i  (关节限位)
     ‖(qᵢ+δqᵢ) - (qᵢ₋₁+δqᵢ₋₁)‖∞ ≤ Δq_max                        ∀i  (arm 轨迹连续性)
     δbᵢ^T Wb δbᵢ ≤ step_limit²                                  ∀i  (信赖域,
                                                                       在李代数空间)
     ‖ξᵢ‖_Wb / Δtᵢ ≤ v_base_max                                  ∀i  (base 速度上限)
     ‖(ξᵢ - ξᵢ₋₁)/Δtᵢ‖_Wb ≤ a_base_max                           ∀i  (base 加速度)
```

注意所有 base 相邻差都通过 ξᵢ 表达，**没有任何地方出现 `bᵢ - bᵢ₋₁` 的欧氏减法**。

**Huber 入参的仿射性（QP 可行的前提）**：

在当次 QP 迭代中，`ξᵢ^{(k)}` 是由**上一步工作点 {bᵢ^{(k)}}** 决定的**常数**，
变量只有 `δbᵢ` 和 `δbᵢ₋₁`。于是 Huber 的入参是 δ 的**仿射函数**：

```
Huber 入参 = ξᵢ^{(k)} + (δbᵢ - δbᵢ₋₁)
             └─────┘   └──────────┘
              常数       δ 的线性组合
```

这一仿射结构是把 Huber 展开为标准 QP（见 2.4.1）的前提。如果入参对 δ 非线性，
则无法写成纯 QP，需要 SOCP 或 NLP。

Huber 本身不是二次型（是分段函数），不能直接塞给 OSQP/qpOASES。具体展开见 2.4.1 节。

#### 2.2.2 松弛变量 sᵢ 的意义

原本的 EE 硬等式约束 `J · δ = Δx` 在初始迭代时会造成 QP 不可行：

- Stage 1 的初始 IK seed 只满足 1cm/1° 的容差，Δxᵢ 范数仍可能较大
- 要一步内通过 Jacobian 一阶近似消灭这么大残差，所需 δq 超出信赖域和关节限位
- 等式约束仿射子空间 × 信赖域 × 限位不等式 → 可行域极可能为空

解决方案：引入松弛变量 sᵢ，EE 约束软化为 `J · δ + s = Δx`。含义：
- 允许本次迭代**不完全消灭残差**，剩余 sᵢ 推迟到下一次 SQP 迭代
- QP 永远可行（至少 δq=δb=0, s=Δx 是可行解）
- λ_slack 控制"硬 vs 软"：越大越接近硬约束
- λ_slack 自适应调度（见 2.3）保证最终收敛到 ‖sᵢ‖ → 0

#### 2.2.3 arm 连续性约束的修正

原 v5 约束 `‖δqᵢ - δqᵢ₋₁‖∞ ≤ Δq_max` 作用在"更新量的差异"上，这是错误的：
- 真正需要保证的是**最终轨迹 {qᵢ}** 连续，不是迭代更新 {δqᵢ} 连续
- 当相邻 waypoint 残差方向不同时（一个向左修一个向右修），原约束会强行拉平 δqᵢ，
  造成残差无法消除

修正：约束作用在更新后的轨迹上：

```
‖(qᵢ + δqᵢ) - (qᵢ₋₁ + δqᵢ₋₁)‖∞ ≤ Δq_max
= ‖(qᵢ - qᵢ₋₁) + (δqᵢ - δqᵢ₋₁)‖∞ ≤ Δq_max
```

(arm 关节是 ℝⁿ 空间的线性关节，直接减法正确；连续 revolute 关节若跨越 ±π 需单独 wrap)

#### 2.2.4 李群操作规范（SE(2)/SE(3)）

SE(2)/SE(3) 是李群而非欧氏空间，相邻位姿的"差"必须通过李代数（tangent space）处理。

**状态表示**：

```
bᵢ ∈ SE(2) 或 SE(3)                      # 李群元素（齐次矩阵 / (t, R) 对）
δbᵢ ∈ 𝔰𝔢(2) = ℝ³ 或 𝔰𝔢(3) = ℝ⁶          # 李代数向量（QP 决策变量）
```

**核心操作**：

```
Exp: 𝔰𝔢 → SE         指数映射（李代数 → 李群）
log: SE → 𝔰𝔢         对数映射（李群 → 李代数）
∘: SE × SE → SE      李群乘法（位姿合成）
Ad(T): 𝔰𝔢 → 𝔰𝔢      伴随表示（twist 在参考系间变换）
```

**状态更新（retraction）**：

```
bᵢ ← bᵢ · Exp(α · δbᵢ)        # body-frame twist, 右乘更新
                                # 与 J_base = Ad(T^base_ee) 的 body-frame 约定一致

（约定：v5 全程使用 body-frame twist / 右乘 retraction）
```

**相邻差（tangent difference）**：

```
ξᵢ = log(bᵢ₋₁⁻¹ · bᵢ)  ∈ 𝔰𝔢

SE(2) 具体形式:
  Δθ = wrap_to_pi(θᵢ - θᵢ₋₁)                          # 必须 wrap!
  [Δx_local; Δy_local] = R(-θᵢ₋₁) · [xᵢ-xᵢ₋₁; yᵢ-yᵢ₋₁]
  (v_x, v_y) = V⁻¹(Δθ) · (Δx_local, Δy_local)         # V 是 SE(2) 左雅可比
  ξᵢ = (v_x, v_y, Δθ)

  (小角度近似: ξᵢ ≈ (Δx_local, Δy_local, Δθ), 足够精确)

SE(3) 具体形式:
  ΔT = Tᵢ₋₁⁻¹ · Tᵢ
  ξᵢ = log_SE3(ΔT)           # 使用 pinocchio::log6 或等价实现
```

**线性化（QP 中的处理）**：

QP 目标/约束需要 δb 的仿射函数。`ξᵢ` 本身是 δb 的非线性函数，在当前工作点做一阶展开：

```
简化线性化（小步长下有效）:
  ξᵢ ≈ ξᵢ^{(k)} + (δbᵢ - Ad_{ΔT^{-1}} · δbᵢ₋₁)
  
  或更简化（忽略伴随变换）:
  ξᵢ ≈ ξᵢ^{(k)} + (δbᵢ - δbᵢ₋₁)

严格线性化（大步长下更精确）:
  ξᵢ ≈ ξᵢ^{(k)} + Jr⁻¹(ξᵢ^{(k)}) · δbᵢ - Jr⁻¹(-ξᵢ^{(k)}) · Ad(ΔT^{-1}) · δbᵢ₋₁
  (Jr 为 SE(3) 右雅可比)
```

v5 默认用简化线性化（信赖域已限制 δb 小量，高阶误差可忽略）；
若数值不稳可切换严格形式（待定项 19）。

**数值稳定性**：

- log_SE3 在恒等变换附近（ξ ≈ 0）需 Taylor 展开处理 0/0（pinocchio 已内建）
- log_SE3 在旋转 π 附近有轴向二义性（v5 中相邻 b 差不会到 π，实际不遇到）
- Huber 梯度在 ξ=0 处光滑（小于 δ 的二次区间），不会因 log 奇异性受影响

**插值（Stage 1 和轨迹重采样用）**：

```
对 t ∈ [0, 1] 从 b_A 到 b_B 的测地线插值:
  ξ = log(b_A⁻¹ · b_B)
  b(t) = b_A · Exp(t · ξ)

禁止: 对角度/姿态做欧氏线性插值
```

### 2.3 λ_slack 的自适应调度

**目标**：前期小（QP 可行且稳定），后期大（残差真正收敛到 0）。

#### 2.3.1 调度策略（基于残差下降的自适应）

```python
# 初始化
lambda_slack = 100       # 中等起点
lambda_min   = 10
lambda_max   = 1e7
beta_up      = 3         # 增长系数
beta_down    = 2         # 衰减系数
threshold    = 0.8       # 相对下降阈值 (下降 > 20% 算"有进展")
epsilon_tight = 1e-3     # 切换到 lambda_max 的残差阈值
max_s_prev   = +∞

# 每次 SQP 迭代后更新
max_s = max_i ‖sᵢ‖_Wx

if max_s < epsilon_tight:
    # 残差已很小，切入精细阶段，硬追剩余残差
    lambda_slack = lambda_max

elif max_s < threshold * max_s_prev:
    # 残差显著下降，逼近硬约束
    lambda_slack = min(lambda_max, beta_up * lambda_slack)

elif max_s > max_s_prev * 1.1:
    # 残差上升（数值异常或病态），放松以恢复稳定
    lambda_slack = max(lambda_min, lambda_slack / beta_down)

# else 残差停滞: 保持 lambda_slack 不变

max_s_prev = max_s
```

#### 2.3.2 收敛判定

```
converged = (max_i ‖Δxᵢ‖_Wx < ε_pos_rot)       AND
            (max_i ‖sᵢ‖_Wx    < ε_slack)         // 防止"假收敛"
```

若只看 Δx 不看 s，会出现"QP 声称 Δx 小，但实际残差被推给 s"的假象。必须两者都满足。

#### 2.3.3 与 line search / 信赖域的协调

- 若启用 line search：λ_slack 作为 merit function 的权重
- 信赖域 step_limit 不变（控制 δb 的幅度）
- Huber 的 δ 参数应 ≥ step_limit：防止信赖域内 Huber 跨越线性区间导致行为不稳

### 2.4 Huber 惩罚（分量级，重构 B）

**设计决定**：v5 采用**分量级 Huber**（每个 ξ 分量独立判断线性/二次区），
而非**整体 L2 Huber**。原因：

1. **纯 QP 兼容**：分量级可展开为 OSQP/qpOASES 标准形式（见 2.4.1）；
   整体 L2 Huber 需要 SOCP 求解器（ECOS/Mosek），增加依赖
2. **物理合理**：SE(2) 底盘的三个轴（x, y, θ）物理上独立驱动，"每轴独立决定
   是否移动"比"整体移动与否"更符合底盘控制结构
3. **稀疏性更强**：分量级 Huber 的稀疏结构是"每个轴独立触发线性区"，
   整体 L2 Huber 只能判断"整体向量是否越界"

**公式（分量级）**：

```
Huber_v5(ξᵢ) = Σⱼ h_{δⱼ}(ξᵢⱼ)

每个分量 j 独立的标量 Huber:
  h_{δⱼ}(x) = { x² / (2δⱼ),        |x| ≤ δⱼ       (二次, 近似免费)
              { |x| - δⱼ/2,          |x| > δⱼ       (线性, 稀疏惩罚)

分量阈值 δⱼ (在归一化空间下):
  平移分量: δ_v  (默认 0.02/L)   # 对应原始约 1cm (L=0.5m 时)
  旋转分量: δ_ω  (默认 0.02/φ)   # 对应原始约 1.15° (φ=1 时)
```

**性质**：
- 作用对象是**李代数差 ξᵢ = log(bᵢ₋₁⁻¹·bᵢ)** 的各分量，自动处理角度 wrap
- 每个 δⱼ 必须 ≥ 信赖域 step_limit 对应的分量，防止单步跨越线性区
- 产生"**base 大部分时间各分量 ξᵢⱼ ≈ 0，必要时某个轴一次到位**"的分段常数轨迹
- ξᵢ 在 QP 中用线性化形式（见 2.2.1 / 2.2.4）
- Wb 退化为每分量 δⱼ 的对角权重（见 2.5.3）

**备注：未采用的替代方案**：

- **整体 L2 Huber**（重构 A）：`h_δ(ξᵢ) = ½ξᵀMξ/δ (‖ξ‖≤δ) / ‖ξ‖_M - δ/2`。
  需要 SOCP：`‖M^{1/2}v‖₂ ≤ t`。v5 不采用（不引入新求解器依赖），但 cost 插件
  `BaseSparsityCost` 接口保留 `norm_type` 参数，未来可扩展
- **L1 惩罚**：`‖ξᵢ‖₁`。更严格的稀疏性，但原点非光滑，优化数值性质差
- **L2 惩罚**：`‖ξᵢ‖²`。无稀疏性，退化为 Tikhonov 正则化，不符合设计目标

#### 2.4.1 Huber 的 QP 标准形式展开

分段 Huber 本身不是二次型，必须通过辅助变量改写为纯 QP（目标二次 + 约束线性）。
对每个 waypoint i（从 i=1 起）和每个分量 j，引入三个辅助变量：

```
uᵢⱼ  ∈ ℝ        # 二次区分量 (|uᵢⱼ| ≤ δⱼ)
vᵢⱼ  ∈ ℝ        # 线性区分量 (|vᵢⱼ| 部分)
tᵢⱼ  ∈ ℝ≥0     # |vᵢⱼ| 的上界 (epigraph)
```

**分解等式**（把 Huber 入参拆成 u + v）：

```
uᵢⱼ + vᵢⱼ = ξᵢⱼ^{(k)} + (δbᵢⱼ - δbᵢ₋₁,ⱼ)

展开为 QP 线性等式约束:
  uᵢⱼ + vᵢⱼ - δbᵢⱼ + δbᵢ₋₁,ⱼ = ξᵢⱼ^{(k)}      ← 常数
```

**盒约束**（限制 u 在二次区范围内）：

```
-δⱼ ≤ uᵢⱼ ≤ δⱼ
```

**L1 上界约束**（tᵢⱼ ≥ |vᵢⱼ|）：

```
vᵢⱼ ≤ tᵢⱼ
-vᵢⱼ ≤ tᵢⱼ
```
（tᵢⱼ ≥ 0 由这两条隐含，无需额外声明）

**目标函数增量**（加到 QP 整体目标里）：

```
λ_base · Σᵢⱼ [ uᵢⱼ² / (2δⱼ)  +  tᵢⱼ ]
         └────────────┘     └──┘
            二次项            线性项
```

**正确性验证**：QP 最优解在每个 `(i, j)` 上自动满足：
- 若 `|ξᵢⱼ^{(k)} + (δbᵢⱼ - δbᵢ₋₁,ⱼ)| ≤ δⱼ`：最优 `vᵢⱼ = 0`, `tᵢⱼ = 0`,
  `uᵢⱼ = 入参`, 目标贡献 = `uᵢⱼ² / (2δⱼ)`（二次区 Huber）✓
- 若 `|...| > δⱼ`：最优 `uᵢⱼ = ±δⱼ` (饱和), `vᵢⱼ = 入参 ∓ δⱼ`,
  `tᵢⱼ = |vᵢⱼ|`, 目标贡献 = `δⱼ/2 + |vᵢⱼ|` = `|入参| - δⱼ/2`（线性区 Huber）✓

**i = 0 的边界处理**：

`ξ_0 = log(b_{-1}⁻¹ · b_0)` 没有定义（无 b_{-1}）。v5 的约定：
- Huber 索引范围：**i = 1, 2, ..., N-1**（跳过 i=0）
- i=0 是起点，没有"移动"概念，不参与稀疏惩罚
- 同理，base 加速度约束 `‖(ξᵢ - ξᵢ₋₁)/Δtᵢ‖` 索引范围 i = 2, ..., N-1

**维度估算（SE(2) + 100 waypoints）**：

```
分量数 n_b = 3 (SE(2))  或  6 (SE(3))
Huber waypoint 数: N-1 = 99

辅助变量:
  u: 99 × 3 = 297
  v: 99 × 3 = 297
  t: 99 × 3 = 297
  合计: 891 ≈ 900

辅助约束:
  分解等式: 99 × 3 = 297
  u 盒约束: 99 × 3 × 2 = 594
  t 上下界: 99 × 3 × 2 = 594
  合计: 1485 ≈ 1500
```

SE(3) 则 1782 变量 + 2970 约束。

**OSQP 兼容性**：
- 目标 P 矩阵：对角块（u 的二次项），稀疏度极高
- 约束矩阵 A：稀疏，每行只有 2~4 个非零（δb, u, v 之间的耦合）
- OSQP 的 ADMM + sparse factorization 对这类结构处理高效，单次 QP 求解预期毫秒级

**cost 插件接口（为未来 v6-online 或重构 A 扩展预留）**：

```python
class BaseSparsityCost:
    def __init__(self,
                 lambda_base: float,
                 deltas: Dict[str, float],     # 每分量 δⱼ (归一化空间)
                 norm_type: str = "component", # "component" | "l2_socp" (未来)
                 target: str = "xi"):          # "xi" | "delta_xi" (v6-online)
        ...
```

### 2.5 量纲处理（归一化）

v5 的所有变量（q, b）、约束（EE 残差 Δx）、松弛变量（s）都必须在**统一归一化空间**下处理，
否则 QP 的目标函数/约束会混合米和弧度，失去物理意义。

#### 2.5.1 决策变量的归一化

虚拟关节归一化到 [-1, 1]：

```
q_base_normalized = [x/L, y/L, yaw/π]                         # SE(2)
q_base_normalized = [x/L, y/L, z/L, rx/π, ry/π, rz/π]         # SE(3)
```

L = 特征长度（arm 有效工作半径，~0.5m）。

#### 2.5.2 EE 残差 / 松弛变量的归一化（Wx）

Δxᵢ 和 sᵢ 都是 6D twist，平移/旋转量纲不同。用对角权重 Wx 统一到无量纲：

```
Wx = diag(1/L², 1/L², 1/L², 1/φ², 1/φ², 1/φ²)

其中:
  L = 特征长度 (arm 工作半径, 默认 0.5m)
  φ = 特征旋转 (常用 π/2 或 1 rad)

归一化的范数:  ‖s‖²_Wx = s^T · Wx · s
等价于把 s 按 [m/L, m/L, m/L, rad/φ, rad/φ, rad/φ] 重构后求平方和
```

Wx 让 λ_slack 成为**单一可解释参数**，不再需要手工平衡位置误差和姿态误差的相对代价。
如果用户需要偏好位置精度，可在 Wx 中加权。

#### 2.5.3 base 移动代价的归一化（分量 δⱼ 形式）

分量级 Huber（2.4 节）不再用整体 Wb 矩阵加权，而是**每个分量独立指定 δⱼ**。
在归一化空间下，分量 δⱼ 的默认值：

```
归一化前的原始阈值:
  δ_v_raw = 0.02 m       # 约 2cm 的 "免费" 平移
  δ_ω_raw = 0.02 rad     # 约 1.15° 的 "免费" 旋转

归一化空间下的分量阈值:
  SE(2):  δ = (δ_v/L,  δ_v/L,  δ_ω/φ)        ∈ ℝ³
          = (0.04, 0.04, 0.02)  典型值 (L=0.5, φ=1)
  SE(3):  δ = (δ_v/L, δ_v/L, δ_v/L,
               δ_ω/φ, δ_ω/φ, δ_ω/φ)           ∈ ℝ⁶

其中:
  L = 特征长度 (2.5.1, 默认 0.5m)
  φ = 特征旋转 (默认 1 rad)
```

**base 速度/加速度约束仍用整体范数（向量形式）**，但范数也按分量归一化：

```
base 速度约束:  ‖ξᵢ‖_Wb / Δtᵢ ≤ v_base_max         (Wb 仍是对角矩阵)
base 加速度:   ‖(ξᵢ - ξᵢ₋₁)/Δtᵢ‖_Wb ≤ a_base_max

Wb 在这里仍用:
  Wb = diag(1, 1, γ)          # SE(2)
  Wb = diag(1, 1, 1, γ, γ, γ)  # SE(3)
  γ = 旋转 vs 平移代价比 (默认 1.0)
```

**两套参数的分工**：
- **δⱼ**（Huber 分量阈值）：决定"多小的 ξᵢⱼ 算免费"，控制稀疏性粒度
- **Wb + v/a_max**（速度/加速度约束）：决定底盘物理能力上限，硬约束

**调参直觉**：
- 默认 δⱼ 对所有分量"同等重要" → 如果某机器人平移远比旋转重要，单独减小 δ_v
  让平移的稀疏触发更严
- 默认 Wb=I, γ=1 → 如果某机器人旋转速度限制严格（例如履带转弯慢），
  增大 Wb 的旋转对角元使同样的 ‖ξ‖ 对应更小的允许 ξ_ω

**注意 δⱼ 与 step_limit 的关系**：δⱼ ≥ step_limit 对应分量上限（2.3.3 节），
否则一步可能跨越 Huber 线性区间，优化数值不稳。

#### 2.5.4 归一化的整体效果

- 所有变量无量纲（q_arm, q_base_norm, δ 系列, s 都在 O(1) 量级）
- Jacobian 条件数改善
- 超参数 λ_* 和权重 W_* 各司其职：W 处理量纲 + 偏好，λ 仅表达项间相对重要性
- 调参更直观：λ_slack=100 比 "用 10m 补 0.01rad 的奇怪权衡" 更有意义

### 2.6 QP 维度估算（SE(2) + 7-DOF + 100 waypoints）

```
变量：
  δb_1..δb_100  (3D)            300 维
  δq_1..δq_100  (7D)            700 维
  s_1..s_100    (6D 松弛)        600 维
  Huber 辅助: u_ij              297 维   (99 × 3)
  Huber 辅助: v_ij              297 维
  Huber 辅助: t_ij              297 维
  总计                          ~2491 维

约束：
  EE 软等式 (J·δ+s=Δx)          600 条
  关节限位                      1400 条
  arm 轨迹连续性                693 条
  信赖域                        100 条
  base 速度上限                 99 条
  base 加速度上限               98 条
  Huber 分解等式 (u+v=ξ+δb diff)  297 条
  Huber u 盒约束 (|u|≤δⱼ)        594 条
  Huber t 上下界 (t≥±v)          594 条
  总计                          ~4475 条
```

**SE(3) 场景**（分量 3 → 6）：Huber 辅助变量增至 ~1800 维，约束 ~3000 条，
总规模约 4000 变量 + 6000 约束。

OSQP/qpOASES 对此规模仍为毫秒~十毫秒级。关键优势：
- 目标 P 矩阵块对角（δq 光滑项 + u 的二次项），**极度稀疏**
- 约束矩阵 A 每行只有 2~4 个非零，**极度稀疏**
- Huber 展开带来的额外维度在 OSQP 的 ADMM 框架下几乎零成本

**与原估算（~1900 变量 + ~3600 约束）的差异**：v5 李群修订前的粗略估算
把"Huber 辅助变量"记为 ~300 维（整体级），重构 B 分量展开后是 ~900 维
（u/v/t 三套），增加了约 600 变量 + 900 约束，仍在求解器轻松范围内。

---

## 3. 算法流程

```
Stage 0: 预处理
  ├── 构建运动学模型（URDF → pinocchio）
  ├── 读取 YAML 配置（base DOF, SE(2)/SE(3), 限位, λ_base, δ, L, ...）
  ├── 轨迹重采样（保留关键点，自适应加密大变化处）
  │   - 相邻 T_ee 的差距用李代数范数 ‖log(Tᵢ₋₁⁻¹·Tᵢ)‖ 度量
  │   - 插值用测地线 (Tₐ · Exp(t·ξ))，不用欧氏插值
  ├── 搜索空间裁剪（base 范围 = EE 包围盒 ± max_reach）
  └── 关节归一化（含 base 李代数向量的分量归一化）

Stage 1: 初始点生成
  ├── 方法 1（启发式，必选）: 5~10 个 b₀ 候选
  │   - 轨迹质心正下方
  │   - 轨迹首/末端点附近
  │   - 轨迹质心 ± R_max/2 偏移
  ├── 方法 2（IRM 辅助，可选）: 10~50 个 b₀ 候选
  │   - IRM 投票 top-M → NMS 去重（NMS 距离度量用李代数范数）
  ├── 方法 3（并行 IK 筛选，可选）:
  │   - 用统一链 IK 快速评估每个 b₀ 的可行性
  │   - CuRobo 风格的 GPU 并行（若可用）
  │
  └── 对每个 b₀，做精确连续 IK 获取初始 {qᵢ}
      - bᵢ 初始全部设为 b₀（从固定 base 出发）
      - 精确容差（1cm / 1° 量级），而非原 5cm/10°
      - 前解做后 seed
      - 允许部分 waypoint 失败：记录失败点并用最佳近似解填充
        （失败点在 SQP 中通过 λ_slack 软约束逐步收敛）
      - 若失败率 > 50%，该 b₀ 候选放弃

Stage 2: Jacobian SQP 迭代（核心）
  │
  │  for each initial (b₀, {qᵢ}):
  │    # 初始化: bᵢ = b₀ ∀i, s_prev = +∞, lambda_slack = lambda_init
  │    for iter = 1..max_sqp_iter:
  │      1. 计算每个 waypoint 的 EE 残差 Δxᵢ (归一化空间下 ‖·‖_Wx)
  │      2. 计算 J_arm_i 和 J_base_i（归一化空间）
  │      3. 构建 QP (含松弛变量 sᵢ):
  │         - 目标: Huber(base 移动) + arm 平滑 + margin + λ_slack·‖s‖²
  │         - 约束: EE 软等式 (J·δ+s=Δx), 关节限位,
  │                 arm 轨迹连续性 (作用在 qᵢ+δqᵢ 上),
  │                 base 速度/加速度, 信赖域
  │      4. 求解 QP → {δbᵢ}, {δqᵢ}, {sᵢ}
  │      5. Line search（可选, merit function 含 λ_slack·‖s‖）
  │      6. 更新 bᵢ ← bᵢ · exp(α · δbᵢ), qᵢ ← qᵢ + α · δqᵢ
  │      7. 更新 λ_slack（基于 max ‖sᵢ‖ 的自适应规则, 见 2.3）
  │      8. 收敛检查:
  │         converged = (max ‖Δxᵢ‖ < ε_pos_rot) AND (max ‖sᵢ‖ < ε_slack)
  │         if converged: break
  │
  └── 收集所有收敛的候选

Stage 3: 精确验证
  ├── 对 SQP 收敛结果做精确连续 IK 验证（1mm / 0.01rad）
  ├── 检查 base 轨迹的实际速度/加速度是否达标
  ├── 碰撞检测（如启用）
  └── 计算质量指标

Stage 4: 排序与输出
  ├── 排序指标:
  │   - min_joint_margin（木桶效应）
  │   - mean_manipulability
  │   - base 总位移（越小越好）
  │   - base 移动事件数（Huber 非零项数，越少越好）
  │   - 零空间余量（鲁棒性度量）
  │
  └── 输出:
      status: "feasible" | "infeasible"
      strict_top_k: 完全通过 Stage 3 验证的候选
      best_effort_top_k: 覆盖率最高的候选
      每个候选包含:
        - arm 轨迹 {qᵢ}
        - base 轨迹 {bᵢ}
        - 质量报告 (margin, manipulability, base 位移统计)

回退路径:
  如果所有 SQP 失败 → 回退到 v3 的 IRM + coarse-to-fine 全局搜索
  如果仍失败 → status = infeasible, 返回 best_effort
```

### 3.1 base 轨迹的预期行为

典型结果是**分段常数**结构：

```
base 位移
  ^
  |          ┌──┐
  |          │  │
  |──────────┘  └──────────────────────
  +──────────────────────────────────→ waypoint
  1    20   30  35                  100

waypoint 1-20:  base 不动
waypoint 20-35: 一次有限移动（arm 要到达新区域）
waypoint 35-100: base 又不动
```

base 移动是"事件"而非"持续过程"。

---

## 4. J_base 的具体形式（body-body 约定）

**职责**：把 "base 在 base body frame 下的 twist δb" 映射到 "EE 在 EE body frame
下产生的 twist"，以便组装 `Δxᵢ = J_arm·δq + J_base·δb`。

**通用公式**：

```
J_base = Ad(T_eb)          # base body twist  →  EE body twist
T_eb   = T_be⁻¹            # 即 "base 相对 EE" 的位姿
       = (arm FK 给出的 T_be 求逆)

Ad(T) 对 T = (R, p) 的展开:
  Ad(T) = [ R     [p]× · R ]
          [ 0        R     ]   ∈ ℝ⁶ˣ⁶
```

**注意方向**：原 v4 文档写的 `Ad(T^base_ee)` 符号含糊，容易误解为 `Ad(T_be)`
（即 `T_be` 的 R, p），那是 "EE body → base body" 的映射，方向反了。
v5 明确使用 **`Ad(T_eb)`**，即 `T_be⁻¹` 的 R, p：

```
若 T_be = (R_be, p_be), 则:
  T_eb = (R_be^T, -R_be^T · p_be)
       = (R_eb,    p_eb)

J_base = Ad(T_eb) = [ R_eb        [p_eb]× · R_eb ]
                    [  0              R_eb        ]
```

每次 SQP 迭代 T_be 变化（因 q 和 b 都在变），J_base 需按当前 (q, b) 重新计算
（6×6 矩阵运算，成本可忽略）。

### SE(3) 情形: δb = (δv_x, δv_y, δv_z, δω_x, δω_y, δω_z) ∈ 𝔰𝔢(3)

直接用上式 `J_base = Ad(T_eb)`，6×6 矩阵。

### SE(2) 情形: δb = (δv_x, δv_y, δω_z) ∈ 𝔰𝔢(2)

SE(2) 底盘在 3D 空间中相当于 "只有 x/y 平移 + z 旋转" 的子流形。把 δb 嵌入到
SE(3) body twist 再套用 `Ad(T_eb)`：

```
δb_SE2 = (δv_x, δv_y, δω_z)
         ↓ 嵌入 SE(3) body twist
δb_SE3 = (δv_x, δv_y, 0, 0, 0, δω_z)

J_base^{SE(2)} = Ad(T_eb) · [e₁  e₂  e₆]   # 选出 1,2,6 列
              = Ad(T_eb) 的第 1, 2, 6 列     ∈ ℝ⁶ˣ³

显式形式（记 T_eb = (R_eb, p_eb)，R_eb 的列为 r1, r2, r3, p_eb = (px, py, pz)）:

J_base^{SE(2)} = [  r1_x     r2_x     py·r1_z - pz·r1_y + (base z 轴在 EE body
                                        下的伴随相关项)     ]
                ...   （完整展开从 Ad(T_eb) 取列）
```

（实现时直接调 `pin.SE3(T_eb).toActionMatrix()` 取列，无需手推）

**归一化**：按 2.5 节，δb 各分量已归一到无量纲 `(v/L, ω/φ)`，J_base 的列按对应
因子缩放（或等价地在 `(δv, δω)` 输入端做缩放）。

### 退化检查

- `T_be = I` 时（EE 恰在 base 原点且朝向对齐），`T_eb = I`, `Ad(T_eb) = I`，
  δb 和 Δx 是同一个 6 维向量，符合直觉
- base 距 EE 远（‖p_eb‖ 大）时，`[p_eb]×·R_eb` 分量大，表示 base 旋转通过杠杆
  放大为 EE 的平移，与物理一致（甩动效应）

### Spatial frame 下的对比（仅作参考，v5 不采用）

若改走 spatial 约定（方案 B），`J_base^s = I₆`，结构最简但 δb 是 world twist，
与 body-frame retraction 冲突且非完整底盘约束变 bilinear。见 2.1.1 节选型说明。

---

## 5. SQP 的数学理论极限（已知问题）

| 问题 | 严重程度 | 说明 | 缓解 |
|------|---------|------|------|
| 奇异位形处 J_arm 病态 | 中 | 伪逆发散，QP 解不稳定 | 阻尼项 / 信赖域约束 |
| 大残差时一阶近似失效 | **高** | FK 非线性在 δq > 0.5rad 时显著 | **λ_slack 松弛 + line search + 信赖域** |
| 初始 QP 不可行（空可行域） | **高** | Stage 1 seed 残差大 + 硬等式约束导致不可行 | **λ_slack 松弛 EE 约束**（2.2.2 节） |
| 局部最优 / 不跨 IK 分支 | **高** | 梯度方法无法跨越不连通可行域 | 多初始点 + IRM 兜底 |
| 连续性退化 | 低 | QP 中显式连续性约束 (作用在最终轨迹上) | 修正约束形式 (2.2.3 节) |
| 关节限位边界行为 | 低 | QP 的有效集方法自然处理 | QP 求解器处理 |

**新增关注**：
- Huber 的 δ 过小 → 退化为 L1，数值差；δ 过大 → 退化为 L2，失去稀疏性
- Huber 的 δ 应 ≥ 信赖域 step_limit，防止单步内跨越线性区间
- base 速度/加速度约束过紧 → QP 不可行；过松 → 抖动
- λ_slack 调度过激进 → 可能陷入"假收敛"（需依赖收敛判定中的 ‖s‖ 检查）
- λ_slack 上下界极端值可能引发数值病态（lambda_max > 1e8 有风险）

**李群相关的数值问题**：
- log_SE3 在 ξ≈0 附近数值敏感（0/0 形式），需 Taylor 展开处理（pinocchio 已内建）
- log_SE3 在旋转 π 附近轴向二义，v5 场景中相邻 b 差不到 π，实际不遇到
- 线性化 ξᵢ ≈ ξᵢ^(k) + (δbᵢ - δbᵢ₋₁) 在大步长下精度下降，信赖域 step_limit
  需足够小（< 0.1 rad 量级）以保证近似有效
- δb 的参数化约定（body-frame twist, 右乘 retraction）必须全程一致，
  与 J_base 的 Adjoint 形式对应

**参考系错配类 bug（2.1.1 约定锁定后的常见陷阱）**：
- J_arm 用 WORLD / Δx 用 body：数值上残差方向错但优化仍会"收敛"到错解
- J_base 写成 Ad(T_be) 而非 Ad(T_eb)：base 扰动方向反，优化逼近速度异常或发散
- retraction 方向与 twist frame 不匹配：单次迭代有效，多次迭代累积漂移
- 以上错误在单点测试（N=1, δb=0）时都难以暴露，**需构造 base 大距离/大旋转的
  回归测试**（test_lie_group.py + test_reference_frame_consistency.py）

---

## 6. 与 v4 的关系

v5 是 v4 的**严格推广**：
- λ_base → ∞ 时，v5 退化为 v4（base 固定）
- v4 的 J_base 推导、SQP 框架、零空间分析全部复用
- 仅 QP 公式有增强（逐点 base + Huber + base 动力学约束）

```
v4 QP 变量: δb (1×n_base)  + {δq_i} (N×n_joints)
v5 QP 变量: {δb_i} (N×n_base) + {δq_i} (N×n_joints)
```

| 方面 | v4 | v5 |
|------|----|----|
| base 假设 | 段内固定 | 段内可动，尽量不动 |
| 优化变量 | δb (全局) | {δb_i} (逐点) |
| base 移动代价 | 无 | Huber/L1 稀疏惩罚 |
| 场景覆盖 | 仅固定 base | 固定/可动/在线 统一 |
| 量纲处理 | 隐式，手工权重 | 归一化，自动 |
| base 动力学约束 | 无 | 速度/加速度上限 |
| QP 规模 | 703 变量 | ~1300 变量 |
| 实现复杂度 | 中 | 中高（多 Huber 辅助变量） |

---

## 7. 开放讨论点 / 待确定细节

继承自 v4：
1. **QP 求解器选择**：OSQP（开源、Python 原生）vs qpOASES（代码库已有、C++）
2. **SQP 信赖域策略**：固定步长 vs 自适应
3. **初始点策略的具体组合**：启发式权重、是否默认启用 IRM
4. **line search 是否必要**：或者信赖域约束足够
5. **margin_cost 的具体形式**：log-barrier vs 二次罚函数
6. **回退触发条件**：SQP 不收敛的判定标准

v5 新增：
7. ~~**Huber 的 δ 参数**~~：**已锁定为分量级 δⱼ**（见 2.4 / 2.5.3），
    归一化空间默认 δ_v=0.04, δ_ω=0.02（对应 2cm / 1.15°）。仅"是否按机器人
    微调 δ_v/δ_ω 比例"作为可选项
8. **λ_base 的默认值和自动调节**：标定方法
9. **base 速度/加速度上限**：从机器人参数读取还是手动配置
10. ~~**Wb 权重矩阵**~~：**分量级 Huber 下不再需要整体 Wb 矩阵**，
    速度/加速度约束的 Wb 对角参数 γ 保留，默认 1.0
11. **是否在归一化空间中做 SQP**：是，建议归一化（条件数 + 权重合理性）
12. **是否需要二阶精修**：SQP 收敛后是否再跑一次全变量优化

v5 QP 可行性相关（新增）：
13. **λ_slack 初始值**：默认 100，是否需要按问题规模自适应
14. **λ_slack 上下界**：[10, 1e7] 默认，极端值可能引发数值病态
15. **λ_slack 调度参数**：β_up=3, β_down=2, threshold=0.8 的稳健性
16. **ε_slack 阈值**：收敛判定中对松弛变量的容差（建议与 ε_pos_rot 同量级）
17. **Stage 1 IK 容差**：从 5cm/10° 收紧到 1cm/1°，失败率阈值如何设定
18. **Wx 中位置/姿态权重**：是否支持用户按任务自定义（某些任务姿态不重要）

v5 李群相关（新增）：
19. **ξᵢ 线性化：简化 vs 严格**：简化 `ξᵢ ≈ ξᵢ^(k) + (δbᵢ - δbᵢ₋₁)` 足够，
    还是需要 `Jr⁻¹ · δbᵢ - Jr⁻¹ · Ad · δbᵢ₋₁` 严格形式
20. ~~**δb 参数化约定**~~：**已锁定 body-frame + right-retraction（方案 A, 见 2.1.1）**
21. **Huber/速度约束作用对象**：是否考虑 "body-frame ξᵢ 的范数" 与
    "twist 的 spatial norm" 对机器人响应的差异（通常前者够）
22. **SE(2) 的 V⁻¹ 雅可比**：小角度直接用单位阵近似，还是全程用精确 V⁻¹
23. **底盘实际执行**：QP 输出的是 bᵢ 轨迹（李群序列），转为底盘控制指令
    (v, ω) 时的积分方式与 ZOH/FOH 选择（body-frame 约定下 ξᵢ/Δtᵢ 直接就是 cmd_vel，
    无需坐标变换，方案 A 的主要工程优势）

v5 Huber QP 展开相关（新增）：
24. ~~**Huber 的 QP 展开形式**~~：**已锁定重构 B / 分量级 Huber（见 2.4.1）**。
    整体 L2 Huber（需 SOCP）作为未来扩展
25. ~~**λ_base 谱系覆盖范围**~~：**v5 仅覆盖稀疏档和固定档**（见 1.2），
    平滑档/自由档交给 v6-online
26. **分量 δⱼ 的各向异性**：是否按机器人平移/旋转能力比例单独调（默认同值）

---

## 8. 文件结构

目录布局遵循 14.0.2 的三层架构（底层 / 中层 / 上层），方便未来 v6-online
复用而不污染 v5。

```
robotics/manipulation/base_placement/
├── core/                              # 底层 + 中层
│   │
│   │  —— 底层（frozen spec，与 horizon 无关）——
│   ├── robot_model.py                 # pinocchio 包装, FK, Jacobian (LOCAL)
│   ├── lie_group_utils.py             # SE(2)/SE(3) 李群操作 (log/Exp/Ad/插值)
│   ├── normalization.py               # 归一化工具 (Wx, Wb, characteristic length)
│   ├── se3_discretizer.py             # SE(2)/SE(3) 离散化（兜底路径用）
│   ├── trajectory_utils.py            # 轨迹重采样（测地线插值）, 关键点提取
│   │
│   │  —— 中层（参数化模块，v5/v6 共享，配置差异）——
│   ├── qp_formulator.py               # LocalQPBuilder: 接 waypoint_indices
│   │                                   #   + frozen_indices + cost 列表
│   ├── costs/                         # 可插拔 cost 模块
│   │   ├── __init__.py
│   │   ├── ee_slack.py                # EESlackCost (所有场景)
│   │   ├── arm_smoothness.py          # ArmSmoothnessCost (所有场景)
│   │   ├── joint_margin.py            # JointMarginCost (所有场景)
│   │   ├── base_sparsity.py           # BaseSparsityCost: Huber (v5 专用)
│   │   └── base_step.py               # BaseStepCost: 单步惩罚 (v6-online 预留)
│   ├── constraints/                   # 约束模块（全部共用）
│   │   ├── trust_region.py
│   │   ├── joint_limits.py
│   │   └── base_dynamics.py           # v_max, a_max
│   ├── continuous_ik.py               # 连续 IK 验证（Stage 3）
│   ├── irm.py                         # IRM 生成/查询（Stage 1 + 兜底）
│   ├── initial_guess.py               # 初始点生成（启发式 + IRM + 并行 IK）
│   ├── metrics.py                     # 质量指标计算
│   │
│   │  —— 上层（Solver 外循环，v5/v6 各自实现）——
│   └── solvers/
│       ├── __init__.py
│       ├── global_sqp.py              # GlobalSQPSolver (v5, 当前实现目标)
│       ├── rolling_mpc.py             # RollingMPCSolver (v6-online, 未来, 留空)
│       └── single_point_ik.py         # SinglePointIKSolver (调试/退化对照)
│
├── config/
│   └── robot_configs/
│       ├── franka_se2.yaml
│       ├── franka_se3.yaml
│       └── ...
├── api/
│   └── bpo_solver.py                  # 对外 API (v5 默认走 global_sqp)
└── tests/
    ├── test_lie_group.py              # 李群操作单元测试（wrap_to_pi, log_SE3 边界）
    ├── test_reference_frame_consistency.py # 参考系约定回归测试（Ad(T_eb), LOCAL jac
    │                                       #   有限差分对比，retraction 方向）
    ├── test_qp_formulator.py          # 中层测试（不同 waypoint_indices 组合）
    ├── test_global_sqp.py             # v5 上层测试
    └── ...
```

**分层边界的强制约束**：
- 底层模块 **不得 import 中层或上层**
- 中层模块 **不得 import 上层**，不得假设"所有 waypoint 都是决策变量"
- 上层模块可组合中层，但不应绕过中层直接操作底层
- cost 模块之间互不依赖

---

## 9. YAML 配置示例

```yaml
robot:
  urdf_path: "path/to/robot.urdf"
  base_link: "base_link"
  ee_link: "ee_link"

base_dof:
  type: "SE2"                          # SE2 | SE3
  limits:                              # 归一化前的原始限位
    x: [-2.0, 2.0]                     # m
    y: [-2.0, 2.0]                     # m
    yaw: [-3.14, 3.14]                 # rad

normalization:
  characteristic_length: 0.5           # arm 有效工作半径 (m)

sqp:
  max_iterations: 50
  convergence_eps_pos: 0.001           # m (归一化前的位置容差)
  convergence_eps_rot: 0.01            # rad
  convergence_eps_slack: 0.001         # 松弛变量容差 (归一化空间)
  trust_region_step: 0.1               # 归一化空间（李代数向量的范数上限）
  qp_solver: "osqp"                    # osqp | qpoases
  enable_line_search: true
  initial_ik_tolerance_pos: 0.01       # Stage 1 精确 IK 容差 (m)
  initial_ik_tolerance_rot: 0.017      # Stage 1 精确 IK 容差 (rad, 约 1°)
  initial_ik_max_failures: 0.5         # Stage 1 失败率上限

lie_group:                             # 李群操作配置
  twist_frame: "body"                  # 锁定为 body (方案 A, 见 2.1.1), 不提供切换
  xi_linearization: "simplified"       # simplified | strict (用 Jr⁻¹ 严格形式)
  se2_small_angle_approx: true         # SE(2) |Δθ|<0.1 时直接用单位 V⁻¹

slack:                                 # λ_slack 自适应调度
  lambda_init: 100.0
  lambda_min: 10.0
  lambda_max: 1.0e7
  beta_up: 3.0                         # 残差下降时的增长系数
  beta_down: 2.0                       # 残差上升时的衰减系数
  threshold: 0.8                       # 相对下降阈值
  epsilon_tight: 1.0e-3                # 切换到 lambda_max 的残差阈值

costs:
  lambda_base: 100.0                   # base 稀疏惩罚（大 λ → 稀疏档, v5 默认）
  lambda_smooth_q: 1.0                 # arm 平滑
  lambda_margin: 10.0                  # 关节余量
  margin_cost_type: "log_barrier"      # log_barrier | quadratic

base_sparsity:                         # 分量级 Huber (重构 B, 见 2.4.1)
  target: "xi"                         # "xi" | "delta_xi" (v5 固定 xi; delta_xi 预留 v6-online)
  norm_type: "component"               # "component" (v5) | "l2_socp" (未来, 需 SOCP 求解器)
  delta_v_normalized: 0.04             # 平移分量 δ (归一化空间, 对应原始 ~2cm)
  delta_omega_normalized: 0.02         # 旋转分量 δ (归一化空间, 对应原始 ~1.15°)
  # 约束: 所有 δⱼ ≥ trust_region_step（防止单步跨越 Huber 线性区）

weights:                               # 归一化权重矩阵（2.5 节）
  Wx_position: 1.0                     # EE 位置误差权重
  Wx_orientation: 1.0                  # EE 姿态误差权重
  Wb_rotation_ratio: 1.0               # base 速度/加速度约束中旋转 vs 平移代价比 γ
                                       # (分量级 Huber 下不再需要整体 Wb 矩阵)

base_dynamics:
  v_max: 0.5                           # m/s (or rad/s)
  a_max: 1.0                           # m/s² (or rad/s²)

initial_guess:
  use_heuristic: true
  use_irm: false
  use_parallel_ik: false
  num_candidates: 10

output:
  top_k: 5
  enable_best_effort: true
  enable_collision_check: false
```

---

## 10. 参考论文

| 论文 | 与方案的关系 |
|------|------------|
| Vahrenkamp 2013 (IRM 经典) | Stage 1 IRM + 兜底路径 |
| RM4D (ICRA 2025) | IRM 后端的高效实现选项 |
| CuRobo (NVIDIA 2023) | 球包络碰撞检测 + 并行 IK 筛选参考 |
| AutoMoMa (CVPR 2026) | 统一链建模思路（作为 Stage 1 并行 IK 选项） |
| Wachter IFAC 2024 | 嵌入轨迹规划器做 base 优化的验证 |
| Pink (INRIA) | 任务约束 IK，零空间利用参考 |

---

## 11. 关键代码参考

| 组件 | 文件路径 |
|------|----------|
| 现有 workspace 分析 | `drivers/manipulators/tools/workspace_visualization/workspace_analyzer.{h,cc}` |
| FK/IK 基础接口 | `manipulation/kinematics_dynamics/kinematics_base.h` |
| Pinocchio 模型 | `manipulation/kinematics_dynamics/pinocchio_model.h` |
| 移动底盘 IK | `manipulation/kinematics_dynamics/kinematics_mobile_base.h` |
| 轨迹数据结构 | `manipulation/actions/move_action_config.h` |
| QP 求解器 (qpOASES) | `third_party/qpOASES/` |
| QP 求解器 (OSQP) | `third_party/osqp/` |
| 离线 IK 验证参考 | `manipulation/kinematics_dynamics/check_ik_offline.cc` |

---

## 12. Python 依赖

```
pinocchio           # 运动学 + 李群操作（log6/exp6/SE3 工具）
numpy scipy pyyaml matplotlib tqdm
osqp                # QP 求解器（或 qpOASES Python binding）
cvxpy               # 可选，用于原型验证 QP 公式
```

**李群操作实现注意**：
- 优先使用 pinocchio 的 `SE3` / `log6` / `exp6` / `Jlog6` 接口
- SE(2) 可自行实现或复用 manif/sophus（注意 manif 有 Python binding）
- 避免自己从零实现 log_SE3（边界情况多，数值易错）

**参考系一致性（方案 A 对应接口）**：

```python
# EE 位姿与 body-frame 残差
T_we_cur = data.oMf[ee_frame_id]                    # SE3, world→ee
dx_body  = pin.log6(T_we_cur.inverse() * T_we_tgt).vector   # 6D EE body twist

# arm body jacobian
pin.computeJointJacobians(model, data, q)
J_arm = pin.getFrameJacobian(model, data, ee_frame_id,
                             pin.ReferenceFrame.LOCAL)      # 6×n_arm

# J_base (body→body): Ad(T_eb)
T_be = T_wb.inverse() * T_we_cur                    # ee 相对 base
T_eb = T_be.inverse()
J_base_full = T_eb.toActionMatrix()                 # 6×6 = Ad(T_eb)
# SE(2) 情形: 取第 [0,1,5] 列得到 6×3
J_base_se2 = J_base_full[:, [0, 1, 5]]
```

**常见错误自查**：
- ❌ `J_arm` 用 `ReferenceFrame.WORLD` 但 `Δx` 用 body twist → 数值错但不报错
- ❌ `J_base = T_be.toActionMatrix()` （应为 `T_eb`）→ 优化方向相反
- ❌ retraction 写成 `T_wb ← Exp(δb) · T_wb` （左乘）→ 与 body δb 不自洽

---

## 13. 实现阶段（未来开始时参考）

按 14.0.2 的分层自底向上推进（底层先稳定，再做中层，最后上层）：

1. **Phase 1（底层）**: RobotModel (pinocchio 包装, LOCAL jacobian) +
                       lie_group_utils + normalization + trajectory_utils
                       (测地线插值) + YAML 加载
                       + test_lie_group / test_reference_frame_consistency
2. **Phase 2（中层 - 约束/cost 框架）**: qp_formulator (接受 waypoint_indices
                       + frozen_indices) + 基础 cost 模块 (EESlack, ArmSmoothness,
                       JointMargin) + 约束模块 (信赖域, 限位, base 动力学)
                       + test_qp_formulator
3. **Phase 3（中层 - Huber 与辅助）**: BaseSparsityCost (Huber → 纯 QP 展开,
                       见 2.4.1 待定) + ξ 线性化 + 松弛变量完整集成
                       + λ_slack 自适应调度
4. **Phase 4（上层 - v5 Solver）**: GlobalSQPSolver (SQP 外循环, 多初始点管理,
                       line search) + SinglePointIKSolver (退化对照, 调试用)
                       + 初始点生成（启发式）+ 精确连续 IK
5. **Phase 5（质量评估）**: 精确 IK 验证 + 质量指标 + 排序输出 +
                       strict/best_effort 语义
6. **Phase 6（可选增强）**: IRM 兜底路径 + 并行 IK 初始点生成 +
                       碰撞检测集成（球包络 + ESDF）
7. **Phase 7（迁移）**: C++ 迁移（如需要）
8. **Phase 8（v6-online, 远期）**: RollingMPCSolver 独立分支，
                       复用 Phase 1-3 的底层和中层

**当前状态**：纯讨论阶段，未开始实现。

---

## 14. 未来目标（记录，不在 v5 范围内）

### 14.0 共同基石：Jacobian-based 迭代框架

v5（离线全局）和未来 v6-online（流式 MPC）**共享同一套 Jacobian 一阶近似核心**，
差别只在"局部问题的规模"和"外层循环的组织方式"。这是设计上有意保留的分层。

#### 14.0.1 Jacobian 家族树

按"局部问题的规模和耦合结构"，常见运动学求解方法构成一棵家族树：

```
                    Jacobian 一阶近似
                           │
      ┌────────┬───────────┼───────────┬──────────────┐
      ↓        ↓           ↓           ↓              ↓
   单点 IK   IK+null    horizon=1   horizon=H     全局 SQP
   (最简)   space      MPC         MPC          (v5 所在)
            (冗余利用)  (v6-online  (v6-online
                        退化形式)    主形式)
      │        │           │           │              │
      └────────┴───────────┴───────────┴──────────────┘
               QP 变量数: O(1) → O(n) → O(H) → O(N)
               耦合深度: 无  → 单点 → 窗口 → 全局
               稀疏性:   无  → 无   → 弱   → 强
```

所有方法可以统一写成：

```
min   f(δq, δb ; history, preview)
s.t.  J_arm · δq + J_base · δb + s = Δx          (共同, 规模不同)
      信赖域 / 关节限位 / base 动力学             (共同)
      跨 waypoint 平滑 / 稀疏惩罚                  (耦合范围不同)
```

v5 选择 horizon=N 的全局 SQP，是这棵树上"视野最宽、稀疏性最强、离线友好"的那个分支。

#### 14.0.2 设计分层

```
┌─────────────────────────────────────────────────────────┐
│  上层: Solver（外循环 / horizon 策略）                   │
│   ├── GlobalSQPSolver       (v5, 当前实现目标)            │
│   ├── RollingMPCSolver      (v6-online, 未来)             │
│   └── SinglePointIKSolver   (调试/退化对照)               │
├─────────────────────────────────────────────────────────┤
│  中层: LocalQPBuilder（局部 QP 构建）                     │
│   ├── 参与集合 (waypoint_indices)                        │
│   ├── 冻结集合 (frozen_indices, 历史状态)                │
│   ├── Cost 模块（可插拔）                                │
│   │    ├── EESlackCost                 (所有场景)         │
│   │    ├── ArmSmoothnessCost           (所有场景)         │
│   │    ├── JointMarginCost             (所有场景)         │
│   │    ├── BaseSparsityCost (Huber)    (v5 专用)          │
│   │    └── BaseStepCost                (v6-online 专用)   │
│   └── 约束模块（全部共用）                               │
│        ├── EE 软等式 (J·δ+s=Δx)                          │
│        ├── 信赖域 / 关节限位                             │
│        └── base 动力学 (v_max, a_max)                    │
├─────────────────────────────────────────────────────────┤
│  底层: 核心基础设施（完全共用，与 horizon 无关）          │
│   ├── robot_model: FK / Jacobian (pinocchio 包装)       │
│   ├── lie_group_utils: Exp/log/Ad (SE(2)/SE(3))         │
│   ├── 参考系约定（2.1.1, body frame + retraction）       │
│   └── 归一化 (Wx, Wb, characteristic length)             │
└─────────────────────────────────────────────────────────┘
```

**分层原则**：
- **底层是 frozen spec**：一旦 v5 锁定，后续分支不得变更（否则数值不一致）
- **中层是参数化模块**：以"参与 waypoint 集合"和"cost 列表"作为配置，v5 传
  `[0..N-1]` 和全局 Huber；v6-online 传 `[k..k+H-1]` 和单步 cost
- **上层是具体策略**：每个 Solver 独立实现外循环，互不污染

#### 14.0.3 v5 实现对未来的约束

为避免未来 v6-online 重写一半代码，v5 实现时须注意：

1. **`qp_formulator` 接口应接受 waypoint 索引集合**，不要假设全部 N 个参与
   ```python
   # 期望接口
   qp = qp_formulator.build(
       waypoint_indices=list(range(N)),    # v5 传全部
       frozen_indices=[],                  # v5 传空
       preview_indices=[],                 # v5 不用
       ...
   )
   ```
2. **SQP 外层循环与 QP 构建分离**，便于替换外壳
   ```python
   for iter in range(max_sqp_iter):
       state_vecs = compute_all(state)    # Jacobian 层 (底层, 完全共用)
       qp         = qp_formulator.build(...) # QP 构建层 (中层, 参数化共用)
       δ          = qp_solve(qp)             # 求解层 (共用)
       state      = retraction(state, δ)     # 更新层 (底层, 完全共用)
   ```
3. **稀疏惩罚独立为 cost 插件**，不要硬编码进 `qp_formulator`
4. **不在底层引入"全局 Huber"的隐式假设**——比如不要假设"所有 waypoint 都是
   决策变量"，否则 frozen_indices 概念引入时要大改

### 14.1 在线流式 BPO（MPC 范式）

**场景**：EE 目标逐个到来（遥操作、视觉伺服、上游流式任务规划），
需要每收到新目标立即产出底盘+关节指令，无法等完整轨迹。

**与 v5 的根本差异**：
- 视角从"全局 N 个 waypoint 联合"变为"当前目标 + 有限历史/预览"
- 每次 QP 变量数从 ~1900 降到 ~10~H×10
- Huber 稀疏性会退化或消失（无全局视野）
- Stage 0~4 的四阶段流程压缩为"首点初始化 + 每点单次 SQP"

**拟采用思路（待设计）**：
- **Horizon=H 的滑动窗口 MPC**：上游维护未来 H 个目标的预览，每次 QP 解 H-
  waypoint 联合问题，只固化首点
- H 的权衡：H=1 完全无稀疏；H=5~10 有限稀疏；H→N 等价于全局 QP
- Warm-start：用上次解的尾部作为本次初始点
- 稀疏性补救方案（若需保留）：
  - 方案 α：horizon>1 的预览窗口
  - 方案 β：离线粗略模板 + 在线 QP 只在模板允许点激活 Huber
  - 方案 γ：滞回 + 事件触发式移动（b_k 默认锁死，残差超阈值才解锁）

**与 v5 的复用点**（得益于 14.0 的分层）：
- 底层全部复用（robot_model, lie_group_utils, 参考系, 归一化）
- 中层 LocalQPBuilder 复用，只改参与集合和 cost 列表
- 上层 Solver 需新写 RollingMPCSolver

**不复用点**：
- 全局 Huber 惩罚结构（稀疏性机制需重新设计，方案 α/β/γ）
- Stage 0/1 的完整轨迹预处理和多初始点策略
- 排序输出与 top-k 语义

**里程碑位置**：v5 稳定落地后另起分支（tentatively "v6-online" 或
"bpo_mpc"），与 v5 并列维护而非替换。

### 14.2 其他已记录但未纳入 v5 的目标

（待补充：AutoMoMa 风格的并行化数据生成、零空间鲁棒性度量、碰撞感知的
动态障碍场景等）

---

## 附录：版本与分支关系

```
v1-v3  →  v4 (固定 base)  →  v5 (离线全局 QP, 段内可动)   ← 当前主线
                                │
                                │  共享底层（robot_model / lie_group /
                                │            参考系 / 归一化）
                                │  共享中层（LocalQPBuilder, 参数化）
                                │
                                └→ v6-online (MPC/流式, 未来目标)
                                     替换上层 Solver + 重新设计稀疏机制
```

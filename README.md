# ProductIntegral Direct

## k/gamma/delta Study

The zero-training finite-slab DtN operator now accepts runtime `k_cat`,
`gamma`, and `delta`, includes higher-accuracy Gauss-Legendre TraceGreen
quadrature, and has a differentiable delta-inversion prototype. A fast
near-exact/SOE Abel-history backend accelerates repeated forward and inverse
solves without changing the nonlinear ProductIntegral closure.

See [KGDELTA_FORWARD_INVERSE_REPORT.md](KGDELTA_FORWARD_INVERSE_REPORT.md) for
the equations, FDM-separated validation, error decomposition, inversion
results, and identifiability limits.

The new multi-scan inverse study simultaneously estimates `k_cat`, `gamma`,
and `delta` in sensitivity-qualified regimes, and automatically supports
two-parameter and one-parameter fallbacks. It uses no FDM data and no neural
network. See
[MULTISCAN_PARAMETER_INVERSE_REPORT.md](MULTISCAN_PARAMETER_INVERSE_REPORT.md)
for the equations, scan-rate design, blind multi-start results, noise
ablation, and the low-reaction/high-Damkoehler failure boundaries.

Run scan design and multi-scan inversion together with:

```bash
OUTPUT_DIR="/content/gdrive/MyDrive/pinn_v96_multiscan_inverse" \
MODE=multiscan \
TRUE_K=1 TRUE_GAMMA=10 TRUE_DELTA=0.035 \
SCAN_RATES="5,40,320" \
INVERSE_MODES="all,fix-gamma" \
bash ./run_colab_delta_forward_inverse.sh
```

Colab can run the FDM-free study with:

```bash
OUTPUT_DIR="/content/gdrive/MyDrive/pinn_v96_delta_research" \
MODE=all \
bash ./run_colab_delta_forward_inverse.sh
```

The runner uses `HISTORY_BACKEND=soe` for stress, inversion, and
identifiability. Set `HISTORY_BACKEND=direct` to reproduce the exact
quadratic-history reference. Run only the timing and accuracy comparison with:

```bash
OUTPUT_DIR="/content/gdrive/MyDrive/pinn_v96_delta_research" \
MODE=benchmark \
bash ./run_colab_delta_forward_inverse.sh
```

本分支是当前 PINN v9.6 的精简主线，只保留两条可复现实验流程：

```text
固定参数基模：
Dynamic Stage 1
-> ProductIntegral 300 epochs
-> zero-training inventory lift
-> posterior FDM comparison

联合参数泛化：
固定 ProductIntegral checkpoint
-> runtime (k_cat, gamma) physics conditioning
-> zero-training inventory lift
-> 17-case posterior FDM comparison
```

最终模型不是一个直接从 `(t,x,k,gamma)` 回归全部浓度的黑箱网络。它由以下部分组成：

1. 薄层反应扩散 PDE 与 Nernst 电极边界；
2. 薄膜准稳态反应闭合；
3. ProductIntegral 求解界面 Abel 历史；
4. TraceGreen 将界面历史解析传播到外域；
5. Hermite 基函数重建薄层浓度；
6. posterior inventory lift 强制 CV 电流与薄层库存守恒一致；
7. 小型神经网络只学习未被上述结构固定的薄层剩余自由度。

FDM 只用于冻结 checkpoint 后的 posterior comparison，不进入训练 loss、
physics early stopping 或 checkpoint selection。

## 1. 无量纲物理问题

代码中的时间、空间、浓度和扩散系数均已无量纲化。默认扫描参数为

$$
\sigma=40,\qquad
\theta_i=10,\qquad
\theta_v=-10,
$$

因此一次完整三角波扫描时间为

$$
T_{\mathrm{sim}}
=\frac{2|\theta_i-\theta_v|}{\sigma}
=1.
$$

薄层厚度和外域长度为

$$
\delta=\lambda\sqrt{T_{\mathrm{sim}}}=0.035,
\qquad
L_{\mathrm{ext}}=6\sqrt{T_{\mathrm{sim}}}=6.
$$

计算域分为

$$
0\le x\le\delta
$$

的薄层区域，以及

$$
\delta\le x\le \delta+L_{\mathrm{ext}}
$$

的外部扩散区域。代码默认

$$
D_A=D_B=D_C=D_D=1.
$$

在 $0\le t\le T_{\mathrm{sim}}$ 上，完整三角波电位可写成一个统一公式：

$$
\theta(t)
=\theta_v
+2(\theta_i-\theta_v)
\left|
\frac{t}{T_{\mathrm{sim}}}-\frac12
\right|.
$$

它在 $t=0$ 时从 $\theta_i$ 出发，在 $t=T_{\mathrm{sim}}/2$ 时到达
$\theta_v$，随后在 $t=T_{\mathrm{sim}}$ 时回到 $\theta_i$。前半段斜率为
$-\sigma$，后半段斜率为 $+\sigma$。

实现位于 `potential_theta()`。网络特征可使用平滑的扫描方向
`potential_theta_dot_smooth()`，但真实物理电位仍采用上面的分段线性函数。

## 2. 反应扩散方程

薄层中存在电活性物种 $A,B$：

$$
\frac{\partial C_A}{\partial t}
=D_A\frac{\partial^2 C_A}{\partial x^2},
\qquad
\frac{\partial C_B}{\partial t}
=D_B\frac{\partial^2 C_B}{\partial x^2},
\qquad 0<x<\delta.
$$

外域中存在催化物种 $C,D$：

$$
\frac{\partial C_C}{\partial t}
=D_C\frac{\partial^2 C_C}{\partial x^2},
\qquad
\frac{\partial C_D}{\partial t}
=D_D\frac{\partial^2 C_D}{\partial x^2},
\qquad x>\delta.
$$

系统满足逐点守恒

$$
C_A+C_B=1,
\qquad
C_C+C_D=\gamma.
$$

因此主线网络只需要独立构造 $C_B$ 和 $C_D$，再使用

$$
C_A=1-C_B,\qquad C_C=\gamma-C_D
$$

精确恢复其余浓度。薄层 PDE loss 仍同时检查 $A,B$；TraceGreen 外域只需检查
一个独立扩散残差。

初始条件为

$$
C_A(x,0)=1,\quad C_B(x,0)=0,
$$

$$
C_C(x,0)=\gamma,\quad C_D(x,0)=0.
$$

外域远场条件为

$$
C_C(\delta+L_{\mathrm{ext}},t)=\gamma,
\qquad
C_D(\delta+L_{\mathrm{ext}},t)=0.
$$

## 3. 电极 Nernst 边界和 CV 电流

在电极 $x=0$ 处，Nernst 平衡写成

$$
C_A(0,t)=C_B(0,t)e^{\theta(t)}.
$$

结合 $C_A+C_B=1$，得到解析表面状态

$$
C_A(0,t)=\frac{e^{\theta}}{1+e^{\theta}}
=\frac{1}{1+e^{-\theta}}
=\mathrm{sigmoid}(\theta),
$$

$$
C_B(0,t)=\frac{1}{1+e^{\theta}}
=\mathrm{sigmoid}(-\theta).
$$

代码中的 `_surface_state()` 直接使用该解析结果，不让界面网络学习 Nernst
映射。

CV 电流按代码的符号约定定义为

$$
J_{\mathrm{surf}}(t)
=-D_A\left.\frac{\partial C_A}{\partial x}\right|_{x=0}
=D_A\left.\frac{\partial C_B}{\partial x}\right|_{x=0}.
$$

`ThinLayerNet_v9_6_MultiscaleHermite.surface_current_state()` 从 Hermite 左端斜率
直接读取该电流。

## 4. 界面反应和薄膜闭合

在 $x=\delta$ 处，反应速率为

$$
J(t)=k_{\mathrm{cat}}C_{B,i}(t)C_{C,i}(t).
$$

代码的界面通量约定为

$$
-D_A C_{A,x}+J=0,
\qquad
-D_B C_{B,x}-J=0,
$$

$$
-D_C C_{C,x}+J=0,
\qquad
-D_D C_{D,x}-J=0.
$$

这里下标 $i$ 表示薄层/外域界面。

薄膜准稳态近似假设 $B$ 在薄层中的反应传质满足

$$
C_{B,i}
=\frac{C_{B,s}}
{1+k_{\mathrm{cat}}\delta C_{C,i}/D_B},
$$

其中 $C_{B,s}=C_B(0,t)$。定义局部 Damköhler 数

$$
Da_i
=\frac{k_{\mathrm{cat}}\delta C_{C,i}}{D_B},
$$

则

$$
C_{B,i}=\frac{C_{B,s}}{1+Da_i},
$$

$$
J
=\frac{D_B}{\delta}C_{B,s}
\frac{Da_i}{1+Da_i}.
$$

`_film_reaction()` 实现上述关系。参数化版本使用

$$
\frac{Da_i}{1+Da_i}
=\frac{1}{1+\exp(-\log Da_i)}
$$

的 log-domain 形式，避免极小或极大 $k_{\mathrm{cat}}C_{C,i}$ 下的数值问题。

用于跨参数缩放的特征通量为

$$
J_{\mathrm{ref}}(k,\gamma)
=\frac{k\gamma}
{1+k\gamma\delta/D_B}.
$$

它对应 $C_{B,s}\approx1,C_{C,i}\approx\gamma$ 时的薄膜限制通量。
`characteristic_reaction_flux()`、`flux_residual_scale()` 使用该尺度，使
不同 $k,\gamma$ 下的通量残差具有可比数量级。

## 5. Abel 历史和 ProductIntegral

外域 $D$ 由界面反应通量产生。半无限扩散问题的 Neumann-to-trace Abel
关系为

$$
C_{D,i}(t)
=\frac{1}{\sqrt{\pi D_D}}
\int_0^t
\frac{J(\tau)}{\sqrt{t-\tau}}\mathrm{d}\tau.
$$

同时

$$
C_{C,i}(t)=\gamma-C_{D,i}(t).
$$

由于 $J$ 又依赖 $C_{C,i}$，这是一个非线性 Volterra 闭合，而不是一个
可以先算 $J$ 再独立积分的显式公式。

### 5.1 已完成时间单元

时间网格为 $t_n=n\Delta t$。在每个已完成单元
$[t_m,t_{m+1}]$ 上，将 $J(\tau)$ 线性插值。代码对

$$
\int_{t_m}^{t_{m+1}}
\frac{J(\tau)}{\sqrt{t_n-\tau}}\mathrm{d}\tau
$$

进行解析积分，而不是用普通 midpoint rule。对应实现是
`_completed_product_integral()`。

若

$$
\ell_h=t_n-t_m,\qquad
\ell_l=t_n-t_{m+1},
$$

并令

$$
\Delta J_m=J_{m+1}-J_m,
\qquad
b_m=J_m+\Delta J_m\ell_h/\Delta t,
$$

则该单元贡献为

$$
2b_m(\sqrt{\ell_h}-\sqrt{\ell_l})
-\frac{2}{3}\frac{\Delta J_m}{\Delta t}
\left(\ell_h^{3/2}-\ell_l^{3/2}\right),
$$

最后整体除以 $\sqrt{\pi D_D}$。

### 5.2 含未知 $J_n$ 的末单元

奇异末单元 $[t_{n-1},t_n]$ 可精确写成

$$
C_{D,i,n}
=I_n^{\mathrm{completed}}
+2\sqrt{\frac{\Delta t}{\pi D_D}}
\left(\frac13J_{n-1}+\frac23J_n\right).
$$

记

$$
a_n=2\sqrt{\frac{\Delta t}{\pi D_D}},
$$

则需要求解标量方程

$$
F(C_{D,i,n})
=C_{D,i,n}-I_n^{\mathrm{completed}}
-a_n\left(
\frac13J_{n-1}
+\frac23J(C_{D,i,n})
\right)=0.
$$

代码先做两次 fixed-point update，再做解析 Newton projection：

$$
C_D^{(q+1)}
=C_D^{(q)}
-\frac{F(C_D^{(q)})}{F'(C_D^{(q)})},
$$

$$
F'
=1-\frac23a_n\frac{dJ}{dC_D}.
$$

固定参数模型默认做一次 Newton projection；联合 KG 版本做四次，以增强高
Damköhler 数下的闭合稳定性。整个过程没有引入可学习 ProductIntegral
系数。

### 5.3 gamma-normalized 状态

联合参数模型直接求解

$$
d_i(t)=\frac{C_{D,i}(t)}{\gamma},
\qquad
c_i(t)=\frac{C_{C,i}(t)}{\gamma}=1-d_i(t).
$$

归一化 Abel 关系为

$$
d_i(t)
=\frac{1}{\gamma\sqrt{\pi D_D}}
\int_0^t\frac{J(\tau)}{\sqrt{t-\tau}}\mathrm{d}\tau.
$$

末单元闭合变成

$$
d_{i,n}
=\frac{I_n^{\mathrm{completed}}}{\gamma}
+\frac{a_n}{\gamma}
\left(\frac13J_{n-1}+\frac23J_n\right).
$$

最后恢复

$$
C_{D,i}=\gamma d_i,
\qquad
C_{C,i}=\gamma(1-d_i).
$$

这样避免大 $\gamma$ 时直接计算 $\gamma-C_D$ 的尺度损失，也让 Newton
状态始终处于理论区间 $0\le d_i\le1$。

`InterfaceStateNet_v9_6_FilmTraceProductIntegral` 实现固定参数闭合；
`InterfaceStateNet_v9_6_FilmTraceKGParam` 实现联合 $k,\gamma$ 闭合。

## 6. TraceGreen 外域传播

令

$$
y=x-\delta\ge0.
$$

已知 Dirichlet 边界历史 $C_{D,i}(t)$ 后，半无限热方程解可以写成

$$
C_D^{G}(y,t)
=\int_0^t
\frac{y}
{2\sqrt{\pi D_D(t-\tau)^3}}
\exp\left[
-\frac{y^2}{4D_D(t-\tau)}
\right]
C_{D,i}(\tau)\mathrm{d}\tau.
$$

普通均匀时间积分在 $y\to0^+$ 时容易漏掉集中在 $\tau\to t$ 的核质量，
从而造成界面浓度跳变。代码使用变量

$$
u=\mathrm{erfc}
\left(
\frac{y}{2\sqrt{D_D(t-\tau)}}
\right)
$$

重参数化积分。此时

$$
\tau
=t-\frac{y^2}
{4D_D[\mathrm{erfc}^{-1}(u)]^2},
$$

并在 $u$ 上积分，使

$$
\lim_{y\to0^+}C_D^{G}(y,t)=C_{D,i}(t)
$$

在离散计算中也能保持。

有限外域的远场误差通过端点 Hermite 修正消除。令

$$
r=\frac{y}{L_{\mathrm{ext}}},
\qquad
z=\frac{1-e^{-\beta r}}{1-e^{-\beta}},
$$

$$
h_{01}(z)=-2z^3+3z^2,
$$

则基础外域场为

$$
C_D^{\mathrm{base}}(y,t)
=C_D^G(y,t)-h_{01}(z)C_D^G(L_{\mathrm{ext}},t).
$$

因此界面值不变，远场值严格回到零。

一般 TraceGreen 类允许一个端点消失的平滑余项

$$
C_D=C_D^{\mathrm{base}}+R_{\mathrm{smooth}}.
$$

但当前 direct ProductIntegral runner 设置

```text
clean_residual_initial_scale = 0
```

所以最终主线从训练开始就使用

$$
R_{\mathrm{smooth}}\equiv0.
$$

也就是说，最终外域浓度完全由 ProductIntegral 界面历史和 TraceGreen
解析传播产生，外域神经网络不再“画”浓度场。

对应代码为
`ExternalNet_v9_6_FilmTraceGreenClean.trace_boundary_convolution_fused()`。

## 7. 薄层 Hermite 重建和神经网络作用

令

$$
s=x/\delta\in[0,1].
$$

三次 Hermite 基函数为

$$
h_{00}=2s^3-3s^2+1,
\qquad
h_{10}=s^3-2s^2+s,
$$

$$
h_{01}=-2s^3+3s^2,
\qquad
h_{11}=s^3-s^2.
$$

薄层 $B$ 的物理基础场为

$$
C_B^{H}(x,t)
=h_{00}C_{B,s}
+\delta h_{10}C_{B,x}(0,t)
+h_{01}C_{B,i}
+\delta h_{11}C_{B,x}(\delta,t).
$$

右端斜率由反应通量固定：

$$
C_{B,x}(\delta,t)=-J/D_B.
$$

网络只添加保持端点值不变的 bubble correction：

$$
C_B
=C_B^H
+g_t(t)\left[
\alpha s(1-s)^2\tanh r_1(t,x)
+\beta s^2(1-s)^2\tanh r_0(t,x)
\right],
$$

其中

$$
g_t(t)=1-\exp[-t/(0.05T_{\mathrm{sim}})].
$$

两个 bubble 在 $s=0,1$ 都为零，所以不会破坏解析端点浓度。
$s^2(1-s)^2$ 也不改变端点斜率；
$s(1-s)^2=h_{10}$ 只修正电极端斜率，因此可直接解释为 CV 电流修正。

网络 `MultiscaleResidualHead` 使用 Fourier features、残差 MLP 和小幅初始化。
它学习的是薄层内部扩散形状与电极斜率剩余量，而不是学习：

- Nernst 表面浓度；
- 界面薄膜反应；
- Abel/ProductIntegral 历史；
- TraceGreen 外域传播；
- 质量守恒恒等式。

对应代码为 `ThinLayerNet_v9_6_MultiscaleHermite`。

## 8. 薄层库存守恒和 posterior inventory lift

定义薄层 $B$ 库存

$$
M_B(t)=\int_0^\delta C_B(x,t)\mathrm{d}x.
$$

由薄层扩散方程和界面反应可得代码使用的电流恒等式

$$
J_{\mathrm{surf}}
=-J-\frac{dM_B}{dt}.
$$

`thin_current_components()` 分别计算

$$
J_{\mathrm{reaction}}=-J,
\qquad
J_{\mathrm{inventory}}=-\frac{dM_B}{dt},
$$

$$
J_{\mathrm{conservative}}
=J_{\mathrm{reaction}}+J_{\mathrm{inventory}}.
$$

posterior lift 在冻结网络后加入

$$
\Delta C_B(x,t)=\delta a(t)h_{10}(x/\delta),
\qquad
\Delta C_A=-\Delta C_B.
$$

该修正满足：

- 两端浓度不变；
- 右端界面斜率不变；
- 左端电极斜率增加 $a(t)$；
- $C_A+C_B=1$ 精确保持。

由于

$$
\int_0^1 h_{10}(s)\mathrm{d}s=\frac1{12},
$$

库存修正为

$$
\Delta M_B=\frac{\delta^2}{12}a(t).
$$

令基础场的电流差为

$$
S(t)=J_{\mathrm{conservative}}^0-J_{\mathrm{surface}}^0,
$$

则幅值满足一阶因果方程

$$
\frac{\delta^2}{12}\frac{da}{dt}
+D_Aa=S(t),
\qquad a(0)=0.
$$

代码使用分段线性源项的 exponential time differencing 求解该方程，而不是
重新训练网络。对应类为 `ThinLayerNet_v9_6_InventoryHermiteLift`。

inventory lift 是 zero-training posterior operator。训练 checkpoint 本身不因
lift 或 FDM 数据发生改变。

## 9. 联合 k/gamma 条件泛化

接口接受任意有限正参数

$$
k_{\mathrm{cat}}>0,\qquad \gamma>0.
$$

正式 posterior 网格覆盖

$$
k\in[0.01,100],
\qquad
\gamma\in[0.1,100].
$$

参数条件特征为

$$
\eta_k
=\tanh\left[\log_{10}(k/k_{\mathrm{ref}})\right],
\qquad k_{\mathrm{ref}}=1,
$$

$$
\eta_\gamma
=\tanh\left[\log_{10}(\gamma/\gamma_{\mathrm{ref}})\right],
\qquad \gamma_{\mathrm{ref}}=10.
$$

联合薄层 adapter 的形式为

$$
r(t,x;k,\gamma)
=r_{\mathrm{base}}(t,x)
+\eta_k\eta_\gamma
\Delta r(t,x,\eta_k,\eta_\gamma).
$$

因此当 $k=1$ 或 $\gamma=10$ 时，联合交互修正严格消失。adapter 最后一层
零初始化；固定 ProductIntegral checkpoint warm-start 到 KG 架构后，
zero-shot 模式没有 optimizer step，所以 adapter 在整个参数域仍输出零。

当前已验证的 zero-training 泛化主要来自：

1. 运行时重新计算 $Da_i$ 和薄膜反应；
2. 对每个 $(k,\gamma)$ 重新求解归一化 ProductIntegral 历史；
3. 对每个参数对重新执行 TraceGreen 外域传播；
4. 对每个参数对重新计算 $J_{\mathrm{ref}}$ 和 inventory lift；
5. 固定网络只提供参考薄层剩余场。

参数不是新的物理坐标，因此 PDE loss 不对下面两个参数方向求导：

$$
\frac{\partial C}{\partial k},
\qquad
\frac{\partial C}{\partial\gamma}.
$$

每个 ProductIntegral batch 必须共享同一个 $(k,\gamma)$，因为同一条因果
历史不能混合不同物理参数。切换参数时，ProductIntegral、TraceGreen 和
inventory-lift cache 会全部清除；cache key 也包含当前参数对。

对应代码为：

- `conditioned_model_inputs()`
- `activate_kg_from_input()`
- `InterfaceStateNet_v9_6_FilmTraceKGParam`
- `ThinLayerNet_v9_6_MultiscaleHermiteKGParam`
- `ExternalNet_v9_6_FilmTraceGreenKGParam`
- `ThinLayerNet_v9_6_InventoryHermiteLiftKGParam`

## 10. 参数一致的残差缩放

外域浓度残差使用

$$
s_C(\gamma)=\frac{\gamma_{\mathrm{ref}}}{\gamma},
\qquad \gamma_{\mathrm{ref}}=10.
$$

通量残差使用

$$
s_J(k,\gamma)
=\frac{J_{\mathrm{ref}}(1,10)}
{J_{\mathrm{ref}}(k,\gamma)}.
$$

这样参考点 $(k,\gamma)=(1,10)$ 的 loss 权重严格保持原值，同时避免低
$\gamma$ 或低 $k$ 案例在绝对量级上被训练/验证指标忽略。

posterior 指标同样报告

$$
C_C/\gamma,\quad C_D/\gamma,\quad J/J_{\mathrm{ref}},
$$

以便跨参数比较。

## 11. Physics-only loss 和 early stopping

训练总损失由以下物理项组成：

$$
\mathcal L
=w_{\mathrm{thin}}\mathcal L_{\mathrm{PDE,thin}}
+w_{\mathrm{ext}}\mathcal L_{\mathrm{PDE,ext}}
+w_s\mathcal L_{\mathrm{Nernst}}
+w_f\mathcal L_{\mathrm{far}}
+w_0\mathcal L_{\mathrm{IC}}
+w_b\mathcal L_{\mathrm{bounds}}
+w_i\mathcal L_{\mathrm{interface}}.
$$

薄层 PDE 项为

$$
\mathcal L_{\mathrm{PDE,thin}}
=\lVert C_{A,t}-D_AC_{A,xx}\rVert_2^2
+\lVert C_{B,t}-D_BC_{B,xx}\rVert_2^2.
$$

外域一般形式为

$$
\mathcal L_{\mathrm{PDE,ext}}
=\left\lVert
s_C(\gamma)
(C_{C,t}-D_CC_{C,xx})
\right\rVert_2^2.
$$

对于当前纯 TraceGreen 主线，解析 Green lift 被视为已满足热方程的结构项，
autograd PDE residual 只作用于 $R_{\mathrm{smooth}}$。direct runner 中
$R_{\mathrm{smooth}}=0$，因此不会对奇异 Green 核错误地计算普通点值 PDE
残差。

TraceGreen 的外域界面通量由解析边界势承担。因为普通 autograd 在 $y=0$
无法捕捉 $\tau\to t$ 的奇异核质量，主线不再用点值导数强迫外域界面通量，
而是检查

$$
C_{C,i}+C_{D,i}=\gamma.
$$

checkpoint selection 和 early stopping 使用固定、确定性的 physics validation
点集。`fixed_physics_validation_score()` 和
`parameterized_physics_validation_score()` 不读取 FDM。

## 12. 训练流程中各阶段的作用

### Dynamic Stage 1

`multiscale_green_grid_dynamic_stage1` 使用同一组 PDE、Nernst、初值、远场和
界面物理 loss，训练一个容易优化的多尺度初始化模型。它的作用是提供薄层
网络权重和基础时空特征，不是最终论文模型。

### ProductIntegral 300 epochs

Stage 1 checkpoint warm-start 到
`multiscale_film_tracegreen_productintegral`：

- 界面历史替换为固定 ProductIntegral 算子；
- 外域替换为纯 TraceGreen 解析传播；
- ProductIntegral 界面参数全部冻结；
- 只微调仍有自由度的薄层 Hermite 网络；
- optimizer、scheduler 和 best score 重新开始。

### Inventory lift

训练结束后使用
`multiscale_film_tracegreen_productintegral_lift`。该阶段 optimizer steps 为零，
只求解库存守恒 ODE 并进行 posterior comparison。

## 13. 公式与代码模块对应表

| 数学/物理部分 | 核心公式 | 代码模块 |
|---|---|---|
| 参数与无量纲尺度 | $T_{\mathrm{sim}},\delta,L_{\mathrm{ext}},J_{\mathrm{ref}}$ | `configure_physical_parameters()`, `characteristic_reaction_flux()` |
| 三角波电位 | $\theta(t)$ | `potential_theta()`, `potential_theta_dot_smooth()` |
| 坐标归一化 | $t,x\mapsto[-1,1]$ | `normalize_time()`, `normalize_thin_x()`, `normalize_ext_x()` |
| Nernst 表面状态 | $C_A=\mathrm{sigmoid}(\theta),\quad C_B=\mathrm{sigmoid}(-\theta)$ | `InterfaceStateNet_v9_6_FilmAbel._surface_state()` |
| 薄膜闭合 | $C_{B,i}=C_{B,s}/(1+Da_i)$ | `_film_reaction()` |
| Damköhler/log-domain transfer | $Da/(1+Da)=1/[1+\exp(-\log Da)]$ | `InterfaceStateNet_v9_6_FilmTraceKParam._film_transfer_fraction()` |
| Abel 历史 | $C_{D,i}=(\pi D_D)^{-1/2}\int J/\sqrt{t-\tau}$ | `InterfaceStateNet_v9_6_FilmTraceProductIntegral` |
| 已完成 PI 单元 | 分段线性通量的解析奇异积分 | `_completed_product_integral()` |
| 末单元闭合 | $I_n+a_n(J_{n-1}/3+2J_n/3)$ | `_history_grid()` |
| Newton projection | $s\leftarrow s-F/F'$ | `_product_integral_d_j_d_c_d()`, `_history_grid()` |
| gamma-normalized PI | $d=C_D/\gamma$ | `InterfaceStateNet_v9_6_FilmTraceGammaParam`, `InterfaceStateNet_v9_6_FilmTraceKGParam` |
| 外域 Dirichlet Green 势 | 热方程 Poisson kernel | `trace_boundary_convolution_fused()` |
| erfc trace preservation | $u=\mathrm{erfc}[y/(2\sqrt{D\Delta t})]$ | `ExternalNet_v9_6_MultiscaleGreenGridFilmTrace` |
| 远场端点修正 | $C_D^G-h_{01}C_D^G(L)$ | `tracegreen_lift_and_residual()` |
| 最终纯 TraceGreen | $R_{\mathrm{smooth}}=0$ | `ExternalNet_v9_6_FilmTraceGreenClean` |
| 薄层 Hermite 场 | $h_{00},h_{10},h_{01},h_{11}$ | `hermite_cubic_basis()`, `ThinLayerNet_v9_6_MultiscaleHermite` |
| 薄层神经剩余 | endpoint-preserving bubble correction | `MultiscaleResidualHead`, `_raw_field()` |
| CV 电流 | $J_{\mathrm{surf}}=-D_AC_{A,x}(0,t)$ | `surface_current_state()`, `thin_current_components()` |
| 库存恒等式 | $J_{\mathrm{surf}}=-J-dM_B/dt$ | `thin_current_components()` |
| posterior inventory lift | $(\delta^2/12)\dot a+D_Aa=S$ | `ThinLayerNet_v9_6_InventoryHermiteLift` |
| 联合条件输入 | `[t,x,k,gamma]`，每 batch 一个参数对 | `conditioned_model_inputs()`, `activate_kg_from_input()` |
| 联合零门控 adapter | $r=r_0+\eta_k\eta_\gamma\Delta r$ | `ThinLayerNet_v9_6_MultiscaleHermiteKGParam` |
| 参数缓存隔离 | cache key 包含 `(k,gamma)` | `set_conditions()`, `condition_cache_key()` |
| 浓度/通量缩放 | $s_C=10/\gamma,\quad s_J=J_{\mathrm{ref}}(1,10)/J_{\mathrm{ref}}$ | `external_residual_scale()`, `flux_residual_scale()` |
| physics-only validation | 固定 PDE/BC/守恒网格 | `fixed_physics_validation_score()`, `parameterized_physics_validation_score()` |
| 模型装配 | ProductIntegral/TraceGreen/Hermite/lift 组合 | `create_models_v96()` |
| posterior FDM comparison | 冻结预测后计算 RMSE/CV | `compare_concentration_fields.py`, `compare_kg_parameter_cases.py` |

## 14. 文件说明

```text
pinn_thin_layer_v9_6.py
    物理算子、网络、训练、physics validation、checkpoint。

compare_concentration_fields.py
    单个固定或参数化 case 的 posterior FDM comparison。

compare_kg_parameter_cases.py
    17 个联合 k/gamma case 的批量 posterior 汇总。

run_colab_productintegral_direct.sh
    Dynamic Stage 1 -> ProductIntegral 300 -> inventory lift。

run_colab_kg_posterior.sh
    固定 checkpoint -> joint KG zero-shot -> mandatory lift -> 17-case FDM。

test_parameter_scale_consistency.py
    k、J_ref、缓存和高-k ProductIntegral 稳定性。

test_gamma_parameterization.py
    gamma-normalized ProductIntegral 与 gamma cache。

test_kg_parameterization.py
    联合参考点一致性、参数切换、physics-only metadata 和 FDM 泄漏保护。
```

内部仍保留少量带 `FilmTraceClean` 名称的父类，以维持历史 checkpoint 的参数
名称和张量形状兼容。它们不是当前公开训练架构。

## 15. Colab: 固定参数基模

```python
from google.colab import drive
drive.mount("/content/gdrive")
```

```bash
!git clone -b codex/productintegral-direct-clean --single-branch \
  https://github.com/FanoShannon/pinn_test.git \
  /content/pinn_productintegral_direct
%cd /content/pinn_productintegral_direct
```

填写物理参数和匹配的 posterior FDM：

```python
K_CAT = 1.0
GAMMA = 10.0
FDM_PKL = (
    "/content/gdrive/MyDrive/FDM_kg_v42/"
    "kg_k1_g10_v42_thin_layer_catalytic_v42.pkl"
)
```

完整训练：

```bash
!K_CAT={K_CAT} GAMMA={GAMMA} FDM_PKL={FDM_PKL} \
  bash ./run_colab_productintegral_direct.sh
```

复用已有 Dynamic Stage 1：

```bash
!K_CAT={K_CAT} GAMMA={GAMMA} FDM_PKL={FDM_PKL} \
STAGE1_BEST="/content/gdrive/MyDrive/pinn_v96_reference/reference_k1.0_gamma10.0/clean_two_stage/stage1_dynamic_fixed/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_stage1_best.pth" \
bash ./run_colab_productintegral_direct.sh
```

默认输出：

```text
/content/gdrive/MyDrive/pinn_v96_productintegral_direct/
  direct_k1.0_gamma10.0/
    stage1_dynamic/
    productintegral_300/
    final_inventory_lift/
    logs/
```

## 16. Colab: 联合 KG zero-shot + lift

17 个 posterior FDM 文件由
[FanoShannon/FDM 的 `codex/kg-grid-v42` 分支](https://github.com/FanoShannon/FDM/tree/codex/kg-grid-v42)
生成和验证。

固定参考 checkpoint：

```python
CHECKPOINT = (
    "/content/gdrive/MyDrive/pinn_v96_reference_direct/"
    "reference_k1.0_gamma10.0/productintegral_300_from_stage1/checkpoints/"
    "pinn_thin_layer_catalytic_v9_6_"
    "multiscale_film_tracegreen_productintegral.pth"
)
```

上面的非 `_best.pth` 文件是复现流程的 epoch 2900 checkpoint；`_best.pth`
对应 physics validation 选出的 epoch 2700。

运行 17-case posterior：

```bash
!CHECKPOINT="{CHECKPOINT}" \
FDM_DIR="/content/gdrive/MyDrive/FDM_kg_v42" \
OUTPUT_DIR="/content/gdrive/MyDrive/pinn_v96_reference_direct/kg_epoch2900_zero_shot_lift" \
bash ./run_colab_kg_posterior.sh
```

主要输出：

```text
kg_parameter_summary.json
kg_vs_historical.json
k_*_gamma_*_metrics.json
```

`run_colab_kg_posterior.sh` 要求 17 个 FDM case 全部存在，避免缺少案例时生成
不可比较的 mean/worst 指标。

## 17. 当前复现基准

epoch 2900 direct ProductIntegral checkpoint 的联合 zero-shot + lift 结果为

```text
overall_dimensionless_rmse:
    mean  = 7.043918724e-4
    worst = 1.105647550e-3

CV_J_over_J_ref_rmse:
    mean  = 1.200610114e-2
    worst = 8.460394533e-2
```

相对历史最优 17-case 结果：

```text
overall mean: +0.113%
overall worst: +0.067%
CV/Jref mean: +0.111%
CV/Jref worst: +0.120%
```

因此固定参数精度和联合 KG zero-training 泛化均视为成功复现。

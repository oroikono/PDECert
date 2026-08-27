# Research landscape and positioning

This note records the public work closest to PDECert and narrows the claims the
project can defend. It is a scoped landscape review, not a systematic review.
The search was last updated on 2026-08-26 and covered SciML benchmarks, PINN
libraries, rigorous neural-PDE verification, formal differential-equation
checking, PDE specification languages, agentic PDE systems, and equation
discovery.

The central conclusion is deliberately conservative: rigorous neural-PDE
certification and symbolic differential-equation checking already exist as
separate research threads. The opportunity for PDECert is not to claim either
one as new. The opportunity is to test whether a small independent harness can
apply explicit evidence semantics to several kinds of candidate solution and
benchmark their distinct failure modes under one case specification.

## 1. Topic reframing

Three progressively stronger formulations are useful:

1. **Engineering formulation:** build an independent verifier that accepts a
   PDE problem and a candidate artifact, audits operator and condition
   obligations, and returns a reproducible report with the strength of the
   supporting evidence.
2. **Benchmark formulation:** measure which failures are detected, missed, or
   left inconclusive when the same mathematical case is represented as a
   symbolic expression, differentiable field, or gridded numerical field.
3. **Research formulation:** study how exact identities, rigorous bounds, and
   empirical diagnostics can coexist without silently upgrading sampled
   evidence into a proof.

The phrase "unified PDE certification" should be treated as a research goal,
not a current capability. A residual bound is also not automatically a bound on
solution error; the latter needs stability, well-posedness, and problem-specific
analysis.

## 2. Search coverage and terminology

The search used combinations of the following terms:

- **Core:** PDE solution verification, candidate solution checking, PDE
  residual certification, boundary-condition verification, PINN certification,
  a posteriori PDE error bounds, validated numerics.
- **Adjacent:** neural operator benchmark, physics-informed validation,
  manufactured solutions, weak residual, variational residual, interval
  arithmetic, ball arithmetic, SMT verification, proof-carrying scientific
  computing.
- **Competing terminology:** code verification, solution verification,
  numerical certification, verified computing, rigorous computation,
  continuous-domain neural-network verification.
- **Specifications:** PDE schema, symbolic PDE system, variational form DSL,
  UFL, ModelingToolkit `PDESystem`, OpenMath, Content MathML.
- **AI systems:** text-to-PDE, agentic PDE solver, symbolic math verifier,
  formal theorem proving, CAS checking, symbolic PDE discovery.
- **Benchmarks:** PDEBench, PDEArena, PINNacle, The Well, miniF2F, LeanDojo,
  PDE-FIND, PySINDy.

Searches included project documentation, repositories, conference proceedings,
journal pages, and recent preprints. A search that does not find a system is not
proof that it does not exist. Consequently this document uses "we did not
identify" rather than "no project exists."

## 3. Closest existing threads

| Thread | Representative work and status | What is already established | Remaining difference from the proposed PDECert scope |
| --- | --- | --- | --- |
| SciML surrogate benchmarks | [PDEBench](https://papers.nips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html), NeurIPS 2022 Datasets and Benchmarks; [PDEArena](https://pdearena.github.io/pdearena/), open framework; [PINNacle](https://papers.nips.cc/paper_files/paper/2024/hash/8c63299fb2820ef41cb05e2ff11836f5-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2024 Datasets and Benchmarks | Diverse nonlinear PDE data, training baselines, standardized tensor evaluation, challenging geometries, and PINN method comparisons. PDEBench already reports boundary, conservation, frequency, and maximum-error metrics in addition to RMSE. | These systems primarily compare learned or numerical predictions with reference fields. They do not provide one evidence-typed decision protocol shared with exact symbolic candidates. |
| PINN and neural-PDE libraries | [DeepXDE](https://doi.org/10.1137/19M1274067), SIAM Review 2021; [TorchPhysics](https://github.com/Qewton-Labs/torchphysics), associated review in *Inverse Problems* 2023; [PhysicsNeMo Sym](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/), maintained framework | Expressive domains, differential operators, boundary conditions, automatic differentiation, variational or integral constraints, training, validation, and inference. | Residual and boundary terms are generally training or sampled-validation objectives. The frameworks are solvers first, rather than independent artifact-neutral certification oracles. |
| Rigorous PINN residual certification | [Efficient Error Certification for Physics-Informed Neural Networks](https://proceedings.mlr.press/v235/eiras24a.html), ICML 2024, with [`partial_crown`](https://github.com/fgirbal/partial_crown) | `partial-CROWN` computes continuous-domain bounds for PINN residual and condition errors on Burgers, Schrodinger, Allen-Cahn, and diffusion-sorption examples. This directly invalidates any claim that continuous-domain PINN residual certification is unclaimed. | It is specialized to neural candidates and supported network/operator classes; it is not a cross-artifact benchmark for symbolic expressions, callables, and numerical fields. |
| A posteriori PINN error bounds | [Hillebrecht and Unger](https://doi.org/10.1109/TNNLS.2023.3335837), IEEE TNNLS 2023; [Berrone, Canuto, and Pintore](https://arxiv.org/abs/2205.00786), 2022 preprint on VPINNs; [Ernst, Rekatsinas, and Urban](https://arxiv.org/abs/2502.20336), 2025 preprint | Problem-dependent theory can translate residual information into rigorous solution-error bounds under explicit well-posedness and regularity assumptions. Recent work includes elliptic, parabolic, transport, Navier-Stokes, Klein-Gordon, and complex-geometry settings. | The estimators are mathematically specialized. A general harness must expose those assumptions instead of presenting a universal residual-to-solution certificate. |
| Emerging end-to-end neural certification | [Mukherjee et al.](https://arxiv.org/abs/2603.19165), 2026 preprint | Combines continuous-domain bounds on residual, initial, and boundary errors with equation-specific stability estimates to obtain neural-solution guarantees. | This is the closest conceptual pressure on PDECert's numerical lane. The remaining hypothesis is whether a neutral benchmark and plugin contract across artifact types adds value beyond a specialized certification pipeline. |
| Symbolic differential-equation checking | [SymPy `checkpdesol`](https://docs.sympy.org/latest/modules/solvers/pde.html); [Hickman, Laursen, and Foster](https://arxiv.org/abs/2102.02679), 2021 preprint | CAS substitution can check supported PDE solutions; Isabelle/HOL work shows how untrusted CAS-produced ODE solutions can be reconstructed and certified in a proof assistant. | SymPy is not a proof kernel and may abstain on hard identities. The Isabelle work is trustworthy but aimed at ODE solutions and proof-assistant integration, not a lightweight multi-artifact PDE benchmark. |
| Mathematical reasoning and proof tooling | [LeanDojo](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4441469427094f8873d0fecb0c4e1cee-Abstract-Conference.html), NeurIPS 2023; [SymCode](https://aclanthology.org/2026.findings-eacl.76/), Findings of EACL 2026 | Reproducible theorem-proving environments and sandboxed symbolic-code execution can provide deterministic feedback for model-generated mathematics. | The evaluated tasks and interfaces are not designed around PDE domains, traces, weak solutions, or numerical solution artifacts. Their execution isolation and proof reconstruction patterns remain relevant. |
| PDE specification languages | [UFL](https://doi.org/10.1145/2566630), ACM TOMS 2014; [SciMLBase `PDESystem`](https://docs.sciml.ai/SciMLBase/stable/interfaces/PDE/); [OpenMath](https://openmath.org/standard/) | UFL has a mature AST for variational forms and tensor calculus. `PDESystem` separates high-level symbolic equations, conditions, variables, parameters, and domains from discretization. OpenMath standardizes semantic interchange of mathematical objects. | A new schema should borrow or adapt these semantics rather than claim that structured PDE representation is novel. None alone defines PDECert's decision and evidence policy. |
| Agentic PDE automation | [PDEFlow](https://arxiv.org/abs/2607.05134), 2026 preprint; [Lang-PINN](https://openreview.net/pdf?id=2e623aa93a21fd6baca73dbcc4731a576df66abb.pdf), ICLR 2026 submission | These systems turn user descriptions into solver-backed data or executable PINN pipelines. Lang-PINN includes symbolic consistency checks for generated PDE loss code and evaluates on PINNacle. | They generate and train solutions. PDECert's proposed role is an independent evaluator of the resulting artifact and its stated mathematical obligations. |
| Symbolic PDE discovery | [PDE-FIND](https://www.science.org/doi/10.1126/sciadv.1602614), *Science Advances* 2017; [PySINDy](https://doi.org/10.21105/joss.02104), JOSS 2020 | Recovers governing equations from observed fields and provides sparse-regression tooling. | This is the inverse direction, field to equation. PDECert evaluates a proposed field against a supplied equation and conditions. The residual implementations may still be useful baselines. |
| Validated numerics | [INTLAB](https://www.tuhh.de/ti3/rump/intlab/), maintained interval toolbox; [Arb](https://flintlib.org/doc/arb.html), maintained ball-arithmetic library; [validated semilinear PDE integrator](https://github.com/MaximeBreden/validated-PDE-integrator), research code | Rigorous floating-point enclosures and computer-assisted PDE proofs predate neural certification and provide the correct numerical standard for a `RIGOROUS_BOUND` claim. | These tools are not turnkey evaluators for arbitrary generated artifacts. Integrating them soundly is a backend and semantics problem, not merely replacing floats with intervals. |

### Claims the evidence does not support

- "SciML benchmarks only report relative RMSE" is false for PDEBench.
- "No one certifies PINN residuals over a continuous domain" is false because
  of `partial-CROWN` and later work.
- "PDE specifications lack structured representations" ignores UFL and
  `PDESystem`.
- "Theorem proving is almost exclusively high-school mathematics" is too
  broad. The popular ML benchmarks emphasize Olympiad and library theorems, but
  proof assistants and validated-numerics communities cover differential
  equations and scientific computing.
- "OpenMath compliance creates certification" is false. OpenMath communicates
  semantics; it does not establish the truth of a residual or boundary claim.

## 4. Repeated limitations in the literature

### Data limitations

- Many certification studies use a small number of canonical or manufactured
  problems, which makes coverage and selection bias difficult to assess.
- Public surrogate datasets contain realistic trajectories but not always the
  continuous problem data, regularity constants, or exact boundary semantics
  needed for rigorous certification.

### Benchmark limitations

- Prediction metrics, residual diagnostics, and mathematical guarantees are
  usually evaluated in separate benchmarks.
- Benchmarks rarely compare the same solution represented as an expression, a
  neural field, and a grid while preserving one mathematical case identity.
- Failure labels often conflate a wrong candidate, an unsupported checker, and
  an inconclusive calculation.

### Theoretical limitations

- Small residual does not imply small solution error without stability and
  well-posedness assumptions.
- Exact zero equivalence is incomplete in practical CAS systems and undecidable
  in broad expression classes.
- Weak, distributional, and entropy-solution semantics are not interchangeable
  with classical pointwise satisfaction.

### Scope and generalization limitations

- Rigorous neural verifiers support restricted activations, architectures,
  operators, or rectangular domain decompositions.
- A posteriori estimators are tied to PDE classes and norms.
- Symbolic checkers struggle with branch cuts, special functions, assumptions,
  and expression growth.

### Compute and scaling limitations

- Interval and neural-bound propagation can become loose and expensive with
  dimension, network depth, domain width, and nonlinear derivatives.
- Symbolic differentiation and canonicalization can exhibit severe
  intermediate-expression growth.

### Identifiability and interpretability limitations

- A low residual may correspond to multiple solutions when the supplied
  conditions do not establish uniqueness.
- A human-readable symbolic expression is not necessarily easier to certify if
  it hides singularities or ambiguous branches.

### Evaluation and ablation weaknesses

- Few studies quantify verifier false positives, false negatives, abstention,
  witness quality, runtime, and certificate tightness together.
- Comparisons often lack systematically corrupted boundary conditions,
  parameters, phases, constants, or localized off-grid failures.

### Engineering constraints

- Problem formats, coordinate conventions, derivative APIs, and tensor layouts
  differ across ecosystems.
- Reproducible reports need versions, tolerances, assumptions, domain
  partitions, random seeds, and backend configuration.
- Executing generated candidate programs requires isolation and a clearly
  documented trust boundary.

## 5. Real gap candidates

| Candidate gap | Category | Novelty risk | Execution burden | Evidence of demand | Venue fit | Why now |
| --- | --- | --- | --- | --- | --- | --- |
| Evidence-typed reports shared by symbolic and callable candidates | Safe incremental | Low | Low | High | Workshop: medium | Existing tools use incompatible notions of pass, residual, and proof. |
| A cross-artifact failure benchmark with matched mathematical cases | Workshop or short paper | Medium | Medium | High | Workshop: high; main track: medium | Symbolic generation, PINNs, and neural operators increasingly coexist, but their evaluation remains siloed. |
| A plugin contract for exact identities, continuous-domain neural bounds, and PDE-specific a posteriori estimators | Ambitious infrastructure | Medium | High | High | TMLR/JMLR: high if mature | Strong components now exist, making careful composition more plausible than inventing every backend. |
| A calibrated abstention study across CAS, sampling, interval, and neural-bound backends | Strong empirical paper | Medium | Medium | Medium-high | Main track: medium-high | Verification systems are usually compared on success or tightness, not decision calibration and unsupported cases. |
| Weak-solution and entropy-condition certification under one explicit semantics layer | Ambitious main-track | High | High | High | Main track: high if technically sound | It addresses a genuine limitation, but a generic implementation is not realistic without restricting PDE families. |

The strongest unifying claim is therefore tentative:

> In the sources reviewed here, we did not identify a lightweight independent
> harness that evaluates symbolic expressions and differentiable neural fields
> against the same versioned PDE obligations while distinguishing exact proof,
> rigorous bounds, empirical counterexamples, and inconclusive outcomes.

This is a falsifiable positioning statement, not a priority claim.

## 6. Low-hanging-fruit opportunities

1. Add import/export adapters for one established problem representation instead
   of inventing a universal AST. UFL is the best weak-form reference;
   `PDESystem` is the best high-level symbolic comparison.
2. Add a `partial-CROWN` comparison adapter or frozen reproduction rather than
   presenting a new neural residual bounder as wholly novel.
3. Construct matched corruptions of Burgers, Allen-Cahn, Poisson, and wave cases:
   wrong phase, missing integration constant, wrong diffusivity, localized
   boundary defect, and between-collocation spike.
4. Report a decision matrix containing status, evidence level, witness,
   tolerance, runtime, and incomplete reason for every backend.
5. Keep empirical weak residuals useful but explicitly non-certifying until
   quadrature and discretization errors are bounded.

## 7. Prototype directions

### Prototype A: matched cross-artifact benchmark

- **Core message:** representation changes what a verifier can establish even
  when the mathematical case is unchanged.
- **Hypothesis:** exact symbolic checks, sampled autodiff, and reference-field
  errors have complementary and measurable failure profiles.
- **Minimal experiment:** 20 mathematical cases, each with one valid and four
  systematically corrupted candidates represented symbolically and as PyTorch
  callables.
- **Strongest baselines:** SymPy `checkpdesol`, fixed collocation, DeepXDE-style
  relative error, and PDEBench metrics where reference grids exist.
- **Likely failure mode:** callable candidates are merely translations of
  symbolic expressions and do not represent real trained-model failures.
- **Publishable result:** a preregistered corpus using independently trained
  PINNs shows reproducible disagreement and abstention patterns that are not
  explained by tolerance alone.

### Prototype B: restricted rigorous expression backend

- **Core message:** a small supported expression class can produce auditable
  domain-wide residual enclosures without pretending to solve general symbolic
  equivalence.
- **Hypothesis:** ball arithmetic plus adaptive box subdivision can certify or
  refute useful polynomial, rational, exponential, and trigonometric residuals
  on compact boxes.
- **Minimal experiment:** implement a versioned certificate payload and test it
  on manufactured Poisson, heat, Burgers traveling-wave, and Allen-Cahn front
  cases.
- **Strongest baselines:** SymPy simplification, dense random probing, Arb, and
  INTLAB-derived reference calculations.
- **Likely failure mode:** dependency inflation makes enclosures too loose,
  especially for repeated variables and high derivatives.
- **Publishable result:** zero unsound proofs on adversarial cases, useful bound
  tightness on a declared class, and a transparent abstention frontier.

### Prototype C: certified neural-backend contract

- **Core message:** PDECert can orchestrate existing neural verifiers without
  weakening their assumptions or relabeling their evidence.
- **Hypothesis:** a common certificate schema can faithfully represent
  `partial-CROWN` residual bounds and one a posteriori solution-error estimator.
- **Minimal experiment:** import pretrained Burgers and Allen-Cahn PINNs, run
  empirical autodiff and `partial-CROWN`, and serialize both results with
  distinct evidence and scope.
- **Strongest baselines:** the official `partial_crown` implementation and the
  associated ICML experiments.
- **Likely failure mode:** backend-specific assumptions cannot be represented
  without turning the common schema into an unhelpful generic envelope.
- **Publishable result:** independent reproduction plus cross-backend reports
  reveals concrete portability or calibration failures and proposes a minimal
  interoperable contract.

### Prototype D: weak-residual diagnostic with convergence evidence

- **Core message:** weak residuals extend evaluation to nonclassical candidates,
  but empirical quadrature must not be called a proof.
- **Hypothesis:** multi-resolution test-function families detect shocks and
  localized defects missed by pointwise collocation.
- **Minimal experiment:** Burgers shock, discontinuous transport, and Poisson on
  an irregular domain, with controlled defect injection and quadrature
  refinement.
- **Strongest baselines:** pointwise residual sampling, hp-VPINN/PINNacle
  implementations, and UFL/FEniCS weak-form evaluation.
- **Likely failure mode:** results depend heavily on test-function selection and
  quadrature, producing another benchmark-specific score.
- **Publishable result:** a preregistered detector study shows consistent gains
  across defect scales and clearly separates empirical convergence evidence from
  rigorous enclosure.

## 8. Core messages worth testing

These should be treated as hypotheses until experiments support them:

1. A verifier benchmark should score abstention and evidence strength, not just
   binary accuracy.
2. Boundary and initial-condition obligations deserve independent metrics and
   witnesses; a small interior residual is insufficient.
3. The same case specification can support multiple artifact types without
   pretending that their verification procedures offer equal guarantees.
4. Existing rigorous backends become more useful when their assumptions,
   obligation scope, and certificates are exposed through a reproducible report.
5. The most valuable benchmark records are realistic failures where established
   methods disagree, not additional easy exact solutions.

## 9. Risks, confounders, and why this may fail

- **Crowded certification literature:** `partial-CROWN` and recent a posteriori
  work already own much of the neural-certification claim. PDECert must compare
  with them, not omit them.
- **Unification may be superficial:** a shared JSON envelope is not a scientific
  contribution unless it enables fair experiments or sound composition.
- **Semantic mismatch:** classical, weak, entropy, and numerical solution
  notions cannot share one undifferentiated `PROVED` label.
- **Residual versus solution error:** reporting a residual enclosure as a bound
  on the unknown solution would be a serious correctness failure.
- **Manufactured benchmark bias:** exact or deliberately corrupted candidates
  can overstate performance on real model outputs.
- **Reference-solution uncertainty:** numerical ground truth has its own
  discretization and solver error.
- **Scalability:** rigorous bounds may be impractical on large neural operators,
  high-dimensional domains, or chaotic long-horizon systems.
- **Trust boundary:** SymPy, PyTorch, interval libraries, solver programs, and
  proof assistants provide different assurance levels.
- **Maintenance:** adapters to fast-moving external libraries can consume more
  effort than the core research.
- **Novelty drift:** claims that are defensible in 2026 can become false as
  adjacent projects add artifact types or certification backends.

## 10. Final recommendation

**Pursue, but reshape the public claim.** PDECert should not present itself as
the first project to certify PINNs, define PDE schemas, compute residuals, or
benchmark SciML. Those areas have strong prior art.

The best near-term research bet is the matched cross-artifact benchmark plus an
evidence-typed verifier contract. It is ambitious enough to matter, small enough
to test, and compatible with the existing code. The first technical credibility
milestone should be either an independent `partial-CROWN` reproduction exposed
through that contract or a narrowly sound interval certificate for symbolic
residuals. Weak-form diagnostics should follow as an explicitly empirical lane.

A strong paper would need three things the repository does not yet have:

1. real outputs from symbolic generators and independently trained neural
   solvers on matched cases;
2. comparisons with rigorous neural-certification and established SciML
   evaluation baselines; and
3. quantitative evidence that the unified report discovers or clarifies
   failures that each siloed evaluator misses.

Until then, the honest description is **an early cross-artifact verification and
benchmarking prototype**, not a general PDE certification standard.

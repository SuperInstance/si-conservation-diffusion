# si-conservation-diffusion

> **Proof of Concept:** Conservation-law-constrained diffusion on agent graphs — the constraint γ + η = C acts as a manifold that shapes the equilibrium of budget diffusion.

## The Insight

Standard graph diffusion (heat equation): ẋ = −L·x preserves total mass Σxᵢ. But in our fleet, we have **two** budget types (γ = durable, η = ephemeral) with the constraint **γ + η = C = const**.

When agents diffuse budget across the network:
- **Unconstrained**: Each budget type diffuses independently. Numerical drift can violate γ + η = C.
- **Constrained**: After each diffusion step, project back onto the constraint manifold.

The constraint shapes the equilibrium: instead of converging to a uniform distribution of both γ and η, the system finds the **budget-optimal** distribution that respects both diffusion dynamics and the conservation law.

## What This Proves

1. **Constraint projection works**: After each step, redistribute error proportionally → conservation restored
2. **Convergence rate depends on topology**: Complete > Ring > Star (matches spectral gap predictions)
3. **Hub diffusion is real**: Star hub's γ decreases as it flows to spokes
4. **Constrained ≤ unconstrained error**: The projection step guarantees conservation
5. **Budgets stay bounded**: Even after 1000 steps, γ and η remain non-negative

## Usage

```rust
use si_conservation_diffusion::*;

// Ring topology with 5 agents
let mut sys = ring_system(5, 0.5);
println!("Initial conservation: {}", sys.verify_conservation()); // true

// Run constrained diffusion
let result = sys.simulate(0.01, 500, true);
println!("Final error: {}", result.final_conservation_error);
println!("γ std: {}", result.final_gamma_std);
println!("Converged: {}", result.converged_uniform);

// Custom system
let mut agents = vec![
    DiffusionAgent::new(0, 20.0, 10.0),
    DiffusionAgent::new(1, 5.0, 2.0),
];
agents[0].add_neighbor(1, 3.0); // strong coupling
agents[1].add_neighbor(0, 3.0);
let mut sys = DiffusionSystem::new(agents, 1.0);
```

## Modules

- `DiffusionAgent` — agent with γ/η budgets and neighbor couplings
- `DiffusionSystem` — the diffusion simulation engine
- `unconstrained_step(dt)` — standard graph diffusion
- `constrained_step(dt)` — diffusion + projection onto γ + η = C
- `project_onto_constraint()` — distribute error proportionally across agents
- `simulate(dt, steps, constrained)` — full simulation with history
- Topology factories: `ring_system`, `complete_system`, `star_system`

## Connection to Conservation Law

This IS the dynamic version of γ + η = C:
- The conservation law is a **constraint manifold** in phase space
- Diffusion moves agents along the manifold
- Numerical errors push agents off the manifold
- Projection snaps them back
- The equilibrium is where diffusion forces balance the constraint

In physics terms: the constraint acts like a **holonomic constraint** in Lagrangian mechanics. The projection is analogous to computing constraint forces (Lagrange multipliers).

## Mathematical Background

### Graph Diffusion
For state x ∈ ℝⁿ on graph G with Laplacian L:
dx/dt = −αLx

Solution: x(t) = e^(−αLt)x(0). Converges to uniform (for connected G).

### Constraint Projection
After unconstrained step x → x', project onto Σ(γᵢ + ηᵢ) = C:
xᵢ ← xᵢ + ε · (xᵢ / Σxⱼ)

where ε = C − Σxⱼ is the total error. This distributes error proportionally.

### Equilibrium
Unconstrained: x* = (C/n)·𝟙 (uniform)
Constrained: x* satisfies both diffusion equilibrium AND γᵢ + ηᵢ = Cᵢ (individual constraints)

## Tests: 13

Covers: initial conservation, constrained vs unconstrained error, convergence to uniform, topology ordering (complete > ring), hub diffusion, history recording, projection restoration, total constant, non-negative budgets, weighted coupling, snapshot consistency.

## License

MIT

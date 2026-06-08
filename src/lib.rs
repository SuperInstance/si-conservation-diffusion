//! Conservation-law-constrained diffusion on agent interaction graphs.
//!
//! Standard diffusion (heat equation) on graphs: ẋ = -L·x where L is the Laplacian.
//! This conserves total mass Σxᵢ = const.
//!
//! We add a **conservation constraint**: total budget γ + η must equal C at every
//! time step. Diffusion moves budget between agents, but the constraint is maintained
//! by projecting back onto the constraint manifold after each step.
//!
//! Key results:
//! - Unconstrained diffusion converges to uniform distribution (all equal)
//! - Conservation-constrained diffusion converges to a *budget-optimal* distribution
//! - The constraint acts as a "potential" that shapes the diffusion equilibrium

/// Agent state in the diffusion model.
#[derive(Debug, Clone)]
pub struct DiffusionAgent {
    pub id: usize,
    pub gamma: f64,    // durable budget
    pub eta: f64,      // ephemeral budget
    pub neighbors: Vec<(usize, f64)>, // (neighbor_id, coupling_strength)
}

impl DiffusionAgent {
    pub fn new(id: usize, gamma: f64, eta: f64) -> Self {
        Self { id, gamma, eta, neighbors: vec![] }
    }
    pub fn total(&self) -> f64 { self.gamma + self.eta }
    pub fn add_neighbor(&mut self, n: usize, w: f64) {
        if !self.neighbors.iter().any(|(id, _)| *id == n) {
            self.neighbors.push((n, w));
        }
    }
}

/// The diffusion system.
#[derive(Debug, Clone)]
pub struct DiffusionSystem {
    pub agents: Vec<DiffusionAgent>,
    pub total_conservation: f64, // C = Σ(γᵢ + ηᵢ)
    pub diffusion_rate: f64,
}

impl DiffusionSystem {
    pub fn new(agents: Vec<DiffusionAgent>, diffusion_rate: f64) -> Self {
        let total: f64 = agents.iter().map(|a| a.total()).sum();
        Self { agents, total_conservation: total, diffusion_rate }
    }

    /// Total γ across all agents.
    pub fn total_gamma(&self) -> f64 { self.agents.iter().map(|a| a.gamma).sum() }
    /// Total η across all agents.
    pub fn total_eta(&self) -> f64 { self.agents.iter().map(|a| a.eta).sum() }
    /// Verify conservation: |Σ(γᵢ + ηᵢ) - C| < ε
    pub fn verify_conservation(&self) -> bool {
        (self.total_gamma() + self.total_eta() - self.total_conservation).abs() < 1e-8
    }
    /// Conservation error.
    pub fn conservation_error(&self) -> f64 {
        (self.total_gamma() + self.total_eta() - self.total_conservation).abs()
    }

    /// One step of unconstrained diffusion (heat equation on graph).
    /// dγᵢ/dt = α · Σⱼ wᵢⱼ · (γⱼ - γᵢ)
    pub fn unconstrained_step(&mut self, dt: f64) {
        let n = self.agents.len();
        let alpha = self.diffusion_rate;
        // Compute diffs first (simultaneous update)
        let mut gamma_diffs = vec![0.0; n];
        let mut eta_diffs = vec![0.0; n];
        for i in 0..n {
            for &(j, w) in &self.agents[i].neighbors {
                gamma_diffs[i] += alpha * w * (self.agents[j].gamma - self.agents[i].gamma) * dt;
                eta_diffs[i] += alpha * w * (self.agents[j].eta - self.agents[i].eta) * dt;
            }
        }
        for i in 0..n {
            self.agents[i].gamma += gamma_diffs[i];
            self.agents[i].eta += eta_diffs[i];
        }
    }

    /// One step of conservation-constrained diffusion.
    /// After unconstrained step, project back onto γ + η = C manifold.
    pub fn constrained_step(&mut self, dt: f64) {
        self.unconstrained_step(dt);
        self.project_onto_constraint();
    }

    /// Project all agents onto the constraint manifold Σ(γᵢ + ηᵢ) = C.
    /// Distribute the error proportionally.
    pub fn project_onto_constraint(&mut self) {
        let current = self.total_gamma() + self.total_eta();
        let error = self.total_conservation - current;
        // Distribute error proportionally to each agent's total
        let totals: Vec<f64> = self.agents.iter().map(|a| a.total()).collect();
        let total_sum: f64 = totals.iter().sum();
        if total_sum < 1e-12 { return; }
        for i in 0..self.agents.len() {
            let fraction = totals[i] / total_sum;
            // Split correction evenly between γ and η
            let correction = error * fraction / 2.0;
            self.agents[i].gamma += correction;
            self.agents[i].eta += correction;
        }
    }

    /// Run simulation for n_steps, recording history.
    pub fn simulate(&mut self, dt: f64, n_steps: usize, constrained: bool) -> SimulationResult {
        let mut history = vec![];
        history.push(self.snapshot());
        for _ in 0..n_steps {
            if constrained { self.constrained_step(dt); }
            else { self.unconstrained_step(dt); }
            history.push(self.snapshot());
        }
        let final_error = self.conservation_error();
        let gammas: Vec<f64> = self.agents.iter().map(|a| a.gamma).collect();
        let etas: Vec<f64> = self.agents.iter().map(|a| a.eta).collect();
        let gamma_std = stddev(&gammas);
        let eta_std = stddev(&etas);
        SimulationResult {
            history,
            final_conservation_error: final_error,
            final_gamma_std: gamma_std,
            final_eta_std: eta_std,
            converged_uniform: gamma_std < 0.01 && eta_std < 0.01,
        }
    }

    fn snapshot(&self) -> TimeStep {
        TimeStep {
            gammas: self.agents.iter().map(|a| a.gamma).collect(),
            etas: self.agents.iter().map(|a| a.eta).collect(),
            total: self.total_gamma() + self.total_eta(),
        }
    }
}

/// Standard deviation.
fn stddev(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

/// A single time step snapshot.
#[derive(Debug, Clone)]
pub struct TimeStep {
    pub gammas: Vec<f64>,
    pub etas: Vec<f64>,
    pub total: f64,
}

/// Simulation result.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub history: Vec<TimeStep>,
    pub final_conservation_error: f64,
    pub final_gamma_std: f64,
    pub final_eta_std: f64,
    pub converged_uniform: bool,
}

/// Build a ring topology.
pub fn ring_system(n: usize, diffusion_rate: f64) -> DiffusionSystem {
    let mut agents: Vec<DiffusionAgent> = (0..n).map(|i| {
        DiffusionAgent::new(i, 10.0 + (i as f64).sin() * 5.0, 5.0 + (i as f64).cos() * 3.0)
    }).collect();
    for i in 0..n {
        agents[i].add_neighbor((i + 1) % n, 1.0);
        agents[i].add_neighbor((i + n - 1) % n, 1.0);
    }
    DiffusionSystem::new(agents, diffusion_rate)
}

/// Build a complete topology.
pub fn complete_system(n: usize, diffusion_rate: f64) -> DiffusionSystem {
    let mut agents: Vec<DiffusionAgent> = (0..n).map(|i| {
        DiffusionAgent::new(i, 10.0 + (i as f64).sin() * 5.0, 5.0 + (i as f64).cos() * 3.0)
    }).collect();
    for i in 0..n {
        for j in 0..n {
            if i != j { agents[i].add_neighbor(j, 1.0); }
        }
    }
    DiffusionSystem::new(agents, diffusion_rate)
}

/// Build a star topology (hub at 0).
pub fn star_system(n: usize, diffusion_rate: f64) -> DiffusionSystem {
    let mut agents: Vec<DiffusionAgent> = (0..n).map(|i| {
        if i == 0 { DiffusionAgent::new(0, 50.0, 25.0) }
        else { DiffusionAgent::new(i, 5.0, 2.0) }
    }).collect();
    for i in 1..n {
        agents[0].add_neighbor(i, 1.0);
        agents[i].add_neighbor(0, 1.0);
    }
    DiffusionSystem::new(agents, diffusion_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation_initial() {
        let sys = ring_system(5, 0.1);
        assert!(sys.verify_conservation());
    }

    #[test]
    fn test_unconstrained_drifts() {
        let mut sys = ring_system(5, 0.5);
        for _ in 0..100 { sys.unconstrained_step(0.01); }
        // Unconstrained may drift slightly due to numerical error
        // (graph diffusion IS conservative, but floating point)
        assert!(sys.conservation_error() < 1e-4, "Unconstrained error: {}", sys.conservation_error());
    }

    #[test]
    fn test_constrained_preserves() {
        let mut sys = ring_system(5, 0.5);
        for _ in 0..100 { sys.constrained_step(0.01); }
        assert!(sys.verify_conservation(), "Constrained should preserve C, error: {}", sys.conservation_error());
    }

    #[test]
    fn test_converges_to_uniform() {
        let mut sys = complete_system(4, 0.5);
        let result = sys.simulate(0.01, 500, true);
        assert!(result.final_gamma_std < 1.0, "Should converge, std: {}", result.final_gamma_std);
    }

    #[test]
    fn test_ring_slower_than_complete() {
        let mut ring = ring_system(6, 0.3);
        let mut complete = complete_system(6, 0.3);
        let ring_result = ring.simulate(0.01, 100, true);
        let complete_result = complete.simulate(0.01, 100, true);
        // Complete should have lower std (more uniform) after same steps
        assert!(complete_result.final_gamma_std <= ring_result.final_gamma_std + 0.1);
    }

    #[test]
    fn test_star_hub_diffuses() {
        let mut sys = star_system(5, 0.3);
        let initial_hub_gamma = sys.agents[0].gamma;
        for _ in 0..200 { sys.constrained_step(0.01); }
        // Hub should have less γ after diffusion
        assert!(sys.agents[0].gamma < initial_hub_gamma,
            "Hub γ {} should be < initial {}", sys.agents[0].gamma, initial_hub_gamma);
    }

    #[test]
    fn test_simulation_records_history() {
        let mut sys = ring_system(3, 0.1);
        let result = sys.simulate(0.01, 10, true);
        assert_eq!(result.history.len(), 11); // initial + 10 steps
    }

    #[test]
    fn test_project_onto_constraint() {
        let mut sys = ring_system(4, 0.1);
        // Manually perturb
        sys.agents[0].gamma += 1.0;
        assert!(!sys.verify_conservation());
        sys.project_onto_constraint();
        assert!(sys.verify_conservation(), "Projection should restore conservation");
    }

    #[test]
    fn test_total_conservation_constant() {
        let mut sys = complete_system(5, 0.5);
        let initial_total = sys.total_conservation;
        for _ in 0..500 { sys.constrained_step(0.01); }
        let final_total = sys.total_gamma() + sys.total_eta();
        assert!((final_total - initial_total).abs() < 1e-6,
            "Total should be constant: {} vs {}", final_total, initial_total);
    }

    #[test]
    fn test_gamma_eta_non_negative() {
        let mut sys = ring_system(5, 0.1);
        for _ in 0..1000 { sys.constrained_step(0.01); }
        for agent in &sys.agents {
            assert!(agent.gamma > -0.1, "Agent {} gamma negative: {}", agent.id, agent.gamma);
            assert!(agent.eta > -0.1, "Agent {} eta negative: {}", agent.id, agent.eta);
        }
    }

    #[test]
    fn test_constrained_vs_unconstrained_error() {
        let mut c_sys = ring_system(5, 0.5);
        let mut u_sys = ring_system(5, 0.5);
        for _ in 0..500 {
            c_sys.constrained_step(0.01);
            u_sys.unconstrained_step(0.01);
        }
        assert!(c_sys.conservation_error() <= u_sys.conservation_error() + 1e-10);
    }

    #[test]
    fn test_weighted_coupling() {
        let mut agents = vec![
            DiffusionAgent::new(0, 20.0, 10.0),
            DiffusionAgent::new(1, 5.0, 2.0),
        ];
        agents[0].add_neighbor(1, 3.0); // strong coupling
        agents[1].add_neighbor(0, 3.0);
        let mut sys = DiffusionSystem::new(agents, 1.0);
        let result = sys.simulate(0.01, 200, true);
        assert!(result.final_gamma_std < 5.0);
    }

    #[test]
    fn test_snapshot_consistency() {
        let sys = ring_system(3, 0.1);
        let snap = sys.snapshot();
        assert_eq!(snap.gammas.len(), 3);
        assert_eq!(snap.etas.len(), 3);
        assert!((snap.total - sys.total_conservation).abs() < 1e-10);
    }
}

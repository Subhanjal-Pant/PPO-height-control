import numpy as np


class LagrangianSolver:
    """Manages adaptive dual multipliers for physical constraints via dual
    gradient ascent, with priority support via per-constraint learning rates.

    NOTE: this class is unchanged from what you already had. The chattering
    regression was NOT in the dual-ascent math here -- update_multipliers()
    just does gradient ascent on whatever mean cost it's handed. The bug was
    upstream, in HOW that mean was computed in train.py's LagrangianCallback
    (it was being averaged over every rollout step, including steps where a
    given direction was inactive and force-zeroed -- see train.py for the
    fix). This file is included for completeness / so you have the full set
    together, not because anything here changed.
    """

    def __init__(self, cost_limits, learning_rates=None, max_lambda: float = 10.0):
        self.limits = {k: float(v) for k, v in cost_limits.items()}
        self.max_lambda = max_lambda

        default_lr = 0.005
        if isinstance(learning_rates, dict):
            self.lrs = {
                k: float(learning_rates.get(k, default_lr))
                for k in cost_limits.keys()
            }
        elif isinstance(learning_rates, (float, int)):
            self.lrs = {k: float(learning_rates) for k in cost_limits.keys()}
        else:
            self.lrs = {k: default_lr for k in cost_limits.keys()}

        self.lambdas = {k: 0.0 for k in cost_limits.keys()}

    def compute_penalized_reward(self, reward, episode_costs):
        penalized_reward = reward
        for key, lambda_val in self.lambdas.items():
            cost_val = episode_costs.get(key, 0.0)
            penalized_reward -= (lambda_val * cost_val)
        return penalized_reward

    def update_multipliers(self, mean_episode_costs):
        for key in self.lambdas:
            observed_cost = mean_episode_costs.get(key, 0.0)
            cost_error = observed_cost - self.limits[key]
            updated_lambda = self.lambdas[key] + (self.lrs[key] * cost_error)
            # Clip between [0.0, max_lambda] to prevent reward signal destruction
            self.lambdas[key] = float(np.clip(updated_lambda, 0.0, self.max_lambda))
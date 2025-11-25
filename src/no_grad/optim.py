from dataclasses import dataclass, field
from typing import Iterable, Optional
import torch
import torch.nn as nn


@dataclass
class OptimizerParams:
    lr: float = 1e-3


@dataclass
class SGDParams(OptimizerParams):
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = False


@dataclass
class AdamParams(OptimizerParams):
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-4


@dataclass
class SampleStrategy:
    temp: float = 1.0


@dataclass
class PopulationAggStrategy:
    weighted_sum: bool | None = True
    sample: SampleStrategy | None = None


@dataclass
class ESParams(OptimizerParams):
    population_size: int = 16
    step_size: float = 0.01
    agg_strategy: PopulationAggStrategy = field(default_factory=PopulationAggStrategy)
    include_parent: bool = True
    persist_parent: bool = True


@dataclass
class OptimizerConfig:
    type: str = "sgd"
    sgd: SGDParams = field(default_factory=SGDParams)
    adam: AdamParams = field(default_factory=AdamParams)
    es: ESParams = field(default_factory=ESParams)

    @staticmethod
    def make_run_id(cfg: "OptimizerConfig") -> str:
        if cfg.type == "sgd":
            return f"sgd-lr{cfg.sgd.lr}"
        elif cfg.type == "es":
            return f"es-lr{cfg.es.lr}-p{cfg.es.population_size}-s{cfg.es.step_size}"
        else:
            return ""


@dataclass
class EvMutation:
    param_seeds: list[int] = field(default_factory=list)
    is_identity: bool = False
    reward: float | None = None
    

class ESOptimizer:

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        step_size=1e-5,
        lr=1e-3,
        population_size=64,
        do_sample=False,
        sample_temp=1.0,
        include_parent=True,
        persist_parent=True,
    ):
        self.params = list(params)
        self.mutations: list[EvMutation] = []
        self.active_mutation: EvMutation | None = None
        self.population_size = population_size
        self.step_size = step_size
        self.lr = lr
        self.do_sample = do_sample
        self.sample_temp = sample_temp
        self.include_parent = include_parent
        self.persist_parent = persist_parent
        self.parent_params = None
        if persist_parent:
            # create separate parameter list with shared data
            self.parent_params = [p.clone() for p in self.params]
            for p, p_parent in zip(self.params, self.parent_params):
                p_parent.data = p.data

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if (
            self.active_mutation is not None and
            self.active_mutation.reward is None
        ):
            raise RuntimeError("exit before collecting reward for active mutation")
        
        # aggregate current mutation set (in case n_mutations
        # is not even with number of loop iterations)
        if self.mutation_index > -1:
            self.aggregate_mutations()

    def _param_delta_iter(self, param_seeds: list[int] | None = None):
        if param_seeds is None:
            param_seeds = [None] * len(self.params)
        elif len(param_seeds) != len(self.params):
            raise RuntimeError("mismatch between number of params and seeds")
        for p, seed in zip(self.params, param_seeds):
            # use cpu random number generator for stability
            if seed is None:
                seed = torch.seed()
            else:
                torch.manual_seed(seed)
            
            yield torch.randn(p.shape, dtype=p.dtype).to(p.device) * self.step_size, seed

    @property
    def mutation_index(self):
        return len(self.mutations) - 1

    @torch.no_grad()
    def revert_mutation(self):
        if not self.active_mutation.is_identity:
            if self.persist_parent:
                # reset data ptr to current parent (non-mutated) tensor
                for p, p_parent in zip(self.params, self.parent_params):
                    p.data = p_parent.data
            else:
                # regenerate deltas from param seeds and subtract to get non-mutated tensor
                for p, (delta_p, _) in zip(self.params, self._param_delta_iter(self.active_mutation.param_seeds)):
                    p -= delta_p
        self.active_mutation = None

    @torch.no_grad()
    def mutate(self):
        if self.include_parent and self.mutation_index == -1:
            # save mutation to reference current unperturbed weights
            new_mutation = EvMutation(is_identity=True)
        else:
            # make new mutation
            param_seeds = []
            for p, (delta_p, seed) in zip(self.params, self._param_delta_iter()):
                if self.persist_parent:
                    # can't use inplace if we have shared data ptr with current parent
                    # instead we create a new tensor from applied delta and update ptr
                    # of mutated weight to point to it
                    p.data = p + delta_p
                else:
                    p += delta_p
                param_seeds.append(seed)
            new_mutation = EvMutation(param_seeds)

        self.mutations.append(new_mutation)
        self.active_mutation = new_mutation

    def reward_step(self, reward: float):
        self.active_mutation.reward = reward
        self.revert_mutation()

        if len(self.mutations) % self.population_size == 0:
            self.aggregate_mutations()

    @torch.no_grad()
    def aggregate_mutations(self):
        if self.active_mutation is not None:
            raise RuntimeError("cannot aggregate while mutation is still active")

        # print("aggregating mutations...", end="")

        if self.do_sample:
            rewards = torch.tensor([m.reward for m in self.mutations])
            if self.sample_temp > 0:
                probs = torch.softmax(rewards / self.sample_temp, dim=0)
                sampled_index: int = torch.multinomial(probs, 1).item()
            else:
                sampled_index: int = torch.argmax(rewards).item()
            sampled = self.mutations[sampled_index]
            if not sampled.is_identity:
                for p, (p_delta, _) in zip(self.params, self._param_delta_iter(sampled.param_seeds)):
                    # if persist_parent is True, p and p_parent should share data
                    # so we need only update one
                    p += p_delta
        else:       
            # weighted avg mutations based on their reward z-score
            # all mutations should have reward set by now
            n = len(self.mutations)
            mean_reward = sum(m.reward for m in self.mutations) / n
            var_reward = sum((m.reward - mean_reward) ** 2 for m in self.mutations) / n
            z_scores = [(m.reward - mean_reward) / (var_reward ** 0.5) for m in self.mutations if not m.is_identity]
    
            deltas = [self._param_delta_iter(m.param_seeds) for m in self.mutations if not m.is_identity]
            for p in self.params:
                p_deltas, _ = zip(*[next(d) for d in deltas])
    
                for p_delta, z in zip(p_deltas, z_scores):
                    p += p_delta * self.lr * z / n

        self.mutations = []
        # print("done")


def get_optimizer(
    params: Iterable[nn.parameter.Parameter],
    config: OptimizerConfig,
) -> torch.optim.Optimizer | ESOptimizer:
    if config.type == "es":
        sample_temp = (config.es.agg_strategy.sample.temp if
                       config.es.agg_strategy.sample is not None else 1.0)
        return ESOptimizer(
            params,
            step_size=config.es.step_size,
            lr=config.es.lr,
            population_size=config.es.population_size,
            do_sample=config.es.agg_strategy.sample is not None,
            sample_temp=sample_temp,
            include_parent=config.es.include_parent,
            persist_parent=config.es.persist_parent,
        )
    elif config.type == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.sgd.lr,
            momentum=config.sgd.momentum,
            weight_decay=config.sgd.weight_decay,
            nesterov=config.sgd.nesterov,
        )
    elif config.type == "adam":
        return torch.optim.AdamW(
            params,
            lr=config.adam.lr,
            betas=config.adam.betas,
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay,
        )
    else:
        raise RuntimeError("no optimizer config specified")

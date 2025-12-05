from dataclasses import dataclass, field
from typing import Iterable
import torch
import torch.nn as nn
import math
from tqdm.auto import tqdm
import warnings


@dataclass
class OptimizerParams:
    lr: float = 1e-3
    lr_gamma: float = 1.0  # no lr decay
    step_gamma: float = 1.0  # no step size decay


@dataclass
class MomentumParams(OptimizerParams):
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
    temp: float = 0.0


@dataclass
class PopulationAggStrategy:
    weighted_sum: bool | None = True
    sample: SampleStrategy = field(default_factory=SampleStrategy)


@dataclass
class ESParams(OptimizerParams):
    population_size: int = 16
    step_size: float = 0.01
    agg_strategy: PopulationAggStrategy = field(default_factory=PopulationAggStrategy)
    include_parent: bool = True
    persist_parent: bool = True


@dataclass
class ESAdamParams(ESParams):
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-4


@dataclass
class OptimizerConfig:
    type: str = "sgd"
    sgd: MomentumParams = field(default_factory=MomentumParams)
    adam: AdamParams = field(default_factory=AdamParams)
    es: ESParams = field(default_factory=ESParams)
    es_adam: ESAdamParams = field(default_factory=ESAdamParams)

    @staticmethod
    def make_run_id(cfg: "OptimizerConfig") -> str:
        if cfg.type == "sgd":
            return f"sgd-lr{cfg.sgd.lr}"
        elif cfg.type == "es":
            if not cfg.es.agg_strategy.weighted_sum:
                return f"es-sample_t{cfg.es.agg_strategy.sample.temp}-p{cfg.es.population_size}-s{cfg.es.step_size}"
            else:
                lr_gamma = f"-lr_gamma{cfg.es.lr_gamma}" if cfg.es.lr_gamma != 1.0 else ""
                ss_gamma = f"-ss_gamma{cfg.es.step_gamma}" if cfg.es.step_gamma != 1.0 else ""
                return f"es-lr{cfg.es.lr}-p{cfg.es.population_size}-s{cfg.es.step_size}{lr_gamma}{ss_gamma}"
        elif cfg.type == "adam":
            return f"adam-lr{cfg.adam.lr}-b{cfg.adam.betas[0]}_{cfg.adam.betas[1]}-w{cfg.adam.weight_decay}-e{cfg.adam.eps}"
        elif cfg.type == "es_adam":
            lr_gamma = f"-lr_gamma{cfg.es_adam.lr_gamma}" if cfg.es_adam.lr_gamma != 1.0 else ""
            ss_gamma = f"-ss_gamma{cfg.es_adam.step_gamma}" if cfg.es_adam.step_gamma != 1.0 else ""
            return f"es_adam-lr{cfg.es_adam.lr}-p{cfg.es_adam.population_size}-s{cfg.es_adam.step_size}-b{cfg.es_adam.betas[0]}_{cfg.es_adam.betas[1]}-w{cfg.es_adam.weight_decay}-e{cfg.es_adam.eps}{lr_gamma}{ss_gamma}"
        elif cfg.type == "adamutate":
            lr_gamma = f"-lr_gamma{cfg.es_adam.lr_gamma}" if cfg.es_adam.lr_gamma != 1.0 else ""
            ss_gamma = f"-ss_gamma{cfg.es_adam.step_gamma}" if cfg.es_adam.step_gamma != 1.0 else ""
            return f"adamutate_v3-lr{cfg.es_adam.lr}-p{cfg.es_adam.population_size}-s{cfg.es_adam.step_size}-b{cfg.es_adam.betas[0]}_{cfg.es_adam.betas[1]}-w{cfg.es_adam.weight_decay}-e{cfg.es_adam.eps}{lr_gamma}{ss_gamma}"
        else:
            return ""


@dataclass
class EvMutation:
    param_seeds: list[int] = field(default_factory=list)
    is_identity: bool = False
    reward: float | None = None
    eval_count: int = 0
    

class ESOptimizer:

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter] = [],
        step_size: float = 1e-5,
        lr: float = 1e-3,
        population_size: int = 64,
        do_sample: bool = False,
        sample_temp: float = 0.0,
        include_parent: bool = True,
        persist_parent: bool = True,
        use_adam: bool = False,
        ada_mutate: bool = False,  # whether to apply adam to mutation deltas
        betas: tuple = (0.9, 0.999),
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        lr_gamma: float = 1.0,
        step_gamma: float = 1.0,
        r_accum_steps: int = 1,
        param_groups: list[dict] = [],
    ):
        if not (param_groups or params):
            raise ValueError("must provide either param_groups or params") 
        self.param_groups = param_groups
        self.r_accum_steps = r_accum_steps
        self.step = 1
        self.r_accum_step = 0
        self.accum_r = 0  # accumulated reward
        self.params = list(params) if params else param_groups[0]["params"]
        self.mutations: list[EvMutation] = []
        self.active_mutation: EvMutation | None = None
        self.population_size = population_size
        self.step_size = step_size
        self.lr = lr
        self.do_sample = do_sample
        self.sample_temp = sample_temp
        self.include_parent = include_parent
        self.persist_parent = persist_parent
        self.use_adam = use_adam
        self.ada_mutate = ada_mutate
        self.betas = betas
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.parent_params = None
        self.lr_gamma = lr_gamma
        self.step_gamma = step_gamma
        if persist_parent:
            with torch.no_grad():
                # create separate parameter list with shared data
                self.parent_params = [p.clone() for p in self.params]
            for p, p_parent in zip(self.params, self.parent_params):
                p_parent.data = p.data
        if param_groups:
            # set params from first param group
            self.lr = param_groups[0]["lr"]
            if self.use_adam:
                self.betas = param_groups[0]["betas"]
                self.weight_decay = param_groups[0]["weight_decay"]
                self.epsilon = param_groups[0]["eps"]
        if use_adam or ada_mutate:
            # initialize first and second moments for each param
            self.exp_avg = [torch.zeros_like(p) for p in self.params]
            self.exp_avg_sq = [torch.zeros_like(p) for p in self.params]
        # elif ada_mutate:
        #     self.exp_avg = [torch.ones_like(p) * step_size for p in self.params]
        #     self.exp_avg_sq = [torch.ones_like(p) * (step_size ** 2) for p in self.params]
        self.aggregation_metrics = {}

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

    def _sync_lr(self):
        if self.param_groups:
            # sync learning rate with param group value (in case updated by external lr scheduler)
            self.lr = self.param_groups[0]["lr"]

    @staticmethod
    def from_torch_optim(optim: torch.optim.Optimizer, **kwargs) -> "ESOptimizer":
        if isinstance(optim, torch.optim.SGD):
            return ESOptimizer(
                param_groups=optim.param_groups,
                **kwargs,
            )
        elif isinstance(optim, torch.optim.AdamW):
            return ESOptimizer(
                use_adam=True,
                param_groups=optim.param_groups,
                **kwargs,
            )
        else:
            raise RuntimeError(f"conversion to ESOptimizer not supported for {type(optim)}")

    def _param_delta_iter(self, param_seeds: list[int] | None = None):
        if param_seeds is None:
            param_seeds = [None] * len(self.params)
        elif len(param_seeds) != len(self.params):
            raise RuntimeError("mismatch between number of params and seeds")
        
        if self.ada_mutate:
            bc1 = 1 - self.betas[0] ** self.step
            bc2 = 1 - self.betas[1] ** self.step

        for i, (p, seed) in enumerate(zip(self.params, param_seeds)):
            # use cpu random number generator for stability
            if seed is None:
                seed = torch.seed()
            else:
                torch.manual_seed(seed)

            if self.ada_mutate:
                # scale mutation by adam moments
                numerator = self.exp_avg[i] + self.epsilon
                denominator = self.exp_avg_sq[i].sqrt() + self.epsilon
                adapted_step = (math.sqrt(bc2) / bc1) * (numerator / denominator)
                yield torch.randn(p.shape, dtype=p.dtype).to(p.device) * adapted_step * self.step_size, seed
            else:
                yield torch.randn(p.shape, dtype=p.dtype).to(p.device) * self.step_size, seed

    @property
    def mutation_index(self):
        return len(self.mutations) - 1
    
    def _is_first_accum_step(self):
        return self.r_accum_step == 0
    
    def _is_last_accum_step(self):
        return self.r_accum_step == self.r_accum_steps - 1

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
        self._sync_lr() # sync lr before model forward pass
        if self._is_first_accum_step():
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

            new_mutation.reward = 0.0  # initialize for accumulation
            self.mutations.append(new_mutation)
            self.active_mutation = new_mutation

    def is_batch_end(self):
        return self._is_last_accum_step() and len(self.mutations) % self.population_size == 0

    def is_batch_start(self):
        return self._is_first_accum_step() and len(self.mutations) % self.population_size == 0

    def reward_step(self, reward: float):
        # TODO handle uneven batches
        self.active_mutation.reward += reward
        self.active_mutation.eval_count += 1

        # print(f"{self.r_accum_step=}/{self.r_accum_steps-1}, {len(self.mutations)}/{self.population_size} mutations")
        if self._is_last_accum_step():
            self.revert_mutation()
            if self.is_batch_end():
                self.aggregate_mutations()
                self.lr_step()

        self.r_accum_step = (self.r_accum_step + 1) % self.r_accum_steps

    def state_dict(self):
        # for torch optim compatibility
        return {}

    @torch.no_grad()
    def aggregate_mutations(self):
        if self.active_mutation is not None:
            raise RuntimeError("cannot aggregate while mutation is still active")

        # print("aggregating mutations...", end="")

        if self.do_sample:
            rewards = torch.tensor([m.reward / m.eval_count for m in self.mutations])
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
            rewards = [m.reward / m.eval_count for m in self.mutations]
            mean_reward = sum(rewards) / n
            var_reward = sum((r - mean_reward) ** 2 for r in rewards) / n
            z_scores = [(r - mean_reward) / (var_reward ** 0.5) for r, m in zip(rewards, self.mutations) if not m.is_identity]

            # record improved mutation count
            if self.include_parent:
                # first mutation is identity (parent)
                self.aggregation_metrics["improved_ratio"] = sum(
                    1 for r, m in zip(rewards, self.mutations)
                    if (not m.is_identity) and r > rewards[0]
                ) / (n - 1)
                self.aggregation_metrics["relative_fitness"] = sum(
                    (r - rewards[0]) / abs(mean_reward) for r, m in zip(rewards, self.mutations)
                    if not m.is_identity
                ) / (n - 1)

            if self.use_adam:
                bc1 = 1 - self.betas[0] ** self.step
                bc2 = 1 - self.betas[1] ** self.step

            deltas = [self._param_delta_iter(m.param_seeds) for m in self.mutations if not m.is_identity]
            for i, p in enumerate(tqdm(self.params)):
                p_deltas, _ = zip(*[next(d) for d in deltas])
    
                agg_p_delta = torch.zeros_like(p)
                for p_delta, z in zip(p_deltas, z_scores):
                    agg_p_delta += p_delta * z / n / self.step_size

                if self.use_adam or self.ada_mutate:
                    # decay first and second moments
                    self.exp_avg[i] = self.betas[0] * self.exp_avg[i] + (1 - self.betas[0]) * agg_p_delta
                    self.exp_avg_sq[i] = self.betas[1] * self.exp_avg_sq[i] + (1 - self.betas[1]) * (agg_p_delta ** 2)
                    
                if self.use_adam:
                    denominator = self.exp_avg_sq[i].sqrt() + self.epsilon
                    step_size = self.lr * math.sqrt(bc2) / bc1
                    p += step_size * (self.exp_avg[i] / denominator - self.weight_decay * p)
                else:
                    p += agg_p_delta * self.lr

        self.mutations = []
        self.step += 1
        # print("done")

    def lr_step(self):
        if self.lr_gamma != 1.0:
            if self.param_groups:
                warnings.warn("decayed learning rate will be overriden by param_group value")
            self.lr *= self.lr_gamma
        if self.step_gamma != 1.0:
            self.step_size *= self.step_gamma


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
            do_sample=not config.es.agg_strategy.weighted_sum,
            sample_temp=sample_temp,
            include_parent=config.es.include_parent,
            persist_parent=config.es.persist_parent,
            lr_gamma=config.es.lr_gamma,
            step_gamma=config.es.step_gamma,
        )
    elif config.type == "es_adam":
        sample_temp = (config.es_adam.agg_strategy.sample.temp if
                       config.es_adam.agg_strategy.sample is not None else 1.0)
        return ESOptimizer(
            params,
            step_size=config.es_adam.step_size,
            lr=config.es_adam.lr,
            population_size=config.es_adam.population_size,
            do_sample=not config.es_adam.agg_strategy.weighted_sum,
            sample_temp=sample_temp,
            include_parent=config.es_adam.include_parent,
            persist_parent=config.es_adam.persist_parent,
            use_adam=True,
            betas=config.es_adam.betas,
            epsilon=config.es_adam.eps,
            weight_decay=config.es_adam.weight_decay,
            lr_gamma=config.es_adam.lr_gamma,
            step_gamma=config.es_adam.step_gamma,
        )
    elif config.type == "adamutate":
        sample_temp = (config.es_adam.agg_strategy.sample.temp if
                       config.es_adam.agg_strategy.sample is not None else 1.0)
        return ESOptimizer(
            params,
            step_size=config.es_adam.step_size,
            lr=config.es_adam.lr,
            population_size=config.es_adam.population_size,
            do_sample=not config.es_adam.agg_strategy.weighted_sum,
            sample_temp=sample_temp,
            include_parent=config.es_adam.include_parent,
            persist_parent=config.es_adam.persist_parent,
            ada_mutate=True,
            use_adam=True,
            betas=config.es_adam.betas,
            epsilon=config.es_adam.eps,
            weight_decay=config.es_adam.weight_decay,
            lr_gamma=config.es_adam.lr_gamma,
            step_gamma=config.es_adam.step_gamma,
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
        raise RuntimeError("invalid optimizer config")

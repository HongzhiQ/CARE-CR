import math
from typing import Callable, Dict, List, Tuple

import numpy as np


NUM_DIMS = 6
DIM_NAMES = ["同理心", "积极性", "理性", "可执行性", "具体性", "可读性"]


def _normalize_g_vector(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    s = float(arr.sum())
    if s <= 0:
        arr = np.ones(NUM_DIMS, dtype=np.float32) / float(NUM_DIMS)
    else:
        arr = arr / s
    return arr


ACTION_DEFS: Dict[str, Dict] = {
    "E1": {
        "step_index": 0,
        "description": "强共情开场，表达理解和陪伴。",
        "g": _normalize_g_vector([0.55, 0.20, 0.05, 0.00, 0.05, 0.15]),
    },
    "E2": {
        "step_index": 0,
        "description": "共情并进行正常化。",
        "g": _normalize_g_vector([0.45, 0.25, 0.05, 0.00, 0.05, 0.20]),
    },
    "E3": {
        "step_index": 0,
        "description": "轻共情并引出认知探索。",
        "g": _normalize_g_vector([0.30, 0.20, 0.30, 0.00, 0.05, 0.15]),
    },
    "C1": {
        "step_index": 1,
        "description": "证据检验，核查支持与反证。",
        "g": _normalize_g_vector([0.05, 0.05, 0.55, 0.00, 0.20, 0.15]),
    },
    "C2": {
        "step_index": 1,
        "description": "替代解释，考虑其他可能解释。",
        "g": _normalize_g_vector([0.10, 0.25, 0.25, 0.25, 0.00, 0.20]),
    },
    "C3": {
        "step_index": 1,
        "description": "概率与严重性重估。",
        "g": _normalize_g_vector([0.05, 0.10, 0.60, 0.00, 0.15, 0.10]),
    },
    "C4": {
        "step_index": 1,
        "description": "标签重构，将自我否定标签转化为具体情境困难。",
        "g": _normalize_g_vector([0.20, 0.20, 0.25, 0.00, 0.15, 0.20]),
    },
    "S1": {
        "step_index": 2,
        "description": "对比式总结，对比旧想法与新的平衡视角。",
        "g": _normalize_g_vector([0.05, 0.15, 0.30, 0.00, 0.25, 0.25]),
    },
    "S2": {
        "step_index": 2,
        "description": "行动建议总结，提炼可执行小步骤。",
        "g": _normalize_g_vector([0.20, 0.15, 0.05, 0.25, 0.25, 0.10]),
    },
    "S3": {
        "step_index": 2,
        "description": "情感安抚式总结，强化希望感与安全感。",
        "g": _normalize_g_vector([0.40, 0.30, 0.05, 0.00, 0.10, 0.15]),
    },
}


STEP_ACTIONS: Dict[int, List[str]] = {
    0: ["E1", "E2", "E3"],
    1: ["C1", "C2", "C3", "C4"],
    2: ["S1", "S2", "S3"],
}


class MCTSConfig:
    def __init__(
        self,
        num_simulations: int = 64,
        c_puct: float = 1.0,
        tau: float = 1.0,
        beta_E: float = 2.0,
        beta_C: float = 2.0,
        beta_S: float = 2.0,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.tau = tau
        self.beta_E = beta_E
        self.beta_C = beta_C
        self.beta_S = beta_S


class MCTSNode:
    def __init__(
        self,
        step_index: int,
        parent: "MCTSNode" = None,
        action_from_parent: str = None,
        priors: Dict[str, float] = None,
    ):
        self.step_index = step_index
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: Dict[str, "MCTSNode"] = {}
        self.N = 0
        self.N_a: Dict[str, int] = {}
        self.Q_a: Dict[str, float] = {}
        self.P_a: Dict[str, float] = priors or {}

    def is_terminal(self) -> bool:
        return self.step_index >= 3


class DG_HMCTS:
    def __init__(self, config: MCTSConfig):
        self.config = config

    def _compute_priors(self, lambda_clin: np.ndarray) -> Dict[str, Dict[str, float]]:
        lam = np.asarray(lambda_clin, dtype=np.float32)
        s = float(lam.sum())
        if s <= 0:
            lam = np.ones(NUM_DIMS, dtype=np.float32) / float(NUM_DIMS)
        else:
            lam = lam / s
        priors_by_layer: Dict[str, Dict[str, float]] = {"E": {}, "C": {}, "S": {}}
        for code, info in ACTION_DEFS.items():
            g_vec = info["g"]
            align = float((lam * g_vec).sum())
            if info["step_index"] == 0:
                priors_by_layer["E"][code] = align
            elif info["step_index"] == 1:
                priors_by_layer["C"][code] = align
            elif info["step_index"] == 2:
                priors_by_layer["S"][code] = align
        for layer, d in priors_by_layer.items():
            if not d:
                continue
            if layer == "E":
                beta = self.config.beta_E
            elif layer == "C":
                beta = self.config.beta_C
            else:
                beta = self.config.beta_S
            keys = list(d.keys())
            vals = np.array([d[k] for k in keys], dtype=np.float32)
            vals = beta * vals
            vals = vals - float(vals.max())
            vals = np.exp(vals)
            vals = vals / (float(vals.sum()) + 1e-12)
            for k, v in zip(keys, vals):
                d[k] = float(v)
        return priors_by_layer

    def _available_actions(self, node: MCTSNode) -> List[str]:
        return STEP_ACTIONS.get(node.step_index, [])

    def _select(self, root: MCTSNode) -> Tuple[MCTSNode, str]:
        node = root
        while True:
            actions = self._available_actions(node)
            if not actions:
                return node, None
            untried = [a for a in actions if a not in node.children]

            def puct_score(a: str) -> float:
                na = node.N_a.get(a, 0)
                qa = node.Q_a.get(a, 0.0)
                pa = node.P_a.get(a, 0.0)
                u = self.config.c_puct * pa * math.sqrt(node.N) / (1.0 + float(na))
                return float(qa + u)

            if untried:
                tau = float(self.config.tau)
                if tau <= 0:
                    tau = 1.0
                scores = np.array([puct_score(a) for a in untried], dtype=np.float32)
                scores = (1.0 / tau) * scores
                scores = scores - float(scores.max())
                probs = np.exp(scores)
                probs = probs / (float(probs.sum()) + 1e-12)
                r = float(np.random.rand())
                acc = 0.0
                chosen = untried[-1]
                for a, p in zip(untried, probs):
                    acc += float(p)
                    if r <= acc:
                        chosen = a
                        break
                return node, chosen

            best_action = None
            best_score = -1e30
            for a in actions:
                score = puct_score(a)
                if score > best_score:
                    best_score = score
                    best_action = a
            node = node.children[best_action]

    def _extract_path_codes(self, node: MCTSNode) -> List[str]:
        codes: List[str] = []
        cur = node
        while cur is not None and cur.action_from_parent is not None:
            codes.append(cur.action_from_parent)
            cur = cur.parent
        codes.reverse()
        return codes

    def _backpropagate(self, leaf: MCTSNode, reward: float) -> None:
        node = leaf
        while node.parent is not None:
            parent = node.parent
            action = node.action_from_parent
            parent.N += 1
            na = parent.N_a.get(action, 0) + 1
            parent.N_a[action] = na
            qa_prev = parent.Q_a.get(action, 0.0)
            parent.Q_a[action] = qa_prev + (reward - qa_prev) / float(na)
            node = parent
        node.N += 1

    def _complete_path_codes(self, start_node: MCTSNode, priors_by_layer: Dict[str, Dict[str, float]]) -> List[str]:
        codes = self._extract_path_codes(start_node)
        step = start_node.step_index
        while step < 3:
            if step == 0:
                layer = "E"
            elif step == 1:
                layer = "C"
            else:
                layer = "S"
            actions = STEP_ACTIONS[step]
            priors = priors_by_layer[layer]
            probs = np.array([priors[a] for a in actions], dtype=np.float32)
            probs = probs / (float(probs.sum()) + 1e-12)
            r = float(np.random.rand())
            acc = 0.0
            chosen = actions[-1]
            for a, p in zip(actions, probs):
                acc += float(p)
                if r <= acc:
                    chosen = a
                    break
            codes.append(chosen)
            step += 1
        return codes

    def run(
        self,
        text: str,
        lambda_clin: List[float],
        gen_fn: Callable[[str, List[str]], str],
        reward_fns: List[Callable[[str, str], float]],
        num_simulations: int = None,
    ) -> List[Dict]:
        if len(lambda_clin) != NUM_DIMS:
            raise ValueError("lambda_clin length must be NUM_DIMS")
        if len(reward_fns) != NUM_DIMS:
            raise ValueError("reward_fns length must be NUM_DIMS")
        priors_by_layer = self._compute_priors(np.asarray(lambda_clin, dtype=np.float32))
        root_priors = priors_by_layer.get("E", {})
        root = MCTSNode(step_index=0, parent=None, action_from_parent=None, priors=root_priors)
        results: List[Dict] = []
        sims = num_simulations or self.config.num_simulations
        print(f"[MCTS] Start searching, num_simulations={sims}")
        for sim_idx in range(sims):
            leaf, action = self._select(root)
            if action is not None:
                next_step = leaf.step_index + 1
                if next_step == 1:
                    priors = priors_by_layer.get("C", {})
                elif next_step == 2:
                    priors = priors_by_layer.get("S", {})
                else:
                    priors = {}
                child = MCTSNode(step_index=next_step, parent=leaf, action_from_parent=action, priors=priors)
                leaf.children[action] = child
                target_node = child
            else:
                target_node = leaf
            path_codes = self._complete_path_codes(target_node, priors_by_layer)
            y = gen_fn(text, path_codes)
            k_index = int(np.random.randint(0, NUM_DIMS))
            primary_reward = float(reward_fns[k_index](text, y))
            self._backpropagate(target_node, primary_reward)
            all_rewards: List[float] = []
            for fn in reward_fns:
                all_rewards.append(float(fn(text, y)))
            results.append(
                {
                    "text": y,
                    "path": path_codes,
                    "rewards": all_rewards,
                }
            )
            if (sim_idx + 1) == sims or ((sim_idx + 1) % max(1, sims // 5) == 0):
                print(f"[MCTS]  {sim_idx + 1}/{sims} simulation has been completed")
        return results

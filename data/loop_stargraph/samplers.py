"""Walk samplers for the loop_stargraph synthetic dataset.

Public surface (used by prepare.py):
  - build_star_graph(num_paths, path_length, max_nodes) -> (path, edge_list, source, goal)
  - sample_random_walk_with_reset(...) -> (walk, edge_list, source, goal)
  - sample_pure_random_walk(...)        -> (walk, edge_list, source, goal)

Both samplers return the canonical (walk, edge_list, source, goal) tuple that
prepare.py serialises into the file format.

Two walk types, matching the paper:

  random_walk_with_reset
      Section 3 of the paper. At every step the walker resets to the source
      with probability `reset_probability`. At the root, with probability
      `exploration_probability` it instead takes a uniformly-chosen wrong arm
      (otherwise it advances on the canonical path with the remaining
      probability). Non-goal leaves return to source with probability 1.

  pure_random_walk
      Section 4 + Appendix E. No per-step reset. From source it goes to the
      root deterministically. At the root it samples one of the visible arms
      (all wrong arms, or correct + a random subset of `num_wrong_arms_explored`
      wrong arms when set). The per-visit distribution combines `margin`
      (visited arms get reduced probability) and `correct_arm_probability`
      (correct arm gets a fixed mass, the rest split over visible wrong arms).
      Internal arm transitions are deterministic; non-goal leaves reset.

Both walks accept `looping_data_frac` (Appendix F): with this probability the
sample is replaced by a deterministic looping trace that fills the token
budget (source<->root for random_walk_with_reset, source->root->wrong-arm
cycle for pure_random_walk).

The order of RNG consumption per sample is the same regardless of which
sampler runs:

  1. random.shuffle(nodes)
  2. random.shuffle(edge_list)
  3. random.random()           # only if looping_data_frac > 0
  4a. (loop branch fires)   random.choice(wrong_heads)  # only for pure_random_walk
  4b. (regular branch)      random.sample(wrong_heads, k) # only for pure_random_walk + subset
  5. random.choices(...) per walk transition (or per root visit, for the
     margin sampler)

Keeping this call order stable makes seeded data generation reproducible
across refactors.
"""

import random
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_star_graph(num_paths: int, path_length: int, max_nodes: int):
    """Build a star graph G(num_paths, path_length) with an explicit source node.

    Layout: source -> root -> {num_paths arms, each of length path_length - 1}.
    The first arm built is the canonical correct arm; its terminal node is the
    goal. Node IDs are sampled without replacement from range(max_nodes) and
    the edge list is shuffled before return.

    Required node count: 2 + num_paths * (path_length - 1).
    """
    required_nodes = 2 + num_paths * (path_length - 1)
    if required_nodes > max_nodes:
        raise ValueError(
            f"Not enough nodes: need {required_nodes} for G({num_paths},{path_length}), "
            f"but max_nodes={max_nodes}."
        )

    nodes = list(range(max_nodes))
    random.shuffle(nodes)

    source = nodes.pop()
    root = nodes.pop()

    edge_list: List[Tuple[int, int]] = [(source, root)]
    path: List[int] = [source, root]
    goal: Optional[int] = None

    for arm_idx in range(num_paths):
        cur = root
        for _ in range(path_length - 1):
            nxt = nodes.pop()
            edge_list.append((cur, nxt))
            cur = nxt
            if arm_idx == 0:
                path.append(cur)
        if arm_idx == 0:
            goal = cur

    assert goal is not None
    random.shuffle(edge_list)

    return path, edge_list, source, goal


# ---------------------------------------------------------------------------
# Walk-string helpers (also used by prepare.py for tokenisation)
# ---------------------------------------------------------------------------

def make_prefix_string(edge_list, source, goal) -> str:
    return "|".join([",".join([str(i) for i in edge]) for edge in edge_list]) + f"/{source},{goal}="


def make_suffix_string(walk) -> str:
    return ",".join([str(i) for i in walk])


def make_full_string(edge_list, source, goal, walk) -> str:
    return make_prefix_string(edge_list, source, goal) + make_suffix_string(walk)


# ---------------------------------------------------------------------------
# Generic transition-matrix walker
# ---------------------------------------------------------------------------

def _sample_walk_from_matrix(start_node, transition_matrix, max_walk_tokens, tokenizer):
    """Walk a Markov chain until the token budget is exhausted or an absorbing
    state is reached.
    """
    walk: List[int] = []
    if max_walk_tokens is not None:
        # Reserve two tokens for the response separator and EOS.
        max_walk_tokens -= 2

    current = start_node
    while True:
        walk.append(current)

        if max_walk_tokens is not None:
            if tokenizer is None:
                raise ValueError("Tokenizer is required when max_walk_tokens is specified")
            # Token count equals len(walk) + 1 (one token per node id, plus the
            # trailing EOS appended by tokenizer.tokenize). Verified against the
            # tokenizer's encode() path which skips commas and emits one token
            # per integer string. This O(1) check avoids rebuilding and
            # re-tokenizing the full walk at every step.
            if len(walk) + 1 >= max_walk_tokens:
                break

        transitions = transition_matrix.get(current, [])
        if not transitions:
            break  # absorbing

        choices = [n for n, _ in transitions]
        weights = [p for _, p in transitions]
        if not any(w > 0 for w in weights):
            current = transitions[-1][0]
        else:
            current = random.choices(choices, weights=weights, k=1)[0]
    return walk


# ---------------------------------------------------------------------------
# Section 3: random_walk_with_reset
# ---------------------------------------------------------------------------

def _build_random_walk_with_reset_matrix(
    path: List[int],
    edge_list: List[Tuple[int, int]],
    reset_probability: float,
    exploration_probability: float,
) -> Dict[int, List[Tuple[int, float]]]:
    """Transition matrix for random_walk_with_reset.

    At every non-goal node on the canonical path the walker can reset to source
    with probability `reset_probability` and advance with `1 - reset_probability`.
    At the root, a wrong arm can be taken with probability
    `exploration_probability` (split uniformly across wrong arms when more than
    one exists). Non-goal leaves (wrong-arm tips) return to source.

    Wrong-arm branches remain in the matrix when
    `exploration_probability == 0`, but they are unreachable and therefore do
    not consume additional random draws.
    """
    if len(path) < 3:
        raise ValueError("random_walk_with_reset requires path length >= 3 (source, root, first arm node)")
    if not (0.0 <= reset_probability <= 1.0):
        raise ValueError("reset_probability must lie in [0, 1]")
    if exploration_probability < 0:
        raise ValueError("exploration_probability must be >= 0")
    if reset_probability + exploration_probability >= 1.0:
        raise ValueError("reset_probability + exploration_probability must be < 1 (correct-arm probability must be > 0)")

    source = path[0]
    root = path[1]
    goal = path[-1]
    correct_head = path[2]

    children: Dict[int, List[int]] = {}
    for u, v in edge_list:
        children.setdefault(u, []).append(v)

    arm_heads = children.get(root, [])
    k = len(arm_heads)
    if k == 0:
        raise ValueError("Root has no outgoing arms; invalid star graph")
    if k == 1 and exploration_probability > 0:
        raise ValueError("exploration_probability must be 0 when only one arm exists")
    if correct_head not in arm_heads:
        raise ValueError("Canonical correct head is not a child of root")

    wrong_heads = [h for h in arm_heads if h != correct_head]

    def follow_chain(head):
        chain = [head]
        cur = head
        while True:
            outs = children.get(cur, [])
            if len(outs) != 1:
                break
            chain.append(outs[0])
            cur = outs[0]
        return chain

    correct_chain = follow_chain(correct_head)
    wrong_chains = [follow_chain(h) for h in wrong_heads]

    matrix: Dict[int, List[Tuple[int, float]]] = {}

    # Source transitions: reset (=source self-loop, written as (source, p)) + forward
    src_transitions: List[Tuple[int, float]] = []
    if reset_probability > 0.0:
        src_transitions.append((source, reset_probability))
    forward_prob = 1.0 - reset_probability
    if forward_prob > 0.0:
        src_transitions.append((root, forward_prob))
    matrix[source] = src_transitions

    # Root transitions: reset + exploration over wrong arms + correct forward
    root_transitions: List[Tuple[int, float]] = []
    if reset_probability > 0.0:
        root_transitions.append((source, reset_probability))
    if k > 1 and exploration_probability > 0.0:
        share = exploration_probability / (k - 1)
        for wh in wrong_heads:
            if share > 0.0:
                root_transitions.append((wh, share))
    prob_correct = 1.0 - reset_probability - exploration_probability
    if prob_correct <= 0.0:
        raise ValueError("Correct-arm probability must be > 0")
    root_transitions.append((correct_head, prob_correct))
    root_transitions = [(n, p) for n, p in root_transitions if p > 0.0]
    total_root = sum(p for _, p in root_transitions)
    for i, (n, p) in enumerate(root_transitions):
        root_transitions[i] = (n, p / total_root)
    matrix[root] = root_transitions

    # Internal nodes of correct arm
    for i in range(len(correct_chain) - 1):
        node = correct_chain[i]
        nxt = correct_chain[i + 1]
        transitions: List[Tuple[int, float]] = []
        if reset_probability > 0.0:
            transitions.append((source, reset_probability))
        fwd = 1.0 - reset_probability
        if fwd > 0.0:
            transitions.append((nxt, fwd))
        matrix[node] = transitions

    matrix[goal] = []

    # Wrong arms
    for chain in wrong_chains:
        if not chain:
            continue
        for i in range(len(chain) - 1):
            node = chain[i]
            nxt = chain[i + 1]
            transitions = []
            if reset_probability > 0.0:
                transitions.append((source, reset_probability))
            fwd = 1.0 - reset_probability
            if fwd > 0.0:
                transitions.append((nxt, fwd))
            matrix[node] = transitions
        leaf = chain[-1]
        if leaf not in matrix:
            matrix[leaf] = [(source, 1.0)]

    return matrix


# ---------------------------------------------------------------------------
# Section 4 + Appendix E: pure_random_walk
# ---------------------------------------------------------------------------

def _build_pure_random_walk_subset_matrix(
    path: List[int],
    edge_list: List[Tuple[int, int]],
    correct_arm_probability: float,
    num_wrong_arms_explored: int,
) -> Dict[int, List[Tuple[int, float]]]:
    """Transition matrix for pure_random_walk with a fixed random subset of
    wrong arms (Appendix E G(10,5)).

    The correct arm head gets probability `correct_arm_probability`; exactly
    `num_wrong_arms_explored` wrong arms are sampled uniformly without
    replacement and share the remaining `1 - correct_arm_probability` mass.
    All other wrong arms get probability 0 for this sample.

    The visible wrong-arm subset is sampled once per generated example.
    """
    if len(path) < 3:
        raise ValueError("pure_random_walk + subset requires path length >= 3")
    if not (0.0 < correct_arm_probability < 1.0):
        raise ValueError("correct_arm_probability must lie in (0, 1)")
    if num_wrong_arms_explored is None or num_wrong_arms_explored < 1:
        raise ValueError("num_wrong_arms_explored must be a positive integer")

    source = path[0]
    root = path[1]
    goal = path[-1]
    correct_head = path[2]

    children: Dict[int, List[int]] = {}
    for u, v in edge_list:
        children.setdefault(u, []).append(v)

    arm_heads = children.get(root, [])
    if not arm_heads:
        raise ValueError("Root has no outgoing arms")
    if correct_head not in arm_heads:
        raise ValueError("Canonical correct head must be a child of root")

    wrong_heads = [h for h in arm_heads if h != correct_head]
    if num_wrong_arms_explored > len(wrong_heads):
        raise ValueError(
            f"Requested {num_wrong_arms_explored} wrong arms but only {len(wrong_heads)} available"
        )

    def follow_chain(head):
        chain = [head]
        cur = head
        while True:
            outs = children.get(cur, [])
            if len(outs) == 0:
                break
            if len(outs) > 1:
                raise ValueError(f"Branching detected at node {cur} in arm starting at {head}")
            chain.append(outs[0])
            cur = outs[0]
        return chain

    arm_chains = {h: follow_chain(h) for h in arm_heads}
    if arm_chains[correct_head][-1] != goal:
        raise ValueError("Canonical arm must terminate at goal")

    chosen_wrong_heads = random.sample(wrong_heads, num_wrong_arms_explored)

    matrix: Dict[int, List[Tuple[int, float]]] = {}
    matrix[source] = [(root, 1.0)]

    root_transitions: List[Tuple[int, float]] = [(correct_head, correct_arm_probability)]
    share = (1.0 - correct_arm_probability) / num_wrong_arms_explored
    for head in chosen_wrong_heads:
        root_transitions.append((head, share))

    total_root = sum(p for _, p in root_transitions)
    if abs(total_root - 1.0) >= 1e-9:
        raise AssertionError(f"Root probabilities sum to {total_root}, expected 1.0")
    matrix[root] = root_transitions

    for head, chain in arm_chains.items():
        for i in range(len(chain) - 1):
            matrix[chain[i]] = [(chain[i + 1], 1.0)]
        leaf = chain[-1]
        if leaf == goal:
            matrix[leaf] = []
        else:
            matrix[leaf] = [(source, 1.0)]

    return matrix


def _sample_pure_random_walk_no_subset(
    path: List[int],
    edge_list: List[Tuple[int, int]],
    margin: float,
    correct_arm_probability: Optional[float],
    max_walk_tokens: Optional[int],
    tokenizer,
) -> List[int]:
    """Stateful sampler for pure_random_walk when num_wrong_arms_explored is
    unset. All arm heads are visible; per-visit probabilities are derived from
    `margin` (visited arms get a fixed discount, leftover redistributed over
    unvisited arms) and `correct_arm_probability` (None => uniform base).

    Descent along each arm is deterministic; non-goal leaves return to source,
    and the goal is absorbing.
    """
    if len(path) < 3:
        raise ValueError("pure_random_walk requires path length >= 3")
    if margin < 0.0:
        raise ValueError(f"margin must be >= 0, got {margin}")

    source = path[0]
    root = path[1]
    goal = path[-1]
    correct_head = path[2]

    children: Dict[int, List[int]] = {}
    for u, v in edge_list:
        children.setdefault(u, []).append(v)

    arm_heads = children.get(root, [])
    if not arm_heads:
        raise ValueError("Root has no outgoing arms")

    k = len(arm_heads)
    if correct_arm_probability is not None:
        if not (0.0 < correct_arm_probability < 1.0):
            raise ValueError("correct_arm_probability must lie in (0, 1)")
        if k < 2:
            raise ValueError("correct_arm_probability requires more than one arm at the root")
        if correct_head not in arm_heads:
            raise ValueError("Canonical correct head not in arm_heads")

    # Validate that every arm is a simple non-branching chain.
    def follow_chain(head):
        chain = [head]
        cur = head
        while True:
            outs = children.get(cur, [])
            if len(outs) == 0:
                break
            if len(outs) > 1:
                raise ValueError(f"Branching detected at node {cur} in arm starting at {head}")
            chain.append(outs[0])
            cur = outs[0]
        return chain

    _ = {h: follow_chain(h) for h in arm_heads}

    def base_probs():
        if correct_arm_probability is None:
            return {h: 1.0 / k for h in arm_heads}
        p_correct = correct_arm_probability
        p_other = (1.0 - p_correct) / (k - 1)
        bp = {h: p_other for h in arm_heads}
        bp[correct_head] = p_correct
        return bp

    if max_walk_tokens is not None:
        max_walk_tokens -= 2

    def token_limit_reached(walk):
        if max_walk_tokens is None:
            return False
        if tokenizer is None:
            raise ValueError("Tokenizer is required when max_walk_tokens is specified")
        # Same O(1) shortcut as in _sample_walk_from_matrix: token count of
        # tokenize(make_suffix_string(walk)) is exactly len(walk) + 1.
        return len(walk) + 1 >= max_walk_tokens

    visits = {h: 0 for h in arm_heads}

    def compute_probs():
        base_p = base_probs()
        if margin == 0.0:
            return base_p
        adjusted = {}
        leftover = 0.0
        for h in arm_heads:
            raw = base_p[h] - (margin if visits[h] > 0 else 0.0)
            if raw < 0.0:
                leftover += base_p[h]
                adjusted[h] = 0.0
            else:
                leftover += base_p[h] - raw
                adjusted[h] = raw
        unvisited = [h for h in arm_heads if visits[h] == 0]
        if leftover > 1e-12:
            if not unvisited:
                raise ValueError("Leftover mass exists but no unvisited arms to redistribute to")
            add_each = leftover / len(unvisited)
            for h in unvisited:
                adjusted[h] += add_each
        total = sum(adjusted.values())
        if total <= 0.0:
            raise ValueError("Computed zero total probability over arms after margin adjustment")
        for h in adjusted:
            adjusted[h] = max(0.0, adjusted[h] / total)
        return adjusted

    walk: List[int] = []
    current = source
    while True:
        walk.append(current)
        if token_limit_reached(walk):
            break
        if current == goal:
            break

        if current == source:
            current = root
            continue
        if current == root:
            probs = compute_probs()
            heads = list(probs.keys())
            weights = [probs[h] for h in heads]
            if not any(w > 0 for w in weights):
                raise ValueError("No positive probability to select any arm at the root")
            chosen = random.choices(heads, weights=weights, k=1)[0]
            visits[chosen] += 1
            current = chosen
            continue

        outs = children.get(current, [])
        if len(outs) == 0:
            if current == goal:
                break
            current = source
            continue
        if len(outs) == 1:
            current = outs[0]
            continue
        raise ValueError(f"Branching detected mid-arm at node {current}")

    return walk


# ---------------------------------------------------------------------------
# Appendix F: looping-data traces
# ---------------------------------------------------------------------------

def _generate_source_root_loop(source: int, root: int, tokenizer, max_walk_tokens: Optional[int]) -> List[int]:
    """Replace the walk with source<->root oscillation filling the budget.
    Used when looping_data_frac fires under random_walk_with_reset.
    """
    if max_walk_tokens is None:
        raise ValueError("max_walk_tokens must be provided when looping_data_frac > 0")
    if tokenizer is None:
        raise ValueError("Tokenizer is required when max_walk_tokens is specified")

    budget = max_walk_tokens - 2
    walk: List[int] = []
    current = source
    while True:
        walk.append(current)
        # O(1) shortcut: tokenize(",".join(str(n) for n in walk))[1] == len(walk) + 1.
        if len(walk) + 1 >= budget:
            break
        current = root if current == source else source
    return walk


def _generate_wrong_arm_loop(path: List[int], edge_list: List[Tuple[int, int]], tokenizer, max_walk_tokens: Optional[int]) -> List[int]:
    """Replace the walk with a source->root->(random non-goal arm)->source cycle
    filling the budget. Used when looping_data_frac fires under
    pure_random_walk.
    """
    if max_walk_tokens is None:
        raise ValueError("max_walk_tokens must be provided when looping_data_frac > 0")
    if tokenizer is None:
        raise ValueError("Tokenizer is required when max_walk_tokens is specified")
    if len(path) < 3:
        raise ValueError("path must have at least source, root, and one arm node")

    source = path[0]
    root = path[1]
    correct_head = path[2]
    goal = path[-1]

    children: Dict[int, List[int]] = {}
    for u, v in edge_list:
        children.setdefault(u, []).append(v)

    arm_heads = children.get(root, [])
    if not arm_heads:
        raise ValueError("Root has no outgoing arms to loop over")

    wrong_heads = [h for h in arm_heads if h != correct_head]
    if not wrong_heads:
        raise ValueError("No wrong arms available for looping bias")

    chosen_head = random.choice(wrong_heads)

    chain = [chosen_head]
    current = chosen_head
    while True:
        outs = children.get(current, [])
        if len(outs) == 0:
            break
        if len(outs) != 1:
            raise ValueError(f"Branching detected on wrong arm starting at {chosen_head}")
        chain.append(outs[0])
        current = outs[0]

    if chain[-1] == goal:
        raise ValueError("Chosen wrong arm unexpectedly reaches the goal")

    budget = max_walk_tokens - 2
    walk: List[int] = []
    cycle = [source, root] + chain
    while True:
        for node in cycle:
            walk.append(node)
            # O(1) shortcut: tokenize(",".join(str(n) for n in walk))[1] == len(walk) + 1.
            if len(walk) + 1 >= budget:
                return walk


# ---------------------------------------------------------------------------
# Top-level dispatch (called from prepare.py)
# ---------------------------------------------------------------------------

def _compute_max_walk_tokens(edge_list, source, goal, max_tokens, tokenizer) -> Optional[int]:
    if max_tokens is None:
        return None
    if tokenizer is None:
        raise ValueError("Tokenizer is required when max_tokens is specified")
    _, prefix_tokens = tokenizer.tokenize(make_prefix_string(edge_list, source, goal))
    return max_tokens - prefix_tokens


def _validate_looping_data_frac(max_tokens: Optional[int], looping_data_frac: float) -> None:
    if not (0.0 <= looping_data_frac <= 1.0):
        raise ValueError(f"looping_data_frac must lie in [0, 1], got {looping_data_frac}")
    if looping_data_frac > 0.0 and max_tokens is None:
        raise ValueError("looping_data_frac > 0 requires max_tokens to be specified")


def sample_random_walk_with_reset(
    num_paths: int,
    path_length: int,
    max_nodes: int,
    max_tokens: Optional[int],
    tokenizer,
    reset_probability: float,
    exploration_probability: float,
    looping_data_frac: float,
):
    """One sample of random_walk_with_reset. Returns (walk, edge_list, source, goal)."""
    _validate_looping_data_frac(max_tokens, looping_data_frac)
    path, edge_list, source, goal = build_star_graph(num_paths, path_length, max_nodes)
    root = path[1]
    max_walk_tokens = _compute_max_walk_tokens(edge_list, source, goal, max_tokens, tokenizer)

    if looping_data_frac > 0.0 and random.random() < looping_data_frac:
        walk = _generate_source_root_loop(source, root, tokenizer, max_walk_tokens)
        return walk, edge_list, source, goal

    matrix = _build_random_walk_with_reset_matrix(
        path=path,
        edge_list=edge_list,
        reset_probability=reset_probability,
        exploration_probability=exploration_probability,
    )
    walk = _sample_walk_from_matrix(source, matrix, max_walk_tokens, tokenizer)
    return walk, edge_list, source, goal


def sample_pure_random_walk(
    num_paths: int,
    path_length: int,
    max_nodes: int,
    max_tokens: Optional[int],
    tokenizer,
    margin: float,
    correct_arm_probability: Optional[float],
    num_wrong_arms_explored: Optional[int],
    looping_data_frac: float,
):
    """One sample of pure_random_walk. Returns (walk, edge_list, source, goal)."""
    _validate_looping_data_frac(max_tokens, looping_data_frac)
    path, edge_list, source, goal = build_star_graph(num_paths, path_length, max_nodes)
    max_walk_tokens = _compute_max_walk_tokens(edge_list, source, goal, max_tokens, tokenizer)

    if looping_data_frac > 0.0 and random.random() < looping_data_frac:
        walk = _generate_wrong_arm_loop(path, edge_list, tokenizer, max_walk_tokens)
        return walk, edge_list, source, goal

    if num_wrong_arms_explored is not None:
        if correct_arm_probability is None:
            raise ValueError("num_wrong_arms_explored requires correct_arm_probability to be set")
        if margin != 0.0:
            raise ValueError("margin must be 0 when num_wrong_arms_explored is set (Appendix E G(10,5))")
        matrix = _build_pure_random_walk_subset_matrix(
            path=path,
            edge_list=edge_list,
            correct_arm_probability=correct_arm_probability,
            num_wrong_arms_explored=num_wrong_arms_explored,
        )
        walk = _sample_walk_from_matrix(source, matrix, max_walk_tokens, tokenizer)
        return walk, edge_list, source, goal

    walk = _sample_pure_random_walk_no_subset(
        path=path,
        edge_list=edge_list,
        margin=margin,
        correct_arm_probability=correct_arm_probability,
        max_walk_tokens=max_walk_tokens,
        tokenizer=tokenizer,
    )
    return walk, edge_list, source, goal

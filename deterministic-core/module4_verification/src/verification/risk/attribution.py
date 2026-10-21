"""Shapley risk attribution: which feeder's uncertainty drives the risk (Step 8 of the method).

Check B says "k of N likely next-hour conditions break a limit". The next question is whose
uncertainty that risk comes from. The players are groups of feeder buses; for any coalition S,

    v(S) = share of the scenarios that fail when only the feeders in S move as drawn,
           every other feeder held at this hour's snapshot.

The Shapley value (Shapley 1953) splits v(all) fairly: each player's average marginal
contribution over every order the players could join in. As "Shapley effects" this splits the
uncertainty in a model's output among its uncertain inputs (Owen 2014; Song, Nelson & Staum
2016) -- here the output is the failure indicator, so the shares add up exactly to the risk
(efficiency).

Players (the plan's decision, recorded in docs/m4/rcv-method.md): Nainativu's two feeder groups
are separate players, and each other island is one player -- five players, 32 coalitions. The
islands are separate grids, so a coalition's outcome is assembled island by island from a few
solves per scenario instead of 32: an island with one player needs only the "all move" solve
Check B already ran; Nainativu's two players need one extra solve each (F1 alone, F2 alone).
Every coalition reuses the same seeded scenarios (common random numbers), so the shares do not
wobble between runs.

Note: the load history is island-level and the model splits it 50/50 between F1 and F2, so the
two Nainativu players always move by the same multiplier. Their split is then about how much
each feeder's share of that move contributes to a failure, not about different forecasts.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

# Islands whose feeder groups are attributed separately; every other island is one player.
SPLIT_ISLANDS = ("Nainativu",)


@dataclass(frozen=True)
class Player:
    """One attribution player: a name an operator reads, and the feeder buses it moves."""

    name: str
    island: str
    buses: frozenset[str]


def players_from_pin(pin: dict, split_islands: tuple[str, ...] = SPLIT_ISLANDS) -> list[Player]:
    """The players, built from the topology pin's feeder-group buses (never hard-coded)."""
    feeders: dict[str, list[str]] = {}
    for b in pin["buses"]:
        if b.get("role") == "feeder_group":
            feeders.setdefault(b["island"], []).append(b["node_id"].upper())
    players: list[Player] = []
    for island in sorted(feeders):
        buses = sorted(feeders[island])
        if island in split_islands:
            players += [Player(bus, island, frozenset({bus})) for bus in buses]
        else:
            players.append(Player(island, island, frozenset(buses)))
    return players


def shapley_values(n_players: int, value: dict[frozenset[int], float]) -> list[float]:
    """Exact Shapley values from the value of every coalition (frozensets of player indices)."""
    phi = [0.0] * n_players
    fact = math.factorial
    for i in range(n_players):
        others = [j for j in range(n_players) if j != i]
        for r in range(len(others) + 1):
            weight = fact(r) * fact(n_players - r - 1) / fact(n_players)
            for coalition in itertools.combinations(others, r):
                s = frozenset(coalition)
                phi[i] += weight * (value[s | {i}] - value[s])
    return phi


def coalition_values(players: list[Player], island_outcomes: list[dict[str, dict]],
                     ) -> dict[frozenset[int], float]:
    """v(S) for every coalition from per-scenario, per-island outcomes.

    ``island_outcomes[s][island][subset]`` is True when that island fails in scenario s with
    only ``subset`` (a frozenset of that island's player indices) moving. The empty subset is
    the snapshot itself (Check A passed, so it never fails).
    """
    n = len(players)
    by_island: dict[str, frozenset[int]] = {}
    for idx, p in enumerate(players):
        by_island[p.island] = by_island.get(p.island, frozenset()) | {idx}
    scenarios = len(island_outcomes)
    values: dict[frozenset[int], float] = {}
    for r in range(n + 1):
        for coalition in itertools.combinations(range(n), r):
            s = frozenset(coalition)
            fails = 0
            for outcome in island_outcomes:
                if any(outcome.get(isl, {}).get(s & members, False)
                       for isl, members in by_island.items()):
                    fails += 1
            values[s] = fails / scenarios if scenarios else 0.0
    return values


def island_subsets(players: list[Player]) -> list[dict[str, frozenset[int]]]:
    """The extra solve configurations needed beyond "all move".

    Configuration j moves, on every island at once, that island's j-th proper non-empty subset
    of players (islands are independent grids, so one solve serves them all).
    """
    by_island: dict[str, list[int]] = {}
    for idx, p in enumerate(players):
        by_island.setdefault(p.island, []).append(idx)
    proper = {
        isl: [frozenset(c) for r in range(1, len(m)) for c in itertools.combinations(m, r)]
        for isl, m in by_island.items()
    }
    depth = max((len(v) for v in proper.values()), default=0)
    return [{isl: subs[j] for isl, subs in proper.items() if j < len(subs)}
            for j in range(depth)]

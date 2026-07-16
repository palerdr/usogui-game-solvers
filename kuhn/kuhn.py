"""A tiny, readable CFR+ solver for two-player Kuhn poker."""

from itertools import permutations


CARDS = "JQK"  # J < Q < K
ACTIONS = {
    "": ("x", "b"),      # player 0: check, bet
    "x": ("x", "b"),     # player 1: check, bet
    "b": ("c", "f"),      # player 1: call, fold
    "xb": ("c", "f"),     # player 0: call, fold
}


class Node:
    """The regrets and average-strategy mass for one information set."""

    def __init__(self, actions):
        self.actions = actions
        self.regret = [0.0] * len(actions)
        self.strategy_sum = [0.0] * len(actions)

    def strategy(self):
        """CFR+ regret matching: negative cumulative regret is discarded."""
        positive = [max(r, 0.0) for r in self.regret]
        total = sum(positive)
        return ([r / total for r in positive]
                if total else [1.0 / len(self.actions)] * len(self.actions))

    def average_strategy(self):
        total = sum(self.strategy_sum)
        return ([s / total for s in self.strategy_sum]
                if total else [1.0 / len(self.actions)] * len(self.actions))


def terminal(history):
    return history in ("xx", "bc", "bf", "xbc", "xbf")


def payoff(cards, history):
    """Return player 0's payoff, excluding the antes."""
    if history == "bf":
        return 1                         # player 1 folded
    if history == "xbf":
        return -1                        # player 0 folded

    winner = 1 if cards[0] > cards[1] else -1
    return winner * (1 if history == "xx" else 2)


def cfr(cards, history, reach0, reach1, nodes, updating_player, average_weight):
    """One full-tree CFR traversal; utilities are always from player 0's view."""
    if terminal(history):
        return payoff(cards, history)

    player = 0 if history in ("", "xb") else 1
    actions = ACTIONS[history]
    info = cards[player] + history  # private card + public action history
    node = nodes.setdefault(info, Node(actions))
    strategy = node.strategy()

    # CFR+ alternates players. Average only the strategy being updated here.
    if player == updating_player:
        reach = reach0 if player == 0 else reach1
        for i, probability in enumerate(strategy):
            node.strategy_sum[i] += average_weight * reach * probability

    utilities = []
    for i, action in enumerate(actions):
        if player == 0:
            utilities.append(cfr(cards, history + action, reach0 * strategy[i],
                                 reach1, nodes, updating_player, average_weight))
        else:
            utilities.append(cfr(cards, history + action, reach0,
                                 reach1 * strategy[i], nodes, updating_player,
                                 average_weight))

    node_utility = sum(p * u for p, u in zip(strategy, utilities))

    # CFR+ regret matching+: cumulative regrets are clipped at zero.
    if player == updating_player:
        for i, utility in enumerate(utilities):
            regret = (utility - node_utility
                      if player == 0 else node_utility - utility)
            opponent_reach = reach1 if player == 0 else reach0
            node.regret[i] = max(0.0, node.regret[i] + opponent_reach * regret)

    return node_utility


def train(iterations=100_000):
    nodes = {}
    deals = list(permutations(CARDS, 2))
    chance = 1.0 / len(deals)

    for iteration in range(1, iterations + 1):
        for updating_player in (0, 1):
            for cards in deals:
                # Chance reaches every deal with probability 1/6.
                cfr(cards, "", chance, chance, nodes, updating_player, iteration)
    return nodes


def evaluate(cards, history, nodes):
    """Expected player 0 payoff when both players use average strategies."""
    if terminal(history):
        return payoff(cards, history)
    player = 0 if history in ("", "xb") else 1
    node = nodes[cards[player] + history]
    return sum(p * evaluate(cards, history + a, nodes)
               for p, a in zip(node.average_strategy(), node.actions))


def print_solution(nodes):
    for info in sorted(nodes):
        node = nodes[info]
        strategy = node.average_strategy()
        actions = ", ".join(f"{a}:{p:.3f}" for a, p in zip(node.actions, strategy))
        print(f"{info:>3}: {actions}")


if __name__ == "__main__":
    nodes = train()
    print_solution(nodes)
    value = sum(evaluate(cards, "", nodes) for cards in permutations(CARDS, 2)) / 6
    print(f"\nAverage payoff for player 0: {value:.4f} (Nash value: -0.0556)")

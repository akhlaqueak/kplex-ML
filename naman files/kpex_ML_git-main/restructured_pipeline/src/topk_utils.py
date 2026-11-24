"""Shared helpers for probability-ranked top-k predictions."""

from __future__ import annotations


def rank_probabilities(row, algorithms, probability_prefix="prob_"):
    probs = {alg: row[f"{probability_prefix}{alg}"] for alg in algorithms}
    return sorted(probs.items(), key=lambda item: item[1], reverse=True)


def get_top3(row, algorithms, probability_prefix="prob_"):
    return rank_probabilities(row, algorithms, probability_prefix=probability_prefix)[:3]

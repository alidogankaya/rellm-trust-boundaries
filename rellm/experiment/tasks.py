"""Tasks for the reverse + causal-pruning experiment.

Each task has three disjoint splits:
  train    - used to find the circuit and to calibrate every pruning method (same data for all arms)
  heldout  - same templates, unseen entities (countries / names / adjectives): the main score
  ood      - unseen templates on heldout entities: does the pruned model still *do the task*, or only the prompts?
Candidates are scored by first token when their first tokens are distinct, else by full sequence log-prob.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Example:
    prompt: str
    answer: str
    candidates: list[str]
    template: str = ""
    key: str = ""


@dataclass
class Task:
    name: str
    description: str
    train: list[Example]
    heldout: list[Example]
    ood: list[Example]
    general_texts: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "n_train": len(self.train), "n_heldout": len(self.heldout),
                "n_ood": len(self.ood), "n_general_texts": len(self.general_texts),
                "train_sample": asdict(self.train[0]) if self.train else None, "ood_sample": asdict(self.ood[0]) if self.ood else None}


def _cap(rng: random.Random, xs: list[Example], n: int) -> list[Example]:
    xs = list(xs)
    rng.shuffle(xs)
    return xs[:n]


GENERAL_EN = [
    "The river flows slowly through the valley, past old stone bridges and quiet farms where cattle graze in the morning light.",
    "To make bread, mix flour, water, salt and yeast, knead the dough for ten minutes, then let it rise until it doubles in size.",
    "The committee met on Tuesday to review the budget, and after a long discussion it approved most of the proposed changes.",
    "Electric cars store energy in large batteries, which are charged from the grid and power a motor that turns the wheels.",
    "She opened the window, listened to the rain on the roof, and decided that the letter could wait until tomorrow.",
    "In mathematics, a prime number has exactly two divisors: one and itself. The first few primes are two, three, five and seven.",
    "The museum's new exhibition traces the history of printing, from carved wooden blocks to modern digital presses.",
    "He packed a map, a torch and enough water for two days before setting out along the coastal path toward the lighthouse.",
]

# --------------------------------------------------------------------------------------------- capital
CAPITALS = [("France", "Paris"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
            ("China", "Beijing"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Canada", "Ottawa"),
            ("Norway", "Oslo"), ("Greece", "Athens"), ("Turkey", "Ankara"), ("Russia", "Moscow"), ("Poland", "Warsaw"),
            ("Austria", "Vienna"), ("Hungary", "Budapest"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Sweden", "Stockholm"),
            ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Belgium", "Brussels"), ("Cuba", "Havana"), ("Chile", "Santiago"),
            ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
            ("Syria", "Damascus"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"), ("Australia", "Canberra"),
            ("Colombia", "Bogota"), ("Venezuela", "Caracas"), ("Nigeria", "Abuja"), ("Morocco", "Rabat"), ("Pakistan", "Islamabad"),
            ("Afghanistan", "Kabul"), ("Philippines", "Manila"), ("Argentina", "Buenos Aires"), ("Ethiopia", "Addis Ababa")]


def capital_task(seed: int = 0, n_train: int = 48, n_heldout: int = 48, n_ood: int = 32) -> Task:
    rng = random.Random(seed)
    pairs = list(CAPITALS)
    rng.shuffle(pairs)
    cut = int(len(pairs) * 0.6)
    tr, ho = pairs[:cut], pairs[cut:]
    cands = [" " + cap for _, cap in CAPITALS]
    tmpl = ["The capital of {c} is", "The capital city of {c} is", "{c}'s capital is the city of"]
    ood_t = ["Q: What is the capital of {c}?\nA:", "Everyone knows that the government of {c} sits in"]
    mk = lambda ps, ts: [Example(t.format(c=c), " " + cap, cands, t, c) for c, cap in ps for t in ts]
    return Task("capital", "country -> capital city (factual recall)", _cap(rng, mk(tr, tmpl), n_train), _cap(rng, mk(ho, tmpl), n_heldout),
                _cap(rng, mk(ho, ood_t), n_ood), GENERAL_EN)


# ------------------------------------------------------------------------------------------------- IOI
NAMES = ["Mary", "John", "Tom", "James", "Sarah", "Kate", "Paul", "Mike", "Anna", "Lisa", "Mark", "David", "Laura", "Peter",
         "Emma", "Chris", "Alice", "Bob", "Daniel", "Jessica"]


def ioi_task(seed: int = 0, n_train: int = 48, n_heldout: int = 48, n_ood: int = 32) -> Task:
    rng = random.Random(seed)
    names = list(NAMES)
    rng.shuffle(names)
    tr, ho = names[:12], names[12:]
    tmpl = ["When {A} and {B} went to the store, {B} gave a drink to",
            "After {A} and {B} went to the park, {B} handed a ball to",
            "While {A} and {B} were working at the office, {B} sent an email to",
            "When {B} and {A} went to the store, {B} gave a drink to"]
    ood_t = ["{A} and {B} found a wallet near the river. Later, {B} returned the wallet to",
             "Yesterday {A} met {B} at the cafe, and {B} bought a coffee for"]

    def mk(pool, ts):
        out = []
        for a in pool:
            for b in pool:
                if a != b:
                    for t in ts:
                        out.append(Example(t.format(A=a, B=b), " " + a, [" " + a, " " + b], t, f"{a}|{b}"))
        return out
    return Task("ioi", "indirect object identification (name moving)", _cap(rng, mk(tr, tmpl), n_train), _cap(rng, mk(ho, tmpl), n_heldout),
                _cap(rng, mk(ho, ood_t), n_ood), GENERAL_EN)


# ------------------------------------------------------------------------------------------- sentiment
POS = ["wonderful", "brilliant", "delightful", "excellent", "fantastic", "charming", "superb", "lovely", "outstanding", "enjoyable",
       "beautiful", "great"]
NEG = ["terrible", "boring", "awful", "dreadful", "dull", "horrible", "disappointing", "painful", "weak", "tedious", "poor", "bad"]
FEWSHOT = "Review: An absolute delight, I loved every minute.\nSentiment: positive\nReview: Far too long and nothing happens.\nSentiment: negative\n"


def sentiment_task(seed: int = 0, n_train: int = 48, n_heldout: int = 48, n_ood: int = 32) -> Task:
    rng = random.Random(seed)
    pos, neg = list(POS), list(NEG)
    rng.shuffle(pos)
    rng.shuffle(neg)
    cands = [" positive", " negative"]
    nouns = ["movie", "film", "story", "acting", "plot", "ending", "soundtrack", "script"]
    products = ["blender", "phone", "jacket", "lamp", "backpack", "kettle"]

    def mk(pa, na, subjects, form):
        out = []
        for adj, lab in [(a, " positive") for a in pa] + [(a, " negative") for a in na]:
            for n in subjects:
                out.append(Example(FEWSHOT + "Review: " + form.format(n=n, a=adj) + "\nSentiment:", lab, cands, form, adj))
        return out
    return Task("sentiment", "few-shot sentiment classification",
                _cap(rng, mk(pos[:8], neg[:8], nouns, "The {n} was {a}."), n_train),
                _cap(rng, mk(pos[8:], neg[8:], nouns, "The {n} was {a}."), n_heldout),
                _cap(rng, mk(pos[8:], neg[8:], products, "This {n} is {a}, honestly."), n_ood), GENERAL_EN)


# ----------------------------------------------------------------------------------- synthetic (demo)
SYN_COUNTRIES = ["france", "italy", "spain", "germany", "japan", "china", "brazil", "egypt", "kenya", "peru", "india", "canada",
                 "norway", "greece", "turkey", "mexico"]
SYN_CITIES = ["paris", "rome", "madrid", "berlin", "tokyo", "beijing", "brasilia", "cairo", "nairobi", "lima", "delhi", "ottawa",
              "oslo", "athens", "ankara", "mexicocity"]
SYN_FILLER = [f"w{i}" for i in range(32)]


def syn_capital_task(seed: int = 0, n_train: int = 48, n_heldout: int = 48, n_ood: int = 32) -> Task:
    """For the synthetic demo model (scripts/demo_synthetic.py): the model knows all 16 facts, so splits are by country."""
    rng = random.Random(seed)
    idx = list(range(16))
    rng.shuffle(idx)
    tr, ho = idx[:10], idx[10:]

    def mk(ids, n_fill):
        out = []
        for i in ids:
            for k in n_fill:
                for rep in range(3):
                    pre = rng.sample(SYN_FILLER, k)
                    out.append(Example(" ".join(pre + ["the", "capital", "of", SYN_COUNTRIES[i], "is"]), SYN_CITIES[i], list(SYN_CITIES),
                                       f"fill{k}", SYN_COUNTRIES[i]))
        return out
    general = []
    for _ in range(6):
        half = [rng.choice(SYN_FILLER) for _ in range(19)]
        general.append(" ".join(half + half))
    return Task("syn_capital", "synthetic country -> city lookup", _cap(rng, mk(tr, [0, 1, 2, 3]), n_train),
                _cap(rng, mk(ho, [0, 1, 2, 3]), n_heldout), _cap(rng, mk(ho, [6, 7]), n_ood), general)


def load_custom_task(path: str) -> Task:
    """JSON: {"name", "description", "train": [{"prompt","answer","candidates"}], "heldout": [...], "ood": [...], "general_texts": [...]}"""
    d = json.load(open(path, encoding="utf-8"))
    ex = lambda rows: [Example(r["prompt"], r["answer"], r["candidates"], r.get("template", ""), r.get("key", "")) for r in rows]
    return Task(d["name"], d.get("description", ""), ex(d["train"]), ex(d["heldout"]), ex(d.get("ood", [])), d.get("general_texts", GENERAL_EN))


TASKS = {"capital": capital_task, "ioi": ioi_task, "sentiment": sentiment_task, "syn_capital": syn_capital_task}


def get_task(name: str, seed: int = 0) -> Task:
    if name.endswith(".json"):
        return load_custom_task(name)
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; built-in: {sorted(TASKS)} or a path to a .json task")
    return TASKS[name](seed=seed)

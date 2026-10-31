#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuro-Evolution CLI - Multi-Dataset Neural Architecture Search
================================================================
Command-line version of the two-phase neuro-evolution experiment.

Phase 1: Search dense architectures via genetic algorithm.
Phase 2: Evolve meta-optimizer coefficients on the best architecture.

Datasets supported:
  1. MNIST-1D
  2. CIFAR-10 (Small)
  3. CIFAR-10 (Medium)
  4. CIFAR-10 (Large)

Usage:
    python main.py
    python main.py --dataset mnist1d --seeds 1 --gen1 5 --gen2 5
    python main.py --dataset cifar_small --seeds 3 --gen1 8 --gen2 8
"""

import argparse
import copy
import json
import math
import os
import pickle
import random
import signal
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

# Ensure UTF-8 output on Windows consoles (Persian messages)
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================
# ANSI color codes (auto-disabled for non-TTY output)
# ============================================================
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


if not sys.stdout.isatty():
    for _a in [a for a in dir(C) if not a.startswith('_')]:
        setattr(C, _a, '')


# ============================================================
# Hardware settings and paths
# ============================================================
torch.backends.mkldnn.enabled = True
_N_CPU = os.cpu_count() or 4
# Reduce worker count to prevent CPU contention
MAX_WORKERS = max(1, _N_CPU // 2)
DETERMINISTIC_MODE = False

if DETERMINISTIC_MODE:
    MAX_WORKERS = 1

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DATA_MNIST_DIR = os.path.join(SCRIPT_DIR, 'data_mnist1d')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
CONFIGS_PATH = os.path.join(SCRIPT_DIR, 'configs.json')

for _d in [DATA_DIR, DATA_MNIST_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

USE_CHANNELS_LAST = True
TRANSFER_ERRORS = 0

# MNIST-1D pickle is fetched from this URL on first run if missing.
MNIST1D_URL = "https://raw.githubusercontent.com/greydanus/mnist1d/master/mnist1d_data.pkl"


# ============================================================
# Default configurations per dataset
# ============================================================
DEFAULT_CONFIGS = {
    "mnist1d": {
        "label": "MNIST-1D",
        "pop1": 8, "batches1": 300, "pop2": 8, "batches2": 500,
        "batch": 64, "n_train": 4000, "n_val": 1000, "n_test": 1000,
        "act_funcs": ["tanh"], "size_pool": [128, 256, 512],
        "min_layers": 1, "max_layers": 6,
    },
    "cifar_small": {
        "label": "CIFAR-10 (Small)",
        "pop1": 12, "batches1": 150, "pop2": 8, "batches2": 150,
        "batch": 128, "n_train": 2000, "n_val": 500, "n_test": 500,
        "act_funcs": ["tanh"], "size_pool": [64, 128],
        "min_layers": 1, "max_layers": 5,
    },
    "cifar_medium": {
        "label": "CIFAR-10 (Medium)",
        "pop1": 8, "batches1": 200, "pop2": 8, "batches2": 400,
        "batch": 128, "n_train": 10000, "n_val": 2000, "n_test": 2000,
        "act_funcs": ["tanh"], "size_pool": [128, 256],
        "min_layers": 1, "max_layers": 6,
    },
    "cifar_large": {
        "label": "CIFAR-10 (Large)",
        "pop1": 8, "batches1": 400, "pop2": 8, "batches2": 800,
        "batch": 256, "n_train": 20000, "n_val": 5000, "n_test": 5000,
        "act_funcs": ["tanh"], "size_pool": [256, 512],
        "min_layers": 1, "max_layers": 7,
    },
}

# Order in which datasets appear in the interactive menu.
DATASET_KEYS = ["mnist1d", "cifar_small", "cifar_medium", "cifar_large"]


def load_configs():
    """Load configs.json if present, otherwise write defaults and return them."""
    if os.path.exists(CONFIGS_PATH):
        try:
            with open(CONFIGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in DEFAULT_CONFIGS.items():
                if k not in data:
                    data[k] = dict(v)
            return data
        except Exception as e:
            print(f"{C.YELLOW}⚠️ Cannot read configs.json ({e}) — using defaults.{C.RESET}")
    try:
        with open(CONFIGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIGS, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return copy.deepcopy(DEFAULT_CONFIGS)


# ============================================================
# Math helpers
# ============================================================
def softplus(x):
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


def _decode_coeffs(c):
    a = softplus(c['a'])
    b = 0.5 + 0.5 * math.tanh(c['b'])
    d = 0.1 * _sigmoid(c['d'])
    lr = softplus(c['lr'])
    return a, b, d, lr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# Optimizers
# ============================================================
class DeltaRuleOptimizer:
    """Simple SGD placeholder optimizer (kept for API symmetry; real train falls back to Adam)."""

    def __init__(self, params, lr=0.005):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data.add_(p.grad, alpha=-self.lr)


class MetaOptimizer:
    """Adam-family optimizer with evolvable coefficients (a, b, d, lr)."""

    def __init__(self, params, coeffs):
        self.params = list(params)
        self.coeffs = coeffs
        self.lr = softplus(coeffs['lr'])
        self.initial_lr = self.lr
        self.mom = [torch.zeros_like(p.data) for p in self.params]
        self.var = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        a, b, d, lr = _decode_coeffs(self.coeffs)
        eps = 1e-8
        self.t += 1
        bc1 = 1.0 - 0.9 ** self.t
        bc2 = 1.0 - 0.999 ** self.t
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                m = self.mom[i]
                v = self.var[i]
                m.mul_(0.9).add_(g, alpha=0.1)
                v.mul_(0.999).addcmul_(g, g, value=0.001)
                m_hat = m / bc1
                v_hat = v / bc2
                p.data.mul_(1.0 - lr * d)
                denom = (v_hat + eps).pow(b)
                p.data.addcdiv_(m_hat, denom, value=-lr * a)


def random_meta_coeffs():
    return {
        'a': random.uniform(-2.0, 2.0),
        'b': random.uniform(-2.0, 2.0),
        'd': random.uniform(-2.0, 2.0),
        'lr': random.uniform(-10.0, -4.0),
    }


def mutate_meta_coeffs(c):
    c = c.copy()
    if random.random() < 0.2:
        return random_meta_coeffs()
    for k in ['a', 'b', 'd', 'lr']:
        if random.random() < 0.3:
            c[k] += random.uniform(-0.5, 0.5)
    return c


def crossover_meta_coeffs(c1, c2):
    c = c1.copy()
    for k in ['a', 'b', 'd', 'lr']:
        if random.random() < 0.5:
            c[k] = c2[k]
    return c


def coeffs_to_str(c):
    a, b, d, lr = _decode_coeffs(c)
    return f"a={a:.3f}, b={b:.3f}, d={d:.5f}, lr={lr:.5f}"


# ============================================================
# Dynamic network
# ============================================================
class DynamicNet(nn.Module):
    """CNN-stem (CIFAR) or Conv1D-stem (MNIST-1D) + dynamic dense head with dense connectivity."""

    def __init__(self, genome, in_dim, out_dim, use_cnn_stem):
        super().__init__()
        self.genome = genome
        self.use_cnn_stem = use_cnn_stem

        if use_cnn_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            )
            flat_dim = 128 * 4 * 4
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            )
            flat_dim = 32 * (in_dim // 4)

        self.transition = nn.Sequential(nn.Linear(flat_dim, 128), nn.BatchNorm1d(128), nn.ReLU())
        base_dim = 128
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        current_dim = base_dim
        for sz, act in zip(genome['layers'], genome['acts']):
            self.layers.append(nn.Linear(current_dim, sz))
            self.bns.append(nn.BatchNorm1d(sz) if sz > 1 else nn.Identity())
            self.acts.append(self._act(act))
            self.dropouts.append(nn.Dropout(0.3))
            current_dim += sz
        self.out_layer = nn.Linear(current_dim, out_dim)

    @staticmethod
    def _act(name):
        return {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
        }[name]

    def forward(self, x):
        if not self.use_cnn_stem:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = torch.flatten(x, 1)
        x = self.transition(x)
        features = [x]
        for i in range(len(self.layers)):
            x_in = torch.cat(features, dim=1)
            out = self.layers[i](x_in)
            out = self.bns[i](out)
            out = self.acts[i](out)
            out = self.dropouts[i](out)
            features.append(out)
        final_in = torch.cat(features, dim=1)
        return self.out_layer(final_in)


# ============================================================
# Genome helpers
# ============================================================
def random_genome(cfg):
    n = random.randint(cfg.get('min_layers', 1), cfg.get('max_layers', 6))
    return {
        'layers': [random.choice(cfg['size_pool']) for _ in range(n)],
        'acts': [random.choice(cfg['act_funcs']) for _ in range(n)],
    }


def mutate_genome(g, cfg):
    g = copy.deepcopy(g)
    if random.random() < 0.2:
        return random_genome(cfg)
    if random.random() < 0.4 and len(g['layers']) > 0:
        idx = random.randrange(len(g['layers']))
        g['layers'][idx] = random.choice(cfg['size_pool'])
    if random.random() < 0.2 and len(g['layers']) < cfg.get('max_layers', 6):
        i = random.randint(0, len(g['layers']))
        g['layers'].insert(i, random.choice(cfg['size_pool']))
        g['acts'].insert(i, random.choice(cfg['act_funcs']))
    if random.random() < 0.2 and len(g['layers']) > cfg.get('min_layers', 1):
        i = random.randrange(len(g['layers']))
        g['layers'].pop(i)
        g['acts'].pop(i)
    return g


def crossover_genome(g1, g2):
    max_len = max(len(g1['layers']), len(g2['layers']))
    new_layers, new_acts = [], []
    for i in range(max_len):
        a_ok = i < len(g1['layers'])
        b_ok = i < len(g2['layers'])
        if a_ok and b_ok:
            if random.random() < 0.5:
                new_layers.append(g1['layers'][i])
                new_acts.append(g1['acts'][i])
            else:
                new_layers.append(g2['layers'][i])
                new_acts.append(g2['acts'][i])
        elif a_ok and random.random() < 0.5:
            new_layers.append(g1['layers'][i])
            new_acts.append(g1['acts'][i])
        elif b_ok and random.random() < 0.5:
            new_layers.append(g2['layers'][i])
            new_acts.append(g2['acts'][i])
    if not new_layers:
        new_layers = [random.choice([g1['layers'][0], g2['layers'][0]])]
        new_acts = [random.choice([g1['acts'][0], g2['acts'][0]])]
    return {'layers': new_layers, 'acts': new_acts}


def genome_to_str(g):
    return f"L={g['layers']} A={g['acts']}"


# ============================================================
# Weight transfer
# ============================================================
def transfer_weights(new_model, old_model, alpha=0.7):
    global TRANSFER_ERRORS
    with torch.no_grad():
        if new_model.use_cnn_stem and old_model.use_cnn_stem:
            try:
                for pn, po in zip(new_model.stem.parameters(), old_model.stem.parameters()):
                    if pn.shape == po.shape:
                        pn.mul_(1 - alpha).add_(po, alpha=alpha)
                for bn, bo in zip(new_model.stem.buffers(), old_model.stem.buffers()):
                    if bn.shape == bo.shape:
                        bn.copy_(bo)
            except Exception:
                TRANSFER_ERRORS += 1
        for pn, po in zip(new_model.transition.parameters(), old_model.transition.parameters()):
            if pn.shape == po.shape:
                pn.mul_(1 - alpha).add_(po, alpha=alpha)
        for nl, ol in zip(new_model.layers, old_model.layers):
            if isinstance(nl, nn.Linear) and isinstance(ol, nn.Linear):
                if nl.weight.shape == ol.weight.shape:
                    nl.weight.data.mul_(1 - alpha).add_(ol.weight.data, alpha=alpha)
                    if nl.bias is not None and ol.bias is not None:
                        nl.bias.data.mul_(1 - alpha).add_(ol.bias.data, alpha=alpha)


# ============================================================
# Training and evaluation
# ============================================================
def _prep_tensor(x, is_cnn):
    if is_cnn and USE_CHANNELS_LAST and x.ndim == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x


def _make_train_loader(dataset, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)


def _make_eval_loader(dataset, batch_size=512):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_model(model, train_loader, val_loader, opt_cls, opt_kwargs, max_batches,
                crit, stop_event, quick_prune=False, out_dim=10):
    if USE_CHANNELS_LAST and model.use_cnn_stem:
        model = model.to(memory_format=torch.channels_last)
    if opt_cls == DeltaRuleOptimizer:
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        opt = opt_cls(model.parameters(), **opt_kwargs)

    model.train()
    n = 0
    check_at = max(1, int(max_batches * 0.25)) if quick_prune else max_batches

    while n < max_batches:
        for x, y in train_loader:
            if stop_event.is_set():
                model.eval()
                return True

            if USE_CHANNELS_LAST and model.use_cnn_stem:
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += 1

            if hasattr(opt, 'initial_lr'):
                progress = n / max_batches
                opt.lr = opt.initial_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

            if quick_prune and n == check_at:
                acc = evaluate(model, val_loader, use_tta=False)
                model.train()
                if acc < (10.0 if out_dim > 2 else 55.0):
                    model.eval()
                    return False
    model.eval()
    return True


@torch.inference_mode()
def evaluate(model, loader, use_tta=False):
    model.eval()
    is_cnn = getattr(model, 'use_cnn_stem', False)
    correct = total = 0
    for x, y in loader:
        if use_tta and x.ndim == 4:
            x = x.contiguous()
            x4 = torch.cat([x, x.flip(3), x.flip(2), x.flip((2, 3))], dim=0)
            x4 = _prep_tensor(x4, is_cnn)
            out = model(x4).softmax(1)
            n = x.size(0)
            out = (out[:n] + out[n:2 * n] + out[2 * n:3 * n] + out[3 * n:]).div_(4)
        else:
            x = _prep_tensor(x, is_cnn)
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)


# ============================================================
# Phase 1 and Phase 2 workers
# ============================================================
def eval_phase1_worker(genome, init_state, in_dim, out_dim, cfg, train_ds, val_ds, worker_seed, stop_event):
    try:
        if DETERMINISTIC_MODE:
            set_seed(worker_seed)
        tl = _make_train_loader(train_ds, cfg['batch'], worker_seed)
        vl = _make_eval_loader(val_ds)
        model = DynamicNet(genome, in_dim, out_dim, cfg['use_cnn_stem'])
        if init_state is not None:
            model.load_state_dict(init_state)
        crit = nn.CrossEntropyLoss(label_smoothing=0.2)
        completed = train_model(model, tl, vl, DeltaRuleOptimizer, {'lr': 0.005},
                                cfg['batches1'], crit, stop_event, quick_prune=True, out_dim=out_dim)
        if not completed:
            return 0.0, 0.0, init_state
        val_acc = evaluate(model, vl, use_tta=False)
        return val_acc, val_acc, model.state_dict()
    except Exception:
        traceback.print_exc()
        return 0.0, 0.0, init_state


def eval_phase2_worker(coeffs, best_genome, init_state, in_dim, out_dim, cfg, train_ds, val_ds, worker_seed, stop_event):
    try:
        if DETERMINISTIC_MODE:
            set_seed(worker_seed)
        tl = _make_train_loader(train_ds, cfg['batch'], worker_seed)
        vl = _make_eval_loader(val_ds)
        model = DynamicNet(best_genome, in_dim, out_dim, cfg['use_cnn_stem'])
        if init_state is not None:
            model.load_state_dict(init_state)
        crit = nn.CrossEntropyLoss(label_smoothing=0.2)
        train_model(model, tl, vl, MetaOptimizer, {'coeffs': coeffs}, cfg['batches2'], crit, stop_event)
        return evaluate(model, vl, use_tta=False), model.state_dict()
    except Exception:
        traceback.print_exc()
        return 0.0, init_state


# ============================================================
# CLI logger (replaces the GUI message queue)
# ============================================================
def _print_progress(label, pct, text='', width=30):
    filled = int(width * pct / 100)
    bar = '#' * filled + '-' * (width - filled)
    sys.stdout.write(f"\r{label} [{bar}] {pct:5.1f}%  {text}")
    sys.stdout.flush()


def _print_progress_done(label, width=30):
    bar = '#' * width
    sys.stdout.write(f"\r{label} [{bar}] 100.0%\n")
    sys.stdout.flush()


class CLILogger:
    """Drop-in replacement for the GUI message queue; writes straight to stdout."""

    def __init__(self):
        self.history_p1 = []
        self.history_p2 = []
        self.start_time = time.time()
        self._active_label = None

    def emit(self, kind, **p):
        if kind == 'log':
            level = p.get('level', 'info')
            msg = p['msg']
            # Move past any in-progress bar.
            if self._active_label is not None:
                sys.stdout.write('\n')
                sys.stdout.flush()
                self._active_label = None
            if level == 'header':
                print(f"\n{C.HEADER}{C.BOLD}{msg}{C.RESET}")
            elif level == 'success':
                print(f"{C.GREEN}{msg}{C.RESET}")
            elif level == 'error':
                print(f"{C.RED}{msg}{C.RESET}")
            else:
                print(msg)
        elif kind == 'progress_p1':
            self._active_label = "  Phase 1"
            _print_progress("  Phase 1", p.get('pct', 0), p.get('text', ''))
        elif kind == 'progress_p2':
            self._active_label = "  Phase 2"
            _print_progress("  Phase 2", p.get('pct', 0), p.get('text', ''))
        elif kind == 'reset_plots':
            self.history_p1 = []
            self.history_p2 = []
            print(f"\n{C.CYAN}━━━ Dataset: {p['name']} | Seed {p['seed']} ━━━{C.RESET}")
        elif kind == 'plot_p1':
            self.history_p1 = list(p['history'])
            if self._active_label == "  Phase 1":
                _print_progress_done("  Phase 1")
                self._active_label = None
            print(f"  {C.YELLOW}Best Phase 1 acc so far: {p['acc']:.2f}%{C.RESET}")
        elif kind == 'plot_p2':
            self.history_p2 = list(p['history'])
            if self._active_label == "  Phase 2":
                _print_progress_done("  Phase 2")
                self._active_label = None
            print(f"  {C.YELLOW}Best Phase 2 acc so far: {p['acc']:.2f}% | LR: {p.get('lr', 0.001):.5f}{C.RESET}")
        elif kind == 'finished':
            elapsed = time.time() - self.start_time
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            print(f"\n{C.GREEN}{C.BOLD}✅ {p['msg']}{C.RESET}")
            print(f"{C.DIM}Total elapsed: {h:02d}:{m:02d}:{s:02d}{C.RESET}")


# ============================================================
# Phase 1 and Phase 2 (with CLI logger)
# ============================================================
def phase1(train_ds, val_ds, cfg, in_dim, out_dim, seed, emit, stop_event):
    set_seed(seed)
    pop = [{'genome': random_genome(cfg), 'state_dict': None, 'fitness': 0.0} for _ in range(cfg['pop1'])]
    for i in range(len(pop)):
        m = DynamicNet(pop[i]['genome'], in_dim, out_dim, cfg['use_cnn_stem'])
        pop[i]['state_dict'] = m.state_dict()

    best = {'acc': 0.0, 'genome': None, 'state_dict': None}
    history = []

    for gen in range(cfg['gen1']):
        if stop_event.is_set():
            break

        emit('log', msg=f"🚀 [Phase 1 | Gen {gen + 1}/{cfg['gen1']}] Searching Dense Architectures...", level='info')
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(eval_phase1_worker, pop[i]['genome'], pop[i]['state_dict'], in_dim, out_dim, cfg,
                          train_ds, val_ds, seed * 10000 + gen * 100 + i, stop_event): i
                for i in range(len(pop))
            }
            completed = 0
            for fut in as_completed(futures):
                if stop_event.is_set():
                    break
                i = futures[fut]
                try:
                    val_acc, score, new_state = fut.result()
                except Exception:
                    traceback.print_exc()
                    continue
                pop[i]['fitness'] = score
                pop[i]['state_dict'] = new_state
                if val_acc > best['acc']:
                    best = {'acc': val_acc, 'genome': copy.deepcopy(pop[i]['genome']),
                            'state_dict': copy.deepcopy(new_state)}
                    emit('log',
                         msg=f"  💾 Better Arch: {genome_to_str(pop[i]['genome'])} | Val: {val_acc:.2f}%",
                         level='success')
                completed += 1
                pct = ((gen * cfg['pop1'] + completed) / (cfg['gen1'] * cfg['pop1'])) * 100
                emit('progress_p1', pct=pct, text=f"Gen {gen + 1}/{cfg['gen1']} ({completed}/{cfg['pop1']})")

        history.append(best['acc'])
        emit('plot_p1', history=history, acc=best['acc'])

        pop = sorted(pop, key=lambda x: x['fitness'], reverse=True)
        elites = pop[:max(2, cfg['pop1'] // 5)]
        new_pop = copy.deepcopy(elites)
        while len(new_pop) < cfg['pop1']:
            p1, p2 = (random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0]))
            child_g = mutate_genome(crossover_genome(p1['genome'], p2['genome']), cfg)
            child_model = DynamicNet(child_g, in_dim, out_dim, cfg['use_cnn_stem'])
            if p1['state_dict']:
                try:
                    parent = DynamicNet(p1['genome'], in_dim, out_dim, cfg['use_cnn_stem'])
                    parent.load_state_dict(p1['state_dict'])
                    transfer_weights(child_model, parent, alpha=0.7)
                except Exception:
                    pass
            new_pop.append({'genome': child_g, 'state_dict': child_model.state_dict(), 'fitness': 0.0})
        pop = new_pop
    return best


def phase2(best_genome, best_state, train_ds, val_ds, cfg, in_dim, out_dim, seed, emit, stop_event):
    set_seed(seed + 7777)
    pop = [random_meta_coeffs() for _ in range(cfg['pop2'])]
    init_states = [copy.deepcopy(best_state) for _ in range(cfg['pop2'])]
    best = {'acc': 0.0, 'coeffs': None, 'state_dict': None}
    history = []

    for gen in range(cfg['gen2']):
        if stop_event.is_set():
            break

        emit('log', msg=f"🚀 [Phase 2 | Gen {gen + 1}/{cfg['gen2']}] Evolving Coefficients (Warm-start)...",
             level='info')
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(eval_phase2_worker, pop[i], best_genome, init_states[i], in_dim, out_dim, cfg,
                          train_ds, val_ds, seed * 20000 + gen * 100 + i, stop_event): i
                for i in range(len(pop))
            }
            completed = 0
            fitness = [0.0] * len(pop)
            for fut in as_completed(futures):
                if stop_event.is_set():
                    break
                i = futures[fut]
                try:
                    val_acc, state_dict = fut.result()
                except Exception:
                    traceback.print_exc()
                    continue
                fitness[i] = val_acc
                if val_acc > best['acc']:
                    best = {'acc': val_acc, 'coeffs': copy.deepcopy(pop[i]), 'state_dict': state_dict}
                    emit('log',
                         msg=f"  💾 Better Coeffs: {coeffs_to_str(pop[i])} | Val: {val_acc:.2f}%",
                         level='success')
                completed += 1
                pct = ((gen * cfg['pop2'] + completed) / (cfg['gen2'] * cfg['pop2'])) * 100
                emit('progress_p2', pct=pct, text=f"Gen {gen + 1}/{cfg['gen2']} ({completed}/{cfg['pop2']})")

        history.append(best['acc'])
        lr_val = _decode_coeffs(best['coeffs'])[3] if best['coeffs'] else 0.001
        emit('plot_p2', history=history, acc=best['acc'], lr=lr_val)

        ranked = [c for _, c in sorted(zip(fitness, pop), key=lambda p: p[0], reverse=True)]
        elites = ranked[:max(2, cfg['pop2'] // 5)]
        new_pop = copy.deepcopy(elites)
        while len(new_pop) < cfg['pop2']:
            p1, p2 = (random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0]))
            child = crossover_meta_coeffs(p1, p2)
            if random.random() < 0.4:
                child = mutate_meta_coeffs(child)
            new_pop.append(child)
        pop = new_pop
    return best


# ============================================================
# Dataset loaders
# ============================================================
def _ensure_mnist1d_data():
    """Make sure the MNIST-1D pickle exists; download it from the official repo if not."""
    file_path = os.path.join(DATA_MNIST_DIR, 'mnist1d_data.pkl')
    if os.path.exists(file_path):
        return file_path
    print(f"{C.CYAN}⬇ Downloading MNIST-1D dataset from {MNIST1D_URL}{C.RESET}")
    os.makedirs(DATA_MNIST_DIR, exist_ok=True)
    try:
        urllib.request.urlretrieve(MNIST1D_URL, file_path)
        print(f"{C.GREEN}✓ Saved to {file_path}{C.RESET}")
        return file_path
    except Exception as e:
        print(f"{C.RED}✗ Failed to download MNIST-1D: {e}{C.RESET}")
        print(f"{C.YELLOW}  Please manually place mnist1d_data.pkl in {DATA_MNIST_DIR}{C.RESET}")
        raise


def load_cifar10(cfg, seed):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_full = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_train)
    eval_full = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_eval)
    test_full = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_eval)
    y = np.array(train_full.targets)
    pool_idx, _ = train_test_split(np.arange(len(y)), train_size=cfg['n_train'] + cfg['n_val'],
                                  stratify=y, random_state=seed)
    tr_idx, va_idx = train_test_split(pool_idx, train_size=cfg['n_train'], stratify=y[pool_idx], random_state=seed)
    y_test = np.array(test_full.targets)
    te_idx, _ = train_test_split(np.arange(len(y_test)), train_size=cfg['n_test'], stratify=y_test, random_state=seed)
    with torch.no_grad():
        val_x = torch.stack([eval_full[i][0] for i in va_idx.tolist()])
        val_y = torch.tensor([eval_full[i][1] for i in va_idx.tolist()])
        test_x = torch.stack([test_full[i][0] for i in te_idx.tolist()])
        test_y = torch.tensor([test_full[i][1] for i in te_idx.tolist()])
    return (Subset(train_full, tr_idx.tolist()),
            TensorDataset(val_x, val_y),
            TensorDataset(test_x, test_y),
            3 * 32 * 32, 10, True)


def load_mnist1d(cfg, seed):
    file_path = _ensure_mnist1d_data()
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    in_dim = X_train.shape[1]
    out_dim = len(np.unique(y_train))
    X_train = (X_train - X_train.mean(0)) / (X_train.std(0) + 1e-8)
    X_test = (X_test - X_test.mean(0)) / (X_test.std(0) + 1e-8)
    X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed)
    return (TensorDataset(X_tr, y_tr),
            TensorDataset(X_va, y_va),
            TensorDataset(X_test, y_test),
            in_dim, out_dim, False)


# ============================================================
# Main experiment runner
# ============================================================
def run_experiment(cli_cfg, stop_event):
    logger = CLILogger()
    emit = logger.emit
    try:
        base_configs = load_configs()
        all_results = []
        for dataset_name in cli_cfg['datasets']:
            if stop_event.is_set():
                break
            if dataset_name not in base_configs:
                continue
            cfg = copy.deepcopy(base_configs[dataset_name])
            cfg['gen1'] = cli_cfg['gen1']
            cfg['gen2'] = cli_cfg['gen2']
            dataset_label = cfg.get('label', dataset_name)

            for seed in range(cli_cfg['num_seeds']):
                if stop_event.is_set():
                    break
                t0 = time.time()
                emit('log', msg=f">>> [{dataset_label}] Seed {seed} - Loading...", level='header')

                if dataset_name.startswith('cifar'):
                    train_ds, val_ds, test_ds, in_dim, out_dim, use_stem = load_cifar10(cfg, seed)
                elif dataset_name == 'mnist1d':
                    train_ds, val_ds, test_ds, in_dim, out_dim, use_stem = load_mnist1d(cfg, seed)
                else:
                    continue

                cfg['use_cnn_stem'] = use_stem
                cfg['out_dim'] = out_dim
                emit('reset_plots', name=dataset_label, seed=seed)

                b1 = phase1(train_ds, val_ds, cfg, in_dim, out_dim, seed, emit, stop_event)
                if stop_event.is_set() or b1['genome'] is None:
                    break

                b2 = phase2(b1['genome'], b1['state_dict'], train_ds, val_ds, cfg, in_dim, out_dim, seed, emit,
                            stop_event)
                if stop_event.is_set() or b2['coeffs'] is None:
                    break

                m = DynamicNet(b1['genome'], in_dim, out_dim, cfg['use_cnn_stem'])
                m.load_state_dict(b2['state_dict'])
                test_loader = _make_eval_loader(test_ds)
                test_acc = evaluate(m, test_loader, use_tta=cfg['use_cnn_stem'])

                elapsed = time.time() - t0
                emit('log',
                     msg=f"🎉 [{dataset_label} | Seed {seed}] Final Test: {test_acc:.2f}% | Time: {elapsed:.0f}s",
                     level='header')
                emit('log', msg=f"🧬 Best Coeffs: {coeffs_to_str(b2['coeffs'])}", level='success')
                emit('log', msg=f"🏗️ Best Arch: {genome_to_str(b1['genome'])}", level='success')

                record = {
                    'dataset': dataset_name,
                    'dataset_label': dataset_label,
                    'seed': seed,
                    'acc': test_acc,
                    'genome': b1['genome'],
                    'coeffs': b2['coeffs'],
                    'val_acc_phase1': b1['acc'],
                    'val_acc_phase2': b2['acc'],
                    'history_phase1': logger.history_p1,
                    'history_phase2': logger.history_p2,
                    'elapsed_seconds': elapsed,
                    'transfer_errors': TRANSFER_ERRORS,
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                }
                all_results.append(record)
                safe_name = dataset_name
                try:
                    with open(os.path.join(RESULTS_DIR, f'{safe_name}_seed{seed}.json'), 'w', encoding='utf-8') as f:
                        json.dump(record, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    emit('log', msg=f"⚠️ Failed to save result: {e}", level='error')

        if all_results:
            summary_path = os.path.join(RESULTS_DIR, 'summary.json')
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                emit('log', msg=f"📊 Saved summary of {len(all_results)} run(s) to {summary_path}",
                     level='success')
            except Exception as e:
                emit('log', msg=f"⚠️ Failed to save summary: {e}", level='error')

        emit('finished', msg="All experiments finished successfully.")
    except Exception as e:
        emit('log', msg=f"❌ Error: {e}", level='error')
        traceback.print_exc()
        emit('finished', msg="Stopped with error.")


# ============================================================
# Interactive menu
# ============================================================
def _read_int(prompt, lo, hi, default=None):
    while True:
        try:
            s = input(prompt).strip()
            if not s and default is not None:
                return default
            v = int(s)
            if lo <= v <= hi:
                return v
            print(f"{C.YELLOW}  → Please enter a number between {lo} and {hi}.{C.RESET}")
        except (ValueError, EOFError, KeyboardInterrupt):
            if default is not None:
                return default
            raise


def interactive_menu():
    print(f"\n{C.BOLD}{C.HEADER}╔════════════════════════════════════════════════════════════╗")
    print(f"║   Neuro-Evolution CLI — Two-Phase Architecture Search     ║")
    print(f"╚════════════════════════════════════════════════════════════╝{C.RESET}\n")

    print(f"{C.BOLD}Select a dataset:{C.RESET}")
    for i, key in enumerate(DATASET_KEYS, 1):
        cfg = DEFAULT_CONFIGS[key]
        print(f"  {C.CYAN}{i}.{C.RESET} {cfg['label']:<22} "
              f"{C.DIM}(n_train={cfg['n_train']}, pop1={cfg['pop1']}, batches1={cfg['batches1']}){C.RESET}")
    print()

    sel = _read_int(f"Enter choice [1-{len(DATASET_KEYS)}] (default=1): ", 1, len(DATASET_KEYS), default=1)
    dataset_key = DATASET_KEYS[sel - 1]
    print(f"{C.GREEN}✓ Selected: {DEFAULT_CONFIGS[dataset_key]['label']}{C.RESET}\n")

    num_seeds = _read_int("Number of seeds [1-10] (default=1): ", 1, 10, default=1)
    gen1 = _read_int("Generations for Phase 1 [1-20] (default=5): ", 1, 20, default=5)
    gen2 = _read_int("Generations for Phase 2 [1-20] (default=5): ", 1, 20, default=5)

    print(f"\n{C.BOLD}Configuration summary:{C.RESET}")
    print(f"  Dataset  : {DEFAULT_CONFIGS[dataset_key]['label']}")
    print(f"  Seeds    : {num_seeds}")
    print(f"  Gen1     : {gen1}")
    print(f"  Gen2     : {gen2}")
    print(f"  Workers  : {MAX_WORKERS}")
    print(f"  Device   : CPU (torch threads={torch.get_num_threads()})")

    confirm = input(f"\n{C.YELLOW}Start experiment? [Y/n]: {C.RESET}").strip().lower()
    if confirm and confirm not in ('y', 'yes'):
        print(f"{C.YELLOW}Aborted by user.{C.RESET}")
        sys.exit(0)

    return {
        'datasets': [dataset_key],
        'num_seeds': num_seeds,
        'gen1': gen1,
        'gen2': gen2,
    }


# ============================================================
# Entry point with signal handling
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Evolution CLI: Two-phase neural architecture search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # interactive menu
  python main.py --dataset cifar_small        # CIFAR-10 (Small), defaults
  python main.py --dataset mnist1d --seeds 3 --gen1 8 --gen2 8
        """,
    )
    parser.add_argument('--dataset', choices=DATASET_KEYS, default=None,
                        help="Dataset key. If omitted, an interactive menu is shown.")
    parser.add_argument('--seeds', type=int, default=1, help="Number of seeds (default: 1)")
    parser.add_argument('--gen1', type=int, default=5, help="Phase 1 generations (default: 5)")
    parser.add_argument('--gen2', type=int, default=5, help="Phase 2 generations (default: 5)")
    args = parser.parse_args()

    if args.dataset:
        cli_cfg = {
            'datasets': [args.dataset],
            'num_seeds': max(1, args.seeds),
            'gen1': max(1, args.gen1),
            'gen2': max(1, args.gen2),
        }
        print(f"\n{C.BOLD}Dataset: {DEFAULT_CONFIGS[args.dataset]['label']} "
              f"| seeds={cli_cfg['num_seeds']} gen1={cli_cfg['gen1']} gen2={cli_cfg['gen2']}{C.RESET}\n")
    else:
        cli_cfg = interactive_menu()

    stop_event = threading.Event()

    def _on_sigint(signum, frame):
        if stop_event.is_set():
            print(f"\n{C.RED}Force quitting.{C.RESET}")
            sys.exit(1)
        print(f"\n{C.YELLOW}⚠ Ctrl+C received — stopping after current worker finishes. "
              f"Press Ctrl+C again to force quit.{C.RESET}")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    run_experiment(cli_cfg, stop_event)


if __name__ == '__main__':
    main()

import os
import sys
import time
import random
import copy
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# ⚙️ تنظیمات سخت‌افزاری و مسیرها
# ============================================================
os.environ["PYTHONHASHSEED"] = "0"
torch.backends.mkldnn.enabled = True
_N_CPU = os.cpu_count() or 4
MAX_WORKERS = max(2, min(6, _N_CPU // 2 if _N_CPU >= 8 else _N_CPU))

# این خط بسیار مهم است: برای جلوگیری از تداخل تردها
torch.set_num_threads(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(SCRIPT_DIR, 'data')
DATA_MNIST_DIR = os.path.join(SCRIPT_DIR, 'data_mnist1d')
DATA_BIO_DIR   = os.path.join(SCRIPT_DIR, 'data_bio')
RESULTS_DIR    = os.path.join(SCRIPT_DIR, 'results')

for d in [DATA_DIR, DATA_MNIST_DIR, DATA_BIO_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 📋 Config
# ============================================================
DEFAULT_CONFIGS = {
    "CIFAR-10 (Small)": {"pop1": 12, "batches1": 150, "pop2": 8, "batches2": 150, "batch": 128, "n_train": 2000, "n_val": 500, "n_test": 500, "act_funcs": ["tanh"], "size_pool": [64, 128], "min_layers": 1, "max_layers": 5},
    "CIFAR-10 (Medium)": {"pop1": 8, "batches1": 200, "pop2": 8, "batches2": 400, "batch": 128, "n_train": 10000, "n_val": 2000, "n_test": 2000, "act_funcs": ["tanh"], "size_pool": [128, 256], "min_layers": 1, "max_layers": 6},
    "CIFAR-10 (Large)": {"pop1": 8, "batches1": 400, "pop2": 8, "batches2": 800, "batch": 256, "n_train": 20000, "n_val": 5000, "n_test": 5000, "act_funcs": ["tanh"], "size_pool": [256, 512], "min_layers": 1, "max_layers": 7},
    "MNIST-1D": {"pop1": 8, "batches1": 300, "pop2": 8, "batches2": 500, "batch": 64, "n_train": 4000, "n_val": 1000, "n_test": 1000, "act_funcs": ["tanh"], "size_pool": [128, 256, 512], "min_layers": 1, "max_layers": 6},
    "Bio (Promoter)": {"pop1": 8, "batches1": 80, "pop2": 8, "batches2": 100, "batch": 16, "n_train": 86, "n_val": 10, "n_test": 10, "act_funcs": ["tanh"], "size_pool": [16, 32, 64], "min_layers": 1, "max_layers": 4},
}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ============================================================
# 🧮 توابع ریاضی و بهینه‌ساز
# ============================================================
def softplus(x):
    if x > 20: return x
    if x < -20: return math.exp(x)
    return math.log1p(math.exp(x))

def _sigmoid(x):
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x); return ez / (1.0 + ez)

def _decode_coeffs(c):
    a  = softplus(c['a'])
    b  = 0.5 + 0.5 * math.tanh(c['b'])
    d  = 0.1 * _sigmoid(c['d'])
    lr = softplus(c['lr'])
    return a, b, d, lr

class DeltaRuleOptimizer:
    def __init__(self, params, lr=0.005):
        self.params = list(params); self.lr = lr
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            p.data.add_(p.grad, alpha=-self.lr)

class MetaOptimizer:
    def __init__(self, params, coeffs):
        self.params  = list(params)
        self.coeffs  = coeffs
        self.lr      = softplus(coeffs['lr'])
        self.initial_lr = self.lr
        self.mom = [torch.zeros_like(p.data) for p in self.params]
        self.var = [torch.zeros_like(p.data) for p in self.params]
        self.t   = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()

    def step(self):
        a, b, d, lr = _decode_coeffs(self.coeffs)
        eps = 1e-8
        self.t += 1
        bc1 = 1.0 - 0.9   ** self.t
        bc2 = 1.0 - 0.999 ** self.t
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None: continue
                g = p.grad
                m = self.mom[i]; v = self.var[i]
                m.mul_(0.9).add_(g, alpha=0.1)
                v.mul_(0.999).addcmul_(g, g, value=0.001)
                m_hat = m / bc1
                v_hat = v / bc2
                p.data.mul_(1.0 - lr * d)
                denom = (v_hat + eps).pow(b)
                p.data.addcdiv_(m_hat, denom, value=-lr * a)

def random_meta_coeffs():
    return {'a': random.uniform(-2.0, 2.0), 'b': random.uniform(-2.0, 2.0), 'd': random.uniform(-2.0, 2.0), 'lr': random.uniform(-10.0, -4.0)}

def mutate_meta_coeffs(c):
    c = c.copy()
    if random.random() < 0.2: return random_meta_coeffs()
    for k in ['a', 'b', 'd', 'lr']:
        if random.random() < 0.3: c[k] += random.uniform(-0.5, 0.5)
    return c

def crossover_meta_coeffs(c1, c2):
    c = c1.copy()
    for k in ['a', 'b', 'd', 'lr']:
        if random.random() < 0.5: c[k] = c2[k]
    return c

def coeffs_to_str(c):
    a, b, d, lr = _decode_coeffs(c)
    return f"a={a:.3f}, b={b:.3f}, d={d:.5f}, lr={lr:.5f}"

# ============================================================
# 🏗 شبکه پویا
# ============================================================
class DynamicNet(nn.Module):
    def __init__(self, genome, in_dim, out_dim, use_cnn_stem):
        super().__init__()
        self.genome = genome; self.use_cnn_stem = use_cnn_stem

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

        self.transition = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        base_dim = 128

        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        current_dim = base_dim
        for i, (sz, act) in enumerate(zip(genome['layers'], genome['acts'])):
            self.layers.append(nn.Linear(current_dim, sz))
            self.bns.append(nn.BatchNorm1d(sz) if sz > 1 else nn.Identity())
            self.acts.append(self._act(act))
            self.dropouts.append(nn.Dropout(0.3)) 
            current_dim += sz

        self.out_layer = nn.Linear(current_dim, out_dim)

    @staticmethod
    def _act(name):
        return {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'gelu': nn.GELU(), 'leaky_relu': nn.LeakyReLU(0.01), 'elu': nn.ELU()}[name]

    def forward(self, x):
        if not self.use_cnn_stem: x = x.unsqueeze(1)
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
# 🧬 توابع ژنوم
# ============================================================
def random_genome(cfg):
    n = random.randint(cfg.get('min_layers', 1), cfg.get('max_layers', 6))
    return {'layers': [random.choice(cfg['size_pool']) for _ in range(n)], 'acts': [random.choice(cfg['act_funcs']) for _ in range(n)]}

def mutate_genome(g, cfg):
    g = copy.deepcopy(g)
    if random.random() < 0.2: return random_genome(cfg)
    if random.random() < 0.4 and len(g['layers']) > 0:
        idx = random.randrange(len(g['layers']))
        g['layers'][idx] = random.choice(cfg['size_pool'])
    if random.random() < 0.2 and len(g['layers']) < cfg.get('max_layers', 6):
        i = random.randint(0, len(g['layers']))
        g['layers'].insert(i, random.choice(cfg['size_pool']))
        g['acts'].insert(i, random.choice(cfg['act_funcs']))
    if random.random() < 0.2 and len(g['layers']) > cfg.get('min_layers', 1):
        i = random.randrange(len(g['layers']))
        g['layers'].pop(i); g['acts'].pop(i)
    return g

def crossover_genome(g1, g2):
    max_len = max(len(g1['layers']), len(g2['layers']))
    new_layers, new_acts = [], []
    for i in range(max_len):
        a_ok = i < len(g1['layers']); b_ok = i < len(g2['layers'])
        if a_ok and b_ok:
            if random.random() < 0.5:
                new_layers.append(g1['layers'][i]); new_acts.append(g1['acts'][i])
            else:
                new_layers.append(g2['layers'][i]); new_acts.append(g2['acts'][i])
        elif a_ok and random.random() < 0.5:
            new_layers.append(g1['layers'][i]); new_acts.append(g1['acts'][i])
        elif b_ok and random.random() < 0.5:
            new_layers.append(g2['layers'][i]); new_acts.append(g2['acts'][i])
    if not new_layers:
        new_layers = [random.choice([g1['layers'][0], g2['layers'][0]])]
        new_acts   = [random.choice([g1['acts'][0],   g2['acts'][0]])]
    return {'layers': new_layers, 'acts': new_acts}

def genome_to_str(g):
    return f"L={g['layers']} A={g['acts']}"

# ============================================================
# 🚀 آموزش و ارزیابی (با Cosine Annealing)
# ============================================================
def _make_train_loader(dataset, batch_size, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

def _make_eval_loader(dataset, batch_size=512):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def train_model(model, train_loader, val_loader, opt_cls, opt_kwargs, max_batches, crit):
    if opt_cls == DeltaRuleOptimizer:
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        opt = opt_cls(model.parameters(), **opt_kwargs)
        
    model.train()
    n = 0
    while n < max_batches:
        for x, y in train_loader:
            if n >= max_batches: break
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); n += 1
            
            # 🌟 بخش بسیار مهم: کاهش کسینوسی نرخ یادگیری
            if hasattr(opt, 'initial_lr'):
                progress = n / max_batches
                opt.lr = opt.initial_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
                
    model.eval(); return True

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(1, total)

# ============================================================
# 🧬 فاز ۱ و ۲ (موازی)
# ============================================================
def eval_p1_worker(genome, init_state, in_dim, out_dim, cfg, train_ds, val_ds, worker_seed):
    try:
        set_seed(worker_seed)
        tl = _make_train_loader(train_ds, cfg['batch'], worker_seed)
        vl = _make_eval_loader(val_ds)
        model = DynamicNet(genome, in_dim, out_dim, cfg['use_cnn_stem'])
        if init_state is not None: model.load_state_dict(init_state)
        crit = nn.CrossEntropyLoss(label_smoothing=0.2)
        train_model(model, tl, vl, DeltaRuleOptimizer, {'lr': 0.005}, cfg['batches1'], crit)
        return evaluate(model, vl), model.state_dict()
    except Exception as e:
        print(f"Error in P1 worker: {e}")
        return 0.0, init_state

def eval_p2_worker(coeffs, best_genome, init_state, in_dim, out_dim, cfg, train_ds, val_ds, worker_seed):
    try:
        set_seed(worker_seed)
        tl = _make_train_loader(train_ds, cfg['batch'], worker_seed)
        vl = _make_eval_loader(val_ds)
        model = DynamicNet(best_genome, in_dim, out_dim, cfg['use_cnn_stem'])
        if init_state is not None: model.load_state_dict(init_state)
        crit = nn.CrossEntropyLoss(label_smoothing=0.2)
        train_model(model, tl, vl, MetaOptimizer, {'coeffs': coeffs}, cfg['batches2'], crit)
        return evaluate(model, vl), model.state_dict()
    except Exception as e:
        print(f"Error in P2 worker: {e}")
        return 0.0, init_state

def phase1(train_ds, val_ds, cfg, in_dim, out_dim, seed):
    pop = [{'genome': random_genome(cfg), 'state_dict': None, 'fitness': 0.0} for _ in range(cfg['pop1'])]
    for i in range(len(pop)):
        pop[i]['state_dict'] = DynamicNet(pop[i]['genome'], in_dim, out_dim, cfg['use_cnn_stem']).state_dict()

    best = {'acc': 0.0, 'genome': None, 'state_dict': None}
    
    for gen in range(cfg['gen1']):
        print(f"\n🚀 [Phase 1 | Gen {gen+1}/{cfg['gen1']}] Evaluating {len(pop)} architectures in parallel...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(eval_p1_worker, pop[i]['genome'], pop[i]['state_dict'], in_dim, out_dim, cfg, train_ds, val_ds, seed*10000+gen*100+i): i for i in range(len(pop))}
            for fut in as_completed(futures):
                i = futures[fut]
                val_acc, new_state = fut.result()
                pop[i]['fitness'] = val_acc; pop[i]['state_dict'] = new_state
                if val_acc > best['acc']:
                    best = {'acc': val_acc, 'genome': copy.deepcopy(pop[i]['genome']), 'state_dict': copy.deepcopy(new_state)}
                    print(f"  💾 Better Arch: {genome_to_str(pop[i]['genome'])} | Val: {val_acc:.2f}%")

        pop = sorted(pop, key=lambda x: x['fitness'], reverse=True)
        elites = pop[:max(2, cfg['pop1'] // 5)]
        new_pop = copy.deepcopy(elites)
        while len(new_pop) < cfg['pop1']:
            p1, p2 = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
            child_g = mutate_genome(crossover_genome(p1['genome'], p2['genome']), cfg)
            child_model = DynamicNet(child_g, in_dim, out_dim, cfg['use_cnn_stem'])
            new_pop.append({'genome': child_g, 'state_dict': child_model.state_dict(), 'fitness': 0.0})
        pop = new_pop
    return best

def phase2(best_genome, best_state, train_ds, val_ds, cfg, in_dim, out_dim, seed):
    pop = [random_meta_coeffs() for _ in range(cfg['pop2'])]
    init_states = [copy.deepcopy(best_state) for _ in range(cfg['pop2'])]
    best = {'acc': 0.0, 'coeffs': None, 'state_dict': None}
    
    for gen in range(cfg['gen2']):
        print(f"\n🚀 [Phase 2 | Gen {gen+1}/{cfg['gen2']}] Evaluating {len(pop)} coefficients in parallel...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(eval_p2_worker, pop[i], best_genome, init_states[i], in_dim, out_dim, cfg, train_ds, val_ds, seed*20000+gen*100+i): i for i in range(len(pop))}
            fitness = [0.0] * len(pop)
            for fut in as_completed(futures):
                i = futures[fut]
                val_acc, state_dict = fut.result()
                fitness[i] = val_acc
                if val_acc > best['acc']:
                    best = {'acc': val_acc, 'coeffs': copy.deepcopy(pop[i]), 'state_dict': state_dict}
                    print(f"  💾 Better Coeffs: {coeffs_to_str(pop[i])} | Val: {val_acc:.2f}%")
                    
        ranked = [c for _, c in sorted(zip(fitness, pop), key=lambda p: p[0], reverse=True)]
        elites = ranked[:max(2, cfg['pop2'] // 5)]
        new_pop = copy.deepcopy(elites)
        while len(new_pop) < cfg['pop2']:
            p1, p2 = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
            child = crossover_meta_coeffs(p1, p2)
            if random.random() < 0.4: child = mutate_meta_coeffs(child)
            new_pop.append(child)
        pop = new_pop
    return best

# ============================================================
# 📥 بارگذاری دیتاست‌ها (خلاصه شده)
# ============================================================
def load_cifar10(cfg, seed):
    tf_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
    tf_eval = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
    train_full = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_train)
    eval_full  = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_eval)
    test_full  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_eval)
    y = np.array(train_full.targets)
    pool_idx, _ = train_test_split(np.arange(len(y)), train_size=cfg['n_train']+cfg['n_val'], stratify=y, random_state=seed)
    tr_idx, va_idx = train_test_split(pool_idx, train_size=cfg['n_train'], stratify=y[pool_idx], random_state=seed)
    y_test = np.array(test_full.targets)
    te_idx, _ = train_test_split(np.arange(len(y_test)), train_size=cfg['n_test'], stratify=y_test, random_state=seed)
    val_x  = torch.stack([eval_full[i][0] for i in va_idx.tolist()]); val_y  = torch.tensor([eval_full[i][1] for i in va_idx.tolist()])
    test_x = torch.stack([test_full[i][0] for i in te_idx.tolist()]); test_y = torch.tensor([test_full[i][1] for i in te_idx.tolist()])
    return Subset(train_full, tr_idx.tolist()), TensorDataset(val_x, val_y), TensorDataset(test_x, test_y), 3*32*32, 10, True

# ============================================================
# 🏃 اجرای اصلی (خط فرمان)
# ============================================================
def main():
    print("="*60)
    print(" Joint Evolutionary Discovery Framework (Parallel Version)")
    print("="*60)
    
    datasets = list(DEFAULT_CONFIGS.keys())
    print("\nPlease select a dataset configuration to run:")
    for i, name in enumerate(datasets):
        print(f" {i+1}. {name}")
        
    try:
        choice = int(input(f"\nEnter your choice (1-{len(datasets)}) [Default: 2]: ") or "2") - 1
        dataset_name = datasets[choice]
        seed = int(input("Enter number of random seeds (Default: 1): ") or "1")
        gen1 = int(input("Enter generations for Phase 1 & 2 (Default: 5): ") or "5")
        gen2 = gen1
    except ValueError:
        print("Invalid input. Exiting.")
        return

    cfg = copy.deepcopy(DEFAULT_CONFIGS[dataset_name])
    cfg['gen1'] = gen1; cfg['gen2'] = gen2
    
    print(f"\n>>> [{dataset_name}] Seed {seed-1} - Loading...")
    
    if dataset_name.startswith('CIFAR'):
        train_ds, val_ds, test_ds, in_dim, out_dim, use_stem = load_cifar10(cfg, seed)
    else:
        print("Dataset loading for this type is not implemented in this snippet.")
        return

    cfg['use_cnn_stem'] = use_stem; cfg['out_dim'] = out_dim

    t0 = time.time()
    b1 = phase1(train_ds, val_ds, cfg, in_dim, out_dim, seed)
    if b1['genome'] is None: return
    
    b2 = phase2(b1['genome'], b1['state_dict'], train_ds, val_ds, cfg, in_dim, out_dim, seed)
    if b2['coeffs'] is None: return

    m = DynamicNet(b1['genome'], in_dim, out_dim, cfg['use_cnn_stem'])
    m.load_state_dict(b2['state_dict'])
    test_acc = evaluate(m, _make_eval_loader(test_ds))
    
    elapsed = time.time() - t0
    print(f"\n🎉 [{dataset_name} | Seed {seed-1}] Final Test: {test_acc:.2f}% | Time: {elapsed:.0f}s")
    print(f"🧬 Best Coeffs: {coeffs_to_str(b2['coeffs'])}")
    print(f"🏗️ Best Arch: {genome_to_str(b1['genome'])}")

if __name__ == '__main__':
    main()

import os, sys, time, copy, random, json, math, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torchaudio
    import torchaudio.transforms as T
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ============================================================
# ⚙️ Global Settings and Paths
# ============================================================
torch.backends.mkldnn.enabled = True
_N_CPU = os.cpu_count() or 4
_GLOBAL_THREADS = max(1, min(4, _N_CPU // 2))
torch.set_num_threads(_GLOBAL_THREADS)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DATA_MNIST_DIR = os.path.join(SCRIPT_DIR, 'data_mnist1d')
DATA_BIO_DIR = os.path.join(SCRIPT_DIR, 'data_bio')
DATA_AUDIO_DIR = os.path.join(SCRIPT_DIR, 'data_audio')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

for d in [DATA_DIR, DATA_MNIST_DIR, DATA_BIO_DIR, DATA_AUDIO_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

MAX_WORKERS = max(2, min(6, _N_CPU // 2 if _N_CPU >= 8 else _N_CPU))
USE_CHANNELS_LAST = True
HEAVY_AUG = False

TRANSFER_ERRORS = 0
LOAD_STATE_ERRORS = 0

# ============================================================
# 🧠 Optimizers
# ============================================================
def softplus(x):
    if x > 20: return x
    if x < -20: return math.exp(x)
    return math.log1p(math.exp(x))

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
        self.params = list(params); self.coeffs = coeffs
        self.lr = softplus(coeffs['lr'])
        self.initial_lr = self.lr
        self.mom = [torch.zeros_like(p.data) for p in self.params]
        self.var = [torch.zeros_like(p.data) for p in self.params]
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
    def step(self):
        a = softplus(self.coeffs['a'])
        b = softplus(self.coeffs['b'])
        d = 0.01 * softplus(self.coeffs['d'])
        eps = 1e-8
        lr = self.lr
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None: continue
                g = p.grad
                m = self.mom[i]; v = self.var[i]
                m.mul_(0.9).add_(g, alpha=0.1)
                v.mul_(0.999).addcmul_(g, g, value=0.001)
                denom = (v + eps).pow_(b)
                p.data.addcdiv_(m, denom, value=-lr * a)
                p.data.add_(p.data, alpha=-lr * d)

def random_meta_coeffs():
    return {'a': random.uniform(-2.0, 2.0), 'b': random.uniform(-2.0, 2.0),
            'd': random.uniform(-2.0, 2.0), 'lr': random.uniform(-10.0, -4.0)}

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
    return f"a={softplus(c['a']):.3f}, b={softplus(c['b']):.3f}, d={softplus(c['d']):.3f}, lr={softplus(c['lr']):.5f}"

# ============================================================
# 🏗 Network and Genome
# ============================================================
class DynamicNet(nn.Module):
    def __init__(self, genome, in_dim, out_dim, use_cnn_stem):
        super().__init__()
        self.genome = genome; self.use_cnn_stem = use_cnn_stem

        if use_cnn_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
            )
            flat_dim = 128 * 4 * 4
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2),
            )
            flat_dim = 32 * 10

        mods, prev = [], flat_dim
        for sz, act in zip(genome['layers'], genome['acts']):
            mods.append(nn.Linear(prev, sz))
            mods.append(self._act(act))
            if sz > 1: mods.append(nn.BatchNorm1d(sz))
            if sz >= 128: mods.append(nn.Dropout(0.2))
            prev = sz
        mods.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*mods)

    @staticmethod
    def _act(name):
        return {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(),
                'gelu': nn.GELU(), 'leaky_relu': nn.LeakyReLU(0.01), 'elu': nn.ELU()}[name]

    def forward(self, x):
        if not self.use_cnn_stem:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = torch.flatten(x, 1)
        return self.net(x)

def random_genome(cfg):
    min_l = cfg.get('min_layers', 1)
    max_l = cfg.get('max_layers', 6)
    n = random.randint(min_l, max_l)
    return {'layers': [random.choice(cfg['size_pool']) for _ in range(n)],
            'acts': [random.choice(cfg['act_funcs']) for _ in range(n)]}

def mutate_genome(g, cfg):
    g = copy.deepcopy(g)
    if random.random() < 0.2: return random_genome(cfg)
    if random.random() < 0.4 and len(g['layers']) > 0:
        idx = random.randrange(len(g['layers']))
        g['layers'][idx] = random.choice(cfg['size_pool'])
    if len(cfg['act_funcs']) > 1 and random.random() < 0.3 and len(g['acts']) > 0:
        idx = random.randrange(len(g['acts']))
        g['acts'][idx] = random.choice(cfg['act_funcs'])
    if random.random() < 0.2 and len(g['layers']) < cfg.get('max_layers', 6):
        insert_idx = random.randint(0, len(g['layers']))
        g['layers'].insert(insert_idx, random.choice(cfg['size_pool']))
        g['acts'].insert(insert_idx, random.choice(cfg['act_funcs']))
    if random.random() < 0.2 and len(g['layers']) > cfg.get('min_layers', 1):
        remove_idx = random.randrange(len(g['layers']))
        g['layers'].pop(remove_idx); g['acts'].pop(remove_idx)
    return g

def crossover_genome(g1, g2):
    max_len = max(len(g1['layers']), len(g2['layers']))
    new_layers, new_acts = [], []
    for i in range(max_len):
        if i < len(g1['layers']) and i < len(g2['layers']):
            if random.random() < 0.5:
                new_layers.append(g1['layers'][i]); new_acts.append(g1['acts'][i])
            else:
                new_layers.append(g2['layers'][i]); new_acts.append(g2['acts'][i])
        elif i < len(g1['layers']):
            if random.random() < 0.5:
                new_layers.append(g1['layers'][i]); new_acts.append(g1['acts'][i])
        elif i < len(g2['layers']):
            if random.random() < 0.5:
                new_layers.append(g2['layers'][i]); new_acts.append(g2['acts'][i])
                
    if len(new_layers) == 0:
        new_layers = [random.choice([g1['layers'][0], g2['layers'][0]])]
        new_acts = [random.choice([g1['acts'][0], g2['acts'][0]])]
    return {'layers': new_layers, 'acts': new_acts}

def genome_to_str(g):
    return f"layers={g['layers']} acts={g['acts']}"

def transfer_weights(new_model, old_model, alpha=0.7):
    global TRANSFER_ERRORS
    if new_model.use_cnn_stem and old_model.use_cnn_stem:
        try: new_model.stem.load_state_dict(old_model.stem.state_dict())
        except Exception as e: TRANSFER_ERRORS += 1; print(f"⚠️ stem transfer: {e}")
    elif not new_model.use_cnn_stem and not old_model.use_cnn_stem:
        try: new_model.stem.load_state_dict(old_model.stem.state_dict())
        except Exception as e: TRANSFER_ERRORS += 1; print(f"⚠️ stem transfer: {e}")
        
    for new_layer, old_layer in zip(new_model.net, old_model.net):
        if type(new_layer) != type(old_layer): continue
        if isinstance(new_layer, nn.Linear):
            if new_layer.weight.shape == old_layer.weight.shape:
                new_layer.weight.data = (1 - alpha) * new_layer.weight.data + alpha * old_layer.weight.data
                if new_layer.bias is not None and old_layer.bias is not None:
                    new_layer.bias.data = (1 - alpha) * new_layer.bias.data + alpha * old_layer.bias.data
        elif isinstance(new_layer, nn.BatchNorm1d):
            if new_layer.weight.shape == old_layer.weight.shape:
                new_layer.weight.data = (1 - alpha) * new_layer.weight.data + alpha * old_layer.weight.data
                new_layer.bias.data = (1 - alpha) * new_layer.bias.data + alpha * old_layer.bias.data
                new_layer.running_mean.data = (1 - alpha) * new_layer.running_mean.data + alpha * old_layer.running_mean.data
                new_layer.running_var.data = (1 - alpha) * new_layer.running_var.data + alpha * old_layer.running_var.data

# ============================================================
# 🚀 Training and Evaluation
# ============================================================
def _prep_tensor(x, is_cnn):
    if is_cnn and USE_CHANNELS_LAST and x.ndim == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x

def train_model(model, train_loader, val_loader, opt_cls, opt_kwargs, max_batches,
                crit, stop_event=None, quick_prune=False, out_dim=10):
    if USE_CHANNELS_LAST and model.use_cnn_stem:
        model = model.to(memory_format=torch.channels_last)
    opt = opt_cls(model.parameters(), **opt_kwargs)
    model.train()
    n = 0
    check_at = max(1, int(max_batches * 0.25)) if quick_prune else max_batches

    while n < max_batches:
        for x, y in train_loader:
            if stop_event is not None and stop_event.is_set():
                model.eval(); return True
            if n >= max_batches:
                model.eval(); return True
                
            if USE_CHANNELS_LAST and model.use_cnn_stem:
                x = x.contiguous(memory_format=torch.channels_last)

            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += 1

            if hasattr(opt, 'initial_lr'):
                progress = n / max_batches
                opt.lr = opt.initial_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

            if quick_prune and n == check_at:
                quick_acc = evaluate(model, val_loader, use_tta=False)
                model.train()
                threshold = 10.0 if out_dim > 2 else 55.0
                if quick_acc < threshold:
                    model.eval(); return False
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
            out = (out[:n] + out[n:2*n] + out[2*n:3*n] + out[3*n:]).div_(4)
        else:
            x = _prep_tensor(x, is_cnn)
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ============================================================
# 🧬 Phases
# ============================================================
def eval_phase1_worker(genome, state_dict, in_dim, out_dim, cfg, tl, vl):
    global LOAD_STATE_ERRORS
    model = DynamicNet(genome, in_dim, out_dim, cfg['use_cnn_stem'])
    if state_dict is not None:
        try: model.load_state_dict(state_dict)
        except Exception as e: LOAD_STATE_ERRORS += 1; print(f"⚠️ load state: {e}")

    smoothing = 0.2 if cfg['use_cnn_stem'] else 0.1
    crit = nn.CrossEntropyLoss(label_smoothing=smoothing)

    completed = train_model(model, tl, vl, DeltaRuleOptimizer, {'lr': 0.005},
                            cfg['batches1'], crit, quick_prune=True, out_dim=out_dim)
    if not completed:
        return 0.0, 0.0, model.state_dict()

    val_acc = evaluate(model, vl, use_tta=False)
    return val_acc, val_acc, model.state_dict()

def phase1(train_loader, val_loader, cfg, in_dim, out_dim, seed):
    set_seed(seed)
    pop = [{'genome': random_genome(cfg), 'state_dict': None, 'fitness': 0.0, 'val_acc': 0.0}
           for _ in range(cfg['pop1'])]
    best = {'acc': 0.0, 'genome': None}

    for gen in range(cfg['gen1']):
        print(f"🚀 [Phase 1 | Gen {gen+1}/{cfg['gen1']}] Evaluating {cfg['pop1']} models...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for i in range(len(pop)):
                futures[executor.submit(eval_phase1_worker, pop[i]['genome'], pop[i]['state_dict'],
                                        in_dim, out_dim, cfg, train_loader, val_loader)] = i
            for future in as_completed(futures):
                i = futures[future]
                val_acc, score, new_state = future.result()
                pop[i]['fitness'] = score; pop[i]['val_acc'] = val_acc; pop[i]['state_dict'] = new_state
                if val_acc > best['acc']:
                    best = {'acc': val_acc, 'genome': copy.deepcopy(pop[i]['genome'])}
                    print(f"  💾 Better Architecture: {genome_to_str(pop[i]['genome'])} | Val: {val_acc:.2f}%")

        pop = sorted(pop, key=lambda x: x['fitness'], reverse=True)
        elites = pop[:max(2, cfg['pop1']//5)]
        new_pop = copy.deepcopy(elites)
        while len(new_pop) < cfg['pop1']:
            p1, p2 = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
            child_g = crossover_genome(p1['genome'], p2['genome'])
            if random.random() < 0.4: child_g = mutate_genome(child_g, cfg)
            child_model = DynamicNet(child_g, in_dim, out_dim, cfg['use_cnn_stem'])
            if p1['state_dict'] is not None:
                parent_model = DynamicNet(p1['genome'], in_dim, out_dim, cfg['use_cnn_stem'])
                try:
                    parent_model.load_state_dict(p1['state_dict'])
                    transfer_weights(child_model, parent_model, alpha=0.7)
                except Exception as e: print(f"⚠️ transfer: {e}")
            new_pop.append({'genome': child_g, 'state_dict': child_model.state_dict(),
                            'fitness': 0.0, 'val_acc': 0.0})
        pop = new_pop
    return best

def eval_phase2_worker(coeffs, best_genome, in_dim, out_dim, cfg, tl, vl):
    model = DynamicNet(best_genome, in_dim, out_dim, cfg['use_cnn_stem'])
    smoothing = 0.2 if cfg['use_cnn_stem'] else 0.1
    crit = nn.CrossEntropyLoss(label_smoothing=smoothing)
    train_model(model, tl, vl, MetaOptimizer, {'coeffs': coeffs},
                cfg['batches2'], crit)
    return evaluate(model, vl, use_tta=False), model.state_dict()

def phase2(best_genome, train_loader, val_loader, cfg, in_dim, out_dim, seed):
    set_seed(seed + 7777)
    pop = [random_meta_coeffs() for _ in range(cfg['pop2'])]
    best = {'acc': 0.0, 'coeffs': None, 'state_dict': None}

    for gen in range(cfg['gen2']):
        print(f"🚀 [Phase 2 | Gen {gen+1}/{cfg['gen2']}] Evaluating {cfg['pop2']} coefficients...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for i, c in enumerate(pop):
                futures[executor.submit(eval_phase2_worker, c, best_genome,
                                        in_dim, out_dim, cfg, train_loader, val_loader)] = i
            fitness = [0.0] * len(pop)
            for future in as_completed(futures):
                i = futures[future]
                val_acc, state_dict = future.result(); fitness[i] = val_acc
                if val_acc > best['acc']:
                    best = {'acc': val_acc, 'coeffs': copy.deepcopy(pop[i]), 'state_dict': state_dict}
                    print(f"  💾 Better Coeffs: {coeffs_to_str(pop[i])} | Val: {val_acc:.2f}%")

        ranked = [c for _, c in sorted(zip(fitness, pop), key=lambda p: p[0], reverse=True)]
        elites = ranked[:max(2, cfg['pop2']//5)]
        new_pop = copy.deepcopy(elites)
        while len(new_pop) < cfg['pop2']:
            p1, p2 = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
            child = crossover_meta_coeffs(p1, p2)
            if random.random() < 0.4: child = mutate_meta_coeffs(child)
            new_pop.append(child)
        pop = new_pop
    return best

# ============================================================
# 📥 Dataset Loaders
# ============================================================
def load_cifar10(cfg, seed):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])

    train_full = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_train)
    eval_full  = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_eval)
    test_full  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_eval)

    y = np.array(train_full.targets)
    pool_idx, _ = train_test_split(np.arange(len(y)), train_size=cfg['n_train']+cfg['n_val'],
                                   stratify=y, random_state=seed)
    tr_idx, va_idx = train_test_split(pool_idx, train_size=cfg['n_train'],
                                      stratify=y[pool_idx], random_state=seed)
    y_test = np.array(test_full.targets)
    te_idx, _ = train_test_split(np.arange(len(y_test)), train_size=cfg['n_test'],
                                 stratify=y_test, random_state=seed)

    print("⏳ Precomputing val/test tensors...")
    with torch.no_grad():
        val_x = torch.stack([eval_full[i][0] for i in va_idx.tolist()])
        val_y = torch.tensor([eval_full[i][1] for i in va_idx.tolist()])
        test_x = torch.stack([test_full[i][0] for i in te_idx.tolist()])
        test_y = torch.tensor([test_full[i][1] for i in te_idx.tolist()])

    tl = DataLoader(Subset(train_full, tr_idx.tolist()), batch_size=cfg['batch'], shuffle=True)
    vl = DataLoader(TensorDataset(val_x, val_y), batch_size=512, shuffle=False)
    tsl = DataLoader(TensorDataset(test_x, test_y), batch_size=512, shuffle=False)
    return tl, vl, tsl, 3*32*32, 10, True

def load_mnist1d(cfg, seed):
    file_path = os.path.join(DATA_MNIST_DIR, 'mnist1d_data.pkl')
    if not os.path.exists(file_path): raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'rb') as f: data = pickle.load(f)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    in_dim = X_train.shape[1]; out_dim = len(np.unique(y_train))
    X_train = (X_train - X_train.mean(dim=0)) / (X_train.std(dim=0) + 1e-8)
    X_test = (X_test - X_test.mean(dim=0)) / (X_test.std(dim=0) + 1e-8)
    X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2,
                                              stratify=y_train, random_state=seed)
    tl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg['batch'], shuffle=True)
    vl = DataLoader(TensorDataset(X_va, y_va), batch_size=512, shuffle=False)
    tsl = DataLoader(TensorDataset(X_test, y_test), batch_size=512, shuffle=False)
    return tl, vl, tsl, in_dim, out_dim, False

def load_bio(cfg, seed):
    file_path = os.path.join(DATA_BIO_DIR, 'promoters.data')
    if not os.path.exists(file_path): raise FileNotFoundError(f"File {file_path} not found.")
    X_raw, y_raw = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                label = 1 if parts[0].strip() == '+' else 0
                seq = parts[2].strip().upper()
                seq = ''.join([c for c in seq if c in 'ACGT'])
                if len(seq) > 0: X_raw.append(seq); y_raw.append(label)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    max_len = max(len(s) for s in X_raw)
    X_encoded = np.zeros((len(X_raw), max_len * 4), dtype=np.float32)
    for i, seq in enumerate(X_raw):
        for j, char in enumerate(seq):
            if char in mapping: X_encoded[i, j * 4 + mapping[char]] = 1.0
    X = torch.tensor(X_encoded, dtype=torch.float32); y = torch.tensor(y_raw, dtype=torch.long)
    in_dim = X.shape[1]; out_dim = 2
    total_len = len(X); n_val = max(5, int(total_len * 0.15)); n_test = max(5, int(total_len * 0.15))
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=(n_val+n_test)/total_len,
                                                  stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_temp, y_temp, test_size=0.5,
                                              stratify=y_temp, random_state=seed)
    tl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg['batch'], shuffle=True)
    vl = DataLoader(TensorDataset(X_va, y_va), batch_size=512, shuffle=False)
    tsl = DataLoader(TensorDataset(X_te, y_te), batch_size=512, shuffle=False)
    return tl, vl, tsl, in_dim, out_dim, False

def load_speech_commands(cfg, seed):
    if not AUDIO_AVAILABLE: raise ImportError("torchaudio is not installed.")
    archive_path = os.path.join(DATA_AUDIO_DIR, 'SpeechCommands', 'speech_commands_v0.02.tar.gz')
    if not os.path.exists(archive_path):
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        urls = ["https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"]
        downloaded = False
        for url in urls:
            try:
                print(f"📥 Downloading from: {url}")
                import urllib.request
                urllib.request.urlretrieve(url, archive_path); downloaded = True; break
            except Exception as e:
                print(f"⚠️ Failed: {e}"); continue
        if not downloaded: raise RuntimeError("Download failed.")
        import tarfile
        try:
            with tarfile.open(archive_path, 'r:gz') as tar: tar.extractall(path=os.path.dirname(archive_path), filter='data')
        except TypeError:
            with tarfile.open(archive_path, 'r:gz') as tar: tar.extractall(path=os.path.dirname(archive_path))

    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=DATA_AUDIO_DIR, download=False, subset='training')
    test_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=DATA_AUDIO_DIR, download=False, subset='testing')
    mfcc_transform = T.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40})
    labels_map = {"yes":0, "no":1, "up":2, "down":3, "left":4, "right":5, "on":6, "off":7, "stop":8, "go":9}

    try:
        train_indices = [i for i, path in enumerate(train_dataset._walker) if os.path.basename(os.path.dirname(path)) in labels_map]
        test_indices = [i for i, path in enumerate(test_dataset._walker) if os.path.basename(os.path.dirname(path)) in labels_map]
    except AttributeError:
        train_indices = [i for i in range(len(train_dataset)) if train_dataset[i][2] in labels_map]
        test_indices = [i for i in range(len(test_dataset)) if test_dataset[i][2] in labels_map]

    rng = np.random.RandomState(seed)
    selected_train = rng.permutation(train_indices)[:cfg.get('n_train', 4000) + cfg.get('n_val', 1000)]
    selected_test = rng.permutation(test_indices)[:cfg.get('n_test', 1000)]

    def process_data(dataset, indices):
        X, y = [], []
        for i in indices:
            waveform, sample_rate, label, *_ = dataset[i]
            mfcc = mfcc_transform(waveform).squeeze(0)
            if mfcc.shape[1] < 100: mfcc = torch.nn.functional.pad(mfcc, (0, 100 - mfcc.shape[1]))
            else: mfcc = mfcc[:, :100]
            X.append(mfcc.flatten()); y.append(labels_map[label])
        return torch.stack(X), torch.tensor(y, dtype=torch.long)

    X_train_raw, y_train_raw = process_data(train_dataset, selected_train)
    X_test, y_test = process_data(test_dataset, selected_test)
    in_dim = X_train_raw.shape[1]; out_dim = 10
    X_train_raw = (X_train_raw - X_train_raw.mean()) / (X_train_raw.std() + 1e-8)
    X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)
    X_tr, X_va, y_tr, y_va = train_test_split(X_train_raw, y_train_raw,
                                              test_size=cfg.get('n_val', 1000) / len(X_train_raw),
                                              stratify=y_train_raw, random_state=seed)
    tl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg['batch'], shuffle=True)
    vl = DataLoader(TensorDataset(X_va, y_va), batch_size=512, shuffle=False)
    tsl = DataLoader(TensorDataset(X_test, y_test), batch_size=512, shuffle=False)
    return tl, vl, tsl, in_dim, out_dim, False

# ============================================================
# 🏃 Main Execution
# ============================================================
def run_experiment(dataset_name, cfg, num_seeds):
    all_results = []
    
    for seed in range(num_seeds):
        print(f"\n>>> [{dataset_name}] Seed {seed} - Loading...")
        
        if dataset_name.startswith('CIFAR'):
            tl, vl, tsl, in_dim, out_dim, use_stem = load_cifar10(cfg, seed)
        elif dataset_name == 'MNIST-1D':
            tl, vl, tsl, in_dim, out_dim, use_stem = load_mnist1d(cfg, seed)
        elif dataset_name == 'Bio (Promoter)':
            tl, vl, tsl, in_dim, out_dim, use_stem = load_bio(cfg, seed)
        elif dataset_name == 'Speech Commands (Audio)':
            tl, vl, tsl, in_dim, out_dim, use_stem = load_speech_commands(cfg, seed)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue

        cfg['use_cnn_stem'] = use_stem; cfg['out_dim'] = out_dim
        print(f"✅ Data Loaded. train={len(tl.dataset)} | val={len(vl.dataset)} | test={len(tsl.dataset)}")

        start_time = time.time()

        # --- Phase 1 ---
        b1 = phase1(tl, vl, cfg, in_dim, out_dim, seed)
        
        # --- Baseline (Adam) ---
        print("📊 Evaluating Baseline (Adam)...")
        baseline_model = DynamicNet(b1['genome'], in_dim, out_dim, cfg['use_cnn_stem'])
        crit = nn.CrossEntropyLoss(label_smoothing=0.2 if cfg['use_cnn_stem'] else 0.1)
        train_model(baseline_model, tl, vl, torch.optim.Adam, {'lr': 0.001}, cfg['batches2'], crit)
        adam_test_acc = evaluate(baseline_model, tsl, use_tta=cfg['use_cnn_stem'])
        print(f"📊 Baseline (Adam): Test={adam_test_acc:.2f}%")

        # --- Phase 2 ---
        b2 = phase2(b1['genome'], tl, vl, cfg, in_dim, out_dim, seed)

        # --- Final Evaluation ---
        print(f"⏳ Final Evaluation on Test (with TTA)...")
        m = DynamicNet(b1['genome'], in_dim, out_dim, cfg['use_cnn_stem'])
        m.load_state_dict(b2['state_dict'])
        test_acc = evaluate(m, tsl, use_tta=cfg['use_cnn_stem'])

        elapsed = time.time() - start_time
        print(f"🎉 [{dataset_name} | Seed {seed}] Final Test: {test_acc:.2f}% | "
              f"Formula: {coeffs_to_str(b2['coeffs'])} | Time: {elapsed:.0f}s")

        all_results.append({
            'dataset': dataset_name, 'seed': seed, 'acc': test_acc,
            'genome': b1['genome'], 'coeffs': b2['coeffs'],
            'val_acc_phase1': b1['acc'], 'val_acc_phase2': b2['acc'],
            'adam_test_acc': adam_test_acc,
            'elapsed_seconds': elapsed
        })
        
    # Save results
    safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    out_file = os.path.join(RESULTS_DIR, f'{safe_name}_results.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Results saved to {out_file}")
    
    if len(all_results) > 1:
        accs = [r['acc'] for r in all_results]
        print(f"📊 [{dataset_name}] Mean Accuracy on {len(accs)} Seeds: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")

def parse_args():
    parser = argparse.ArgumentParser(description="Joint Evolutionary Discovery of Neural Topologies and Optimizers")
    parser.add_argument('--dataset', type=str, default=None, 
                        choices=['CIFAR-10 (Small)', 'CIFAR-10 (Medium)', 'CIFAR-10 (Large)', 
                                 'MNIST-1D', 'Bio (Promoter)', 'Speech Commands (Audio)'],
                        help="Name of the dataset to run.")
    parser.add_argument('--seeds', type=int, default=3, help="Number of random seeds.")
    parser.add_argument('--gen1', type=int, default=5, help="Generations for Phase 1.")
    parser.add_argument('--gen2', type=int, default=5, help="Generations for Phase 2.")
    return parser.parse_args()

def get_interactive_menu():
    """Interactive menu for reviewers to easily select configurations."""
    print("\n" + "="*60)
    print("   🧬 Joint Evolutionary Discovery Framework")
    print("="*60)
    print("Please select a dataset configuration to run:")
    print("  1. CIFAR-10 (Small)   [2,000 train images, max 5 layers]")
    print("  2. CIFAR-10 (Medium)  [10,000 train images, max 6 layers]")
    print("  3. CIFAR-10 (Large)   [20,000 train images, max 7 layers]")
    print("  4. MNIST-1D           [4,000 train images, max 6 layers]")
    print("  5. Bio (Promoter)     [86 train images, max 4 layers]")
    print("  6. Speech Commands    [4,000 train audio, max 6 layers]")
    print("-" * 60)
    
    choices = {
        '1': 'CIFAR-10 (Small)',
        '2': 'CIFAR-10 (Medium)',
        '3': 'CIFAR-10 (Large)',
        '4': 'MNIST-1D',
        '5': 'Bio (Promoter)',
        '6': 'Speech Commands (Audio)'
    }
    
    choice = input("Enter your choice (1-6) [Default: 2]: ").strip()
    dataset_name = choices.get(choice, 'CIFAR-10 (Medium)')
    
    seeds_str = input("Enter number of random seeds (Default: 3): ").strip()
    num_seeds = int(seeds_str) if seeds_str.isdigit() else 3
    
    gen_str = input("Enter generations for Phase 1 & 2 (Default: 5): ").strip()
    gen1 = gen2 = int(gen_str) if gen_str.isdigit() else 5
    
    return dataset_name, num_seeds, gen1, gen2

def main():
    args = parse_args()
    
    # If no command-line arguments are provided, show the interactive menu
    if args.dataset is None:
        dataset_name, num_seeds, gen1, gen2 = get_interactive_menu()
    else:
        dataset_name = args.dataset
        num_seeds = args.seeds
        gen1 = args.gen1
        gen2 = args.gen2
    
    # Default configurations (Overridden based on dataset)
    configs = {
        'MNIST-1D': {
            'pop1': 8, 'gen1': gen1, 'batches1': 300,
            'pop2': 8, 'gen2': gen2, 'batches2': 500,
            'batch': 64, 'n_train': 4000, 'n_val': 1000, 'n_test': 1000,
            'act_funcs': ['tanh'], 'size_pool': [128, 256, 512],
            'min_layers': 1, 'max_layers': 6 
        },
        'Bio (Promoter)': {
            'pop1': 8, 'gen1': gen1, 'batches1': 80,
            'pop2': 8, 'gen2': gen2, 'batches2': 100,
            'batch': 16, 'n_train': 86, 'n_val': 10, 'n_test': 10,
            'act_funcs': ['tanh'], 'size_pool': [16, 32, 64],
            'min_layers': 1, 'max_layers': 4
        },
        'CIFAR-10 (Small)': {
            'pop1': 12, 'gen1': gen1, 'batches1': 150,
            'pop2': 8, 'gen2': gen2, 'batches2': 150,
            'batch': 128, 'n_train': 2000, 'n_val': 500, 'n_test': 500,
            'act_funcs': ['tanh'], 'size_pool': [64, 128],
            'min_layers': 1, 'max_layers': 5
        },
        'CIFAR-10 (Medium)': {
            'pop1': 8, 'gen1': gen1, 'batches1': 200,
            'pop2': 8, 'gen2': gen2, 'batches2': 400,
            'batch': 128, 'n_train': 10000, 'n_val': 2000, 'n_test': 2000,
            'act_funcs': ['tanh'], 'size_pool': [128, 256],
            'min_layers': 1, 'max_layers': 6
        },
        'CIFAR-10 (Large)': {
            'pop1': 8, 'gen1': gen1, 'batches1': 400,
            'pop2': 8, 'gen2': gen2, 'batches2': 800,
            'batch': 256, 'n_train': 20000, 'n_val': 5000, 'n_test': 5000,
            'act_funcs': ['tanh'], 'size_pool': [256, 512],
            'min_layers': 1, 'max_layers': 7
        },
        'Speech Commands (Audio)': {
            'pop1': 8, 'gen1': gen1, 'batches1': 200,
            'pop2': 8, 'gen2': gen2, 'batches2': 300,
            'batch': 64, 'n_train': 4000, 'n_val': 1000, 'n_test': 1000,
            'act_funcs': ['tanh'], 'size_pool': [64, 128, 256],
            'min_layers': 1, 'max_layers': 6
        }
    }
    
    cfg = configs[dataset_name]
    run_experiment(dataset_name, cfg, num_seeds)

if __name__ == '__main__':
    main()

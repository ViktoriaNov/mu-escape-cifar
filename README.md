# MU Escape — eksperymenty 

Kod źródłowy do reprodukcji wyników opisanych w pracy magisterskiej  
*Wykorzystanie metod oduczenia maszynowego w celu uniknięcia „ostrych” minimów lokalnych w trakcie uczenia sieci neuronowych*

Autorka: **Viktoria Novorodskaia**

## Co zawiera repozytorium

| Element | Opis |
|---------|------|
| `thesis_code/` | Pakiet Python: pięć metod (ES, SGDR, SWA, NI, MU Escape), pomiar ostrości, wizualizacje |
| `results/` | Tabele i surowe wyniki z pracy (`tabela_finalna.csv`, `surowe_wyniki.json`, diagnostyka patience) |
| `notebook/` | Notatnik Jupyter do uruchamiania scenariuszy lokalnie |
| `scripts/` | Skrypt wsadowy diagnostyki patience (Windows) |

Architektura: **ResNet-18** (CIFAR 32×32). Zbiory: **CIFAR-10** i **CIFAR-100**. Protokół **multi-seed** (42, 123, 777).

## Wymagania

- Python 3.10+
- GPU NVIDIA z CUDA (zalecane; pełny protokół na CPU jest bardzo wolny)
- ~4 GB VRAM przy batch size 128

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Szybki start

Z katalogu głównego repozytorium (tam, gdzie leży folder `thesis_code/`):

```bash
# Test potoku (1 ziarno, ~1 h na GPU)
python -m thesis_code.experiments --quick

# Pełny protokół multi-seed (CIFAR-10)
python -m thesis_code.experiments

# CIFAR-100
python -m thesis_code.experiments --dataset CIFAR100

# Wznowienie po przerwaniu (pomija gotowe checkpointy)
python -m thesis_code.experiments --resume

# Diagnostyka patience (rozdz. eksperymentów)
python -m thesis_code.experiments --patience --dataset CIFAR10
```

Eksport metryk przebiegów (bez ponownego treningu):

```bash
python -m thesis_code.trajectory_analysis --dataset CIFAR10 --seed 777
```

## Struktura wyników

Po uruchomieniu powstają katalogi:

```
checkpoints/{CIFAR10|CIFAR100}/   # stany modelu (.pt)
results/{CIFAR10|CIFAR100}/       # CSV, JSON, PNG
```

W repozytorium dołączono **wyniki liczbowe** użyte w pracy. Checkpointy i wykresy PNG generuje się lokalnie (patrz `.gitignore`).

## Metody

1. **Early Stopping (ES)** — baseline, punkt startowy post-hoc  
2. **SGDR** — cosine warm restarts  
3. **SWA** — stochastic weight averaging  
4. **Noise Injection (NI)** — perturbacja wag + fine-tuning  
5. **MU Escape** — gradient ascent na podzbiorze + fine-tuning (metoda proponowana)

Hiperparametry: `thesis_code/config.py` (zgodne z tabelą parametrów w pracy).

## Notatnik

`notebook/thesis_sharp_minima_local.ipynb` — interfejs do uruchamiania pojedynczych scenariuszy i podglądu wyników. Wymaga ustawienia `PROJECT_ROOT` na katalog repozytorium.

## Licencja

Kod udostępniony wyłącznie w celach naukowych i reprodukcji wyników pracy dyplomowej.  
Szczegóły licencji — do uzupełnienia przez autorkę przed publikacją repozytorium.

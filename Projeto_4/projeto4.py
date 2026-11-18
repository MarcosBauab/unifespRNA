"""Projeto 4: previsão de temperatura com GRU e seleção temporal por validação."""

# %% Ambiente
import argparse
import copy
import hashlib
import io
import itertools
import json
import os
import platform
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/projeto4-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_URL = "https://gksmyth.github.io/ozdasl/oz/melbtemp.txt"
DATA_DESCRIPTION = "https://gksmyth.github.io/ozdasl/oz/melbtemp.html"


@dataclass(frozen=True)
class Config:
    windows: tuple = (7, 30)
    topologies: tuple = ((16,), (32, 16))
    learning_rates: tuple = (0.001, 0.0003)
    horizon: int = 1
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    seed: int = 42
    train_end: str = "1987-12-31"
    val_end: str = "1988-12-31"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment(device):
    return dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), torch=torch.__version__,
                numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                device=str(device), threads=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                strict_determinism=False)


# %% Dados e janelas
def load_data(raw_path=None):
    """Reconstrói datas conforme a documentação original, que exclui 29/02."""
    raw_path = Path(raw_path) if raw_path else ROOT / "data" / "melbtemp.tsv"
    if raw_path.exists():
        raw = raw_path.read_bytes()
    else:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            raw = response.read()
    source = pd.read_csv(io.BytesIO(raw), sep=r"\s+")
    if source.shape != (3650, 2) or list(source.columns) != ["Min", "Max"]:
        raise ValueError("A fonte mudou: eram esperadas 3650 linhas e colunas Min/Max.")
    if not np.isfinite(source.to_numpy(dtype=float)).all():
        raise ValueError("A base contém valores não numéricos ou ausentes.")
    dates = pd.date_range("1981-01-01", "1990-12-31", freq="D")
    observed_dates = dates[~((dates.month == 2) & (dates.day == 29))]
    observed = pd.Series(source["Min"].to_numpy(dtype=float), index=observed_dates)
    frame = observed.reindex(dates).to_frame("temperature")
    frame.index.name = "date"
    frame["observed"] = frame.temperature.notna()
    # Só usa o passado. Valores imputados nunca viram alvos.
    frame["input_temperature"] = frame.temperature.ffill()
    info = dict(source=DATA_URL, documentation=DATA_DESCRIPTION,
                ingested_sha256=hashlib.sha256(raw).hexdigest(),
                observations=len(source), calendar_days=len(frame),
                missing_dates=[str(d.date()) for d in frame.index[~frame.observed]],
                feature="Min", unit="graus Celsius")
    return frame, info


def fit_scaler(frame, train_end):
    values = frame.loc[:train_end, "temperature"].dropna().to_numpy()
    return dict(mean=float(values.mean()), std=float(values.std(ddof=0)),
                observed_training_days=len(values))


def make_windows(frame, scaler, window, horizon, max_window, split, train_end, val_end):
    """Entrada termina em t; alvo em t+k. Partição definida pelo alvo e origem."""
    if not 1 <= window <= max_window or horizon < 1:
        raise ValueError("Janela e horizonte devem ser positivos; window <= max_window.")
    values = frame.input_temperature.to_numpy(dtype=float)
    targets = frame.temperature.to_numpy(dtype=float)
    dates = frame.index
    x, y, target_indices, origins = [], [], [], []
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    for j in range(max_window + horizon - 1, len(frame)):
        origin = j - horizon
        if not frame.observed.iloc[j]:
            continue
        if split == "train":
            keep = dates[j] <= train_end
        elif split == "val":
            keep = train_end < dates[j] <= val_end and dates[origin] >= train_end
        elif split == "test":
            keep = dates[j] > val_end and dates[origin] >= val_end
        else:
            raise ValueError(f"Partição desconhecida: {split}")
        if keep:
            x.append(values[origin - window + 1:origin + 1])
            y.append(targets[j])
            target_indices.append(j)
            origins.append(origin)
    if not x:
        raise ValueError(f"Sem exemplos na partição {split}")
    x = ((np.asarray(x) - scaler["mean"]) / scaler["std"]).astype("float32")[..., None]
    y = ((np.asarray(y) - scaler["mean"]) / scaler["std"]).astype("float32")
    return dict(x=torch.from_numpy(x), y=torch.from_numpy(y),
                indices=np.asarray(target_indices), origins=np.asarray(origins))


def prepare(out, config, raw_path=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use uma pasta vazia para preservar resultados anteriores: {out}")
    if config.horizon < 1 or config.max_epochs < 1 or config.patience < 1:
        raise ValueError("Horizonte, épocas e paciência devem ser positivos.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir()
    (out / "historicos").mkdir()
    (out / "figuras").mkdir()
    frame, info = load_data(raw_path)
    scaler = fit_scaler(frame, config.train_end)
    meta = dict(config=asdict(config), data=info, scaler=scaler)
    save_json(out / "experimento.json", meta)
    frame.to_csv(out / "serie_diaria.csv")
    return frame, meta


# %% Rede e métricas
class TemperatureGRU(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layers = nn.ModuleList()
        input_size = 1
        for width in hidden:
            self.layers.append(nn.GRU(input_size, width, batch_first=True))
            input_size = width
        self.head = nn.Linear(input_size, 1)

    def forward(self, x):
        for layer in self.layers:
            # O estado é reiniciado em cada janela: treinamento stateless.
            x, _ = layer(x)
        return self.head(x[:, -1, :]).squeeze(-1)


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Predições incompatíveis ou não finitas.")
    residual = y - prediction
    mse = float(np.mean(residual ** 2))
    denominator = float(np.sum((y - y.mean()) ** 2))
    return dict(mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(abs(residual))),
                r2=1 - float(np.sum(residual ** 2)) / denominator if denominator else None)


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch.to(device)).cpu() for batch in x.split(512)]).numpy()


def evaluate(model, data, scaler, device):
    prediction = predict(model, data["x"], device) * scaler["std"] + scaler["mean"]
    target = data["y"].numpy() * scaler["std"] + scaler["mean"]
    return metrics(target, prediction)


def build_split(frame, meta, window, split):
    c = meta["config"]
    return make_windows(frame, meta["scaler"], window, c["horizon"], max(c["windows"]),
                        split, c["train_end"], c["val_end"])


# %% Busca por validação
def search(frame, meta, out, device):
    """Não cria nem avalia janelas de teste."""
    out = Path(out)
    if (out / "busca.csv").exists():
        raise FileExistsError("Busca já iniciada; escolha uma nova pasta de resultados.")
    c, scaler = meta["config"], meta["scaler"]
    rows = []
    for window, hidden, lr in itertools.product(c["windows"], c["topologies"], c["learning_rates"]):
        seed_all(c["seed"])
        train = build_split(frame, meta, window, "train")
        valid = build_split(frame, meta, window, "val")
        name = f"w{window}_h{'-'.join(map(str, hidden))}_lr{lr}"
        model = TemperatureGRU(hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
        loader = DataLoader(TensorDataset(train["x"], train["y"]), batch_size=c["batch_size"],
                            shuffle=True, generator=torch.Generator().manual_seed(c["seed"]))
        best, stale, history = float("inf"), 0, []
        start = time.perf_counter()
        for epoch in range(1, c["max_epochs"] + 1):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Custo não finito em {name}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
            # Treino e validação avaliados com os mesmos pesos ao final da época.
            tr = evaluate(model, train, scaler, device)
            va = evaluate(model, valid, scaler, device)
            history.append(dict(epoch=epoch, **{f"train_{k}": v for k, v in tr.items()},
                                **{f"val_{k}": v for k, v in va.items()}))
            pd.DataFrame(history).to_csv(out / "historicos" / f"{name}.csv", index=False)
            if va["mse"] < best:
                best, stale, best_epoch = va["mse"], 0, epoch
                checkpoint = dict(state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                  hidden=list(hidden), window=window, lr=lr, epoch=epoch,
                                  scaler=scaler, horizon=c["horizon"], seed=c["seed"])
                torch.save(checkpoint, out / "checkpoints" / f"{name}.pt")
                best_tr, best_va = tr.copy(), va.copy()
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                print(f"{name} | época {epoch} | RMSE val {va['rmse']:.4f} °C", flush=True)
            if stale >= c["patience"]:
                break
        rows.append(dict(name=name, window=window, hidden=json.dumps(list(hidden)), lr=lr,
                         parameters=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch,
                         epochs=epoch, seconds=time.perf_counter() - start,
                         train_mse=best_tr["mse"], val_mse=best_va["mse"],
                         val_rmse=best_va["rmse"], val_mae=best_va["mae"]))
        pd.DataFrame(rows).to_csv(out / "busca.csv", index=False)
        print(f"Finalizado {name}: melhor época {best_epoch}; RMSE val {best_va['rmse']:.4f}", flush=True)
    results = pd.DataFrame(rows).sort_values(["val_mse", "name"])
    winner = results.iloc[0].to_dict()
    # O vencedor é congelado antes de qualquer cálculo de teste.
    save_json(out / "selecao.json", winner)
    return results


# %% Avaliação final e referências simples
def final_evaluate(frame, meta, out, device):
    out = Path(out)
    if (out / "metricas_teste.csv").exists():
        raise FileExistsError("O teste deste experimento já foi avaliado e salvo.")
    c = meta["config"]
    search_table = pd.read_csv(out / "busca.csv")
    expected = len(c["windows"]) * len(c["topologies"]) * len(c["learning_rates"])
    if len(search_table) != expected:
        raise ValueError("Conclua toda a busca antes de avaliar o teste.")
    winner = json.loads((out / "selecao.json").read_text())
    best_name = search_table.sort_values(["val_mse", "name"]).iloc[0]["name"]
    if winner["name"] != best_name:
        raise ValueError("A seleção foi alterada após a busca.")
    checkpoint = torch.load(out / "checkpoints" / f"{best_name}.pt", map_location="cpu", weights_only=True)
    model = TemperatureGRU(checkpoint["hidden"]).to(device)
    model.load_state_dict(checkpoint["state"])
    test = build_split(frame, meta, checkpoint["window"], "test")
    scaler = meta["scaler"]
    prediction = predict(model, test["x"], device) * scaler["std"] + scaler["mean"]
    # O alvo original é usado para reportar métricas em unidades físicas.
    target = frame.temperature.iloc[test["indices"]].to_numpy()
    values = frame.input_temperature.to_numpy()
    # O mês da data prevista é conhecido; a média é ajustada só em 1981–1987.
    monthly = frame.loc[:c["train_end"], "temperature"].groupby(
        frame.loc[:c["train_end"]].index.month).mean()
    predictions = {
        "GRU": prediction,
        "Persistência": values[test["origins"]],
        "Média dos últimos 7 dias": np.array([values[i-6:i+1].mean() for i in test["origins"]]),
        "Média mensal do treino": monthly.loc[frame.index[test["indices"]].month].to_numpy(),
    }
    table = pd.DataFrame([dict(model=name, **metrics(target, pred)) for name, pred in predictions.items()])
    table.to_csv(out / "metricas_teste.csv", index=False)
    exported = pd.DataFrame(dict(date=frame.index[test["indices"]],
                                 origin=frame.index[test["origins"]], observed=target, **predictions))
    exported.to_csv(out / "predicoes_teste.csv", index=False)
    split_info = {}
    for split in ["train", "val", "test"]:
        data = build_split(frame, meta, checkpoint["window"], split)
        split_info[split] = dict(n=len(data["y"]), first=str(frame.index[data["indices"][0]].date()),
                                 last=str(frame.index[data["indices"][-1]].date()))
    save_json(out / "particoes.json", split_info)
    summary = dict(winner=winner, partitions=split_info, test=table.to_dict("records"),
                   note="Pesos do melhor checkpoint de validação; sem retreino em treino+validação.")
    save_json(out / "resumo.json", summary)
    return table


# %% Figuras
def make_figures(frame, meta, out):
    out = Path(out)
    figdir = out / "figuras"
    c = meta["config"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figdir / name, dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ["#287a66", "#d69d29", "#3e6ba5"]
    for (start, end, label), color in zip([
        ("1981", c["train_end"], "Treino"),
        (str((pd.Timestamp(c["train_end"]) + pd.Timedelta(days=1)).date()), c["val_end"], "Validação"),
        (str((pd.Timestamp(c["val_end"]) + pd.Timedelta(days=1)).date()), "1990-12-31", "Teste")], colors):
        s = frame.loc[start:end, "temperature"]
        ax.plot(s.index, s, lw=0.55, color=color, label=label)
    ax.set(ylabel="Temperatura mínima (°C)", xlabel="Data", title="Série histórica e partições cronológicas")
    ax.legend(ncol=3)
    save(fig, "01_serie.png")

    winner = json.loads((out / "selecao.json").read_text())
    history = pd.read_csv(out / "historicos" / f"{winner['name']}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, metric, label in zip(axes, ["mse", "mae"], ["MSE (°C²)", "MAE (°C)"]):
        ax.plot(history.epoch, history[f"train_{metric}"], label="Treino", color=colors[0])
        ax.plot(history.epoch, history[f"val_{metric}"], label="Validação", color=colors[1])
        ax.axvline(winner["best_epoch"], color="gray", linestyle="--", lw=1, label="Checkpoint escolhido")
        ax.set(xlabel="Época", ylabel=label)
        ax.legend(fontsize=8)
    save(fig, "02_aprendizagem.png")

    search_table = pd.read_csv(out / "busca.csv")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for row in search_table.itertuples():
        h = pd.read_csv(out / "historicos" / f"{row.name}.csv")
        ax.plot(h.epoch, h.val_rmse, lw=1, label=f"J={row.window}; h={row.hidden}; lr={row.lr}")
    ax.set(xlabel="Época", ylabel="RMSE de validação (°C)", title="Todas as configurações avaliadas")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "03_busca.png")

    predictions = pd.read_csv(out / "predicoes_teste.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5))
    for ax, subset, title in [(axes[0], predictions, "Teste completo"),
                               (axes[1], predictions.iloc[:90], "Primeiros 90 dias do teste")]:
        ax.plot(subset.date, subset.observed, label="Observado", lw=1, color="#303d49")
        ax.plot(subset.date, subset.GRU, label="GRU", lw=1, color="#c65340")
        if ax is axes[1]:
            ax.plot(subset.date, subset["Persistência"], label="Persistência", lw=0.7, alpha=0.65)
        ax.set(ylabel="Temperatura (°C)", title=title)
        ax.legend(fontsize=8, ncol=3)
    save(fig, "04_previsoes.png")

    residual = predictions.observed - predictions.GRU
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    axes[0].scatter(predictions.observed, predictions.GRU, s=9, alpha=0.4)
    limits = [min(predictions.observed.min(), predictions.GRU.min()),
              max(predictions.observed.max(), predictions.GRU.max())]
    axes[0].plot(limits, limits, "--", color="gray")
    axes[0].set(xlabel="Observado (°C)", ylabel="Previsto (°C)")
    axes[1].scatter(predictions.GRU, residual, s=9, alpha=0.4)
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set(xlabel="Previsto (°C)", ylabel="Resíduo: observado − previsto (°C)")
    axes[2].hist(residual, bins=25, color=colors[2], alpha=0.8)
    axes[2].set(xlabel="Resíduo (°C)", ylabel="Frequência")
    save(fig, "05_residuos.png")
    return sorted(figdir.glob("*.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "resultados")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "melbtemp.tsv")
    args = parser.parse_args()
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    frame, meta = prepare(args.out, config, args.data)
    save_json(args.out / "ambiente.json", environment(device))
    print("Dados preparados:", meta["data"], flush=True)
    search(frame, meta, args.out, device)
    print(final_evaluate(frame, meta, args.out, device).to_string(index=False), flush=True)
    make_figures(frame, meta, args.out)


if __name__ == "__main__":
    main()

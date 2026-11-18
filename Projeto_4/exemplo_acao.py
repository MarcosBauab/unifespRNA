"""Exemplo didático de previsão de uma ação FICTÍCIA cinco pregões à frente."""

# %% Dados fictícios e janelas
import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/projeto4-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def generate_prices(n=1200, seed=2026):
    """Passeio aleatório em log-preços; não representa uma empresa real."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0002, scale=0.008, size=n)
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"date": dates, "price": 100 * np.exp(np.cumsum(log_returns))})


def stock_windows(prices, window=30, horizon=5):
    """Cortes 70/15/15; a origem deve ser posterior ao ajuste/seleção."""
    values = prices.price.to_numpy()
    train_end, val_end = int(len(values) * 0.70) - 1, int(len(values) * 0.85) - 1
    mean, std = float(values[:train_end+1].mean()), float(values[:train_end+1].std())
    splits = {}
    for split in ["train", "val", "test"]:
        x, y, indices, origins = [], [], [], []
        for target in range(window + horizon - 1, len(values)):
            origin = target - horizon
            keep = (target <= train_end if split == "train" else
                    train_end < target <= val_end and origin >= train_end if split == "val" else
                    target > val_end and origin >= val_end)
            if keep:
                x.append(values[origin-window+1:origin+1])
                y.append(values[target])
                indices.append(target)
                origins.append(origin)
        splits[split] = dict(x=torch.tensor((np.array(x)-mean)/std, dtype=torch.float32).unsqueeze(-1),
                             y=torch.tensor((np.array(y)-mean)/std, dtype=torch.float32),
                             indices=np.array(indices), origins=np.array(origins))
    return splits, dict(mean=mean, std=std, train_end=train_end, val_end=val_end)


# %% Modelo e busca pequena
class StockGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1]).squeeze(-1)


def stock_predict(model, data):
    model.eval()
    with torch.no_grad():
        return model(data["x"]).numpy()


def stock_metrics(observed, predicted):
    error = observed - predicted
    return dict(mse=float(np.mean(error**2)), rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(abs(error))))


def run_stock_example(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Preserve a execução anterior e use uma pasta vazia: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    prices = generate_prices()
    prices.to_csv(out / "precos_FICTICIOS.csv", index=False)
    splits, scale = stock_windows(prices)
    rows, histories, winner = [], {}, None
    for lr in [0.001, 0.0003]:
        torch.manual_seed(42)
        model = StockGRU()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train = splits["train"]
        loader = DataLoader(TensorDataset(train["x"], train["y"]), batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, 61):
            model.train()
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Custo não finito")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            tr = stock_metrics(train["y"].numpy(), stock_predict(model, train))
            va = stock_metrics(splits["val"]["y"].numpy(), stock_predict(model, splits["val"]))
            history.append(dict(epoch=epoch, train_mse=tr["mse"] * scale["std"]**2,
                                val_mse=va["mse"] * scale["std"]**2))
            if va["mse"] < best:
                best, stale, best_epoch = va["mse"], 0, epoch
                state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= 8:
                break
        name = f"lr{lr}"
        pd.DataFrame(history).to_csv(out / f"historico_{name}.csv", index=False)
        torch.save(state, out / f"checkpoint_{name}.pt")
        row = dict(lr=lr, best_epoch=best_epoch, epochs=epoch,
                   val_rmse=float(np.sqrt(best) * scale["std"]))
        rows.append(row)
        histories[name] = history
        if winner is None or row["val_rmse"] < winner["val_rmse"]:
            winner = row.copy()
            winning_state = state
    pd.DataFrame(rows).to_csv(out / "busca.csv", index=False)
    (out / "selecao.json").write_text(json.dumps(winner, indent=2))
    # Só após congelar a seleção, consultamos os alvos de teste.
    model = StockGRU()
    model.load_state_dict(winning_state)
    test = splits["test"]
    observed = prices.price.to_numpy()[test["indices"]]
    gru = stock_predict(model, test) * scale["std"] + scale["mean"]
    persistence = prices.price.to_numpy()[test["origins"]]
    results = pd.DataFrame([dict(model=name, **stock_metrics(observed, pred))
                            for name, pred in [("GRU", gru), ("Persistência", persistence)]])
    results.to_csv(out / "metricas_teste.csv", index=False)
    pd.DataFrame(dict(target_date=prices.date.iloc[test["indices"]].to_numpy(),
                      origin_date=prices.date.iloc[test["origins"]].to_numpy(),
                      observed=observed, GRU=gru, persistence=persistence)).to_csv(
                          out / "predicoes_teste.csv", index=False)
    config = dict(fictitious=True, simulator="P_i=100*exp(soma(r_j)); r_j~N(0.0002,0.008²)",
                  data_seed=2026, training_seed=42, window=30, horizon=5, hidden=16,
                  learning_rates=[0.001, 0.0003], max_epochs=60, patience=8, batch_size=64,
                  scaler=scale, n={k: len(v["y"]) for k, v in splits.items()},
                  python=__import__("platform").python_version(), torch=torch.__version__,
                  numpy=np.__version__, pandas=pd.__version__, device="cpu",
                  note="Dias úteis sintéticos: fins de semana excluídos, sem calendário de feriados da bolsa.")
    (out / "experimento.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    history = pd.DataFrame(histories[f"lr{winner['lr']}"])
    axes[0].plot(history.epoch, history.train_mse, label="Treino")
    axes[0].plot(history.epoch, history.val_mse, label="Validação")
    axes[0].axvline(winner["best_epoch"], color="gray", linestyle="--")
    axes[0].set(xlabel="Época", ylabel="MSE (unidades monetárias²)", title="Ação FICTÍCIA: curva do vencedor")
    dates = prices.date.iloc[test["indices"]]
    axes[1].plot(dates, observed, label="Observado fictício", lw=1.4)
    axes[1].plot(dates, gru, label="GRU, k=5", lw=1)
    axes[1].plot(dates, persistence, label="Persistência, k=5", lw=1, alpha=0.75)
    axes[1].set(xlabel="Data fictícia do alvo", ylabel="Preço (unidades monetárias)",
                title="Previsões com origem móvel; pesos fixos")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "acao_ficticia.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("DADOS FICTÍCIOS. Configuração selecionada:", winner)
    print(results.to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "resultados_acao")
    args = parser.parse_args()
    run_stock_example(args.out)


if __name__ == "__main__":
    main()

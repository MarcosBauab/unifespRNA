"""Projeto 3: VAE no Fashion-MNIST, exportado de fashion_mnist_vae_web.ipynb.

Dependências: pip install torch torchvision numpy scikit-learn pandas matplotlib
Execução: python3 fashion_mnist_vae.py
"""


# %% [markdown]
# # VAEs no Fashion-MNIST
# Execute uma seção por vez. Primeiro confira os dados e o teste rápido; depois execute a busca. Ative GPU em Ambiente de execução → Alterar tipo de ambiente de execução.
# A seleção usa somente validação. Os rótulos servem para visualizar e estratificar, não entram no VAE.

# %%
import json, random, time, copy, platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados'
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
seed()
print('Dispositivo:', DEVICE)
(OUT/'ambiente.json').write_text(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':str(DEVICE)},indent=2))

# %% [markdown]
# ## 1. Dados
# Pixels em [0,1]; 50 mil imagens de treino, 10 mil de validação e 10 mil de teste.

# %%
base = FashionMNIST(str(ROOT / 'data'), train=True, download=True, transform=ToTensor())
test = FashionMNIST(str(ROOT / 'data'), train=False, download=True, transform=ToTensor())
tr, va = train_test_split(np.arange(len(base)), test_size=10000, random_state=42, stratify=base.targets.numpy())
assert not set(tr) & set(va)
np.savez(OUT/'particoes.npz', treino=tr, validacao=va)
train = DataLoader(Subset(base,tr), batch_size=256, shuffle=True, num_workers=0)
valid = DataLoader(Subset(base,va), batch_size=256, num_workers=0)
testloader = DataLoader(test, batch_size=256, num_workers=0)
classes = base.classes
fig, axes = plt.subplots(2,5,figsize=(12,5))
for i, ax in enumerate(axes.flat):
    x,y=base[i]; ax.imshow(x.squeeze(),cmap='gray'); ax.set_title(classes[y]); ax.axis('off')
plt.tight_layout(); plt.show()
print(len(tr),len(va),len(test))

# %% [markdown]
# ## 2. Modelo e custo
# O encoder retorna média e log-variância. A reparametrização permite propagar gradientes. O custo soma erro quadrático por imagem e KL; ambos são registrados separadamente. O decoder espelha o encoder.

# %%
class VAE(nn.Module):
    def __init__(self, hidden, latent):
        super().__init__()
        def stack(dims):
            layers=[]
            for a,b in zip(dims[:-1],dims[1:]): layers.extend([nn.Linear(a,b),nn.ReLU()])
            return nn.Sequential(*layers)
        self.encoder=stack([784]+list(hidden))
        self.mu=nn.Linear(hidden[-1],latent)
        self.logvar=nn.Linear(hidden[-1],latent)
        self.decoder=nn.Sequential(stack([latent]+list(reversed(hidden))),nn.Linear(hidden[0],784),nn.Sigmoid())
    def encode(self,x):
        h=self.encoder(x.flatten(1)); return self.mu(h),self.logvar(h)
    def forward(self,x):
        mu,lv=self.encode(x)
        z=mu+torch.exp(.5*lv)*torch.randn_like(mu)
        return self.decoder(z),mu,lv

def loss_parts(x,recon,mu,lv):
    rec=(recon-x.flatten(1)).square().sum(1)
    kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1)
    return torch.stack([rec+kl,rec,kl],1)

def evaluate(model,loader):
    model.eval(); total=np.zeros(3); n=0
    # Restaura o RNG de treino após avaliação e fixa o ruído da validação.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE.type=='cuda' else []):
        torch.manual_seed(123)
        with torch.no_grad():
            for x,_ in loader:
                x=x.to(DEVICE); parts=loss_parts(x,*model(x))
                total+=parts.sum(0).cpu().numpy(); n+=len(x)
    return total/n
seed(); probe=VAE([256,128],2).to(DEVICE)
x,_=next(iter(train)); x=x[:8].to(DEVICE)
parts=loss_parts(x,*probe(x)); assert parts.shape==(8,3) and torch.isfinite(parts).all()
parts[:,0].mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
print('Teste de dimensões e gradientes passou.')

# %% [markdown]
# ## 3. Busca independente para 2D e 8D
# Quatro configurações por espaço latente, Adam, até 20 épocas e paciência 4. Execute após conferir as seções anteriores. Se precisar reduzir para 10 épocas, faça isso antes da busca inteira, preservando o mesmo orçamento para todos. Checkpoints são salvos a cada melhora.

# %%
MAX_EPOCHS=20
rows=[]
for latent in [2,8]:
    for hidden in [[256,128],[512,256]]:
        for lr in [1e-3,3e-4]:
            seed(); model=VAE(hidden,latent).to(DEVICE)
            opt=torch.optim.Adam(model.parameters(),lr=lr)
            name=f'z{latent}_h{hidden[0]}_lr{lr}'
            best=float('inf'); stale=0; history=[]; start=time.time()
            for epoch in range(1,MAX_EPOCHS+1):
                model.train(); sums=np.zeros(3); n=0
                for x,_ in train:
                    x=x.to(DEVICE); opt.zero_grad()
                    parts=loss_parts(x,*model(x)); loss=parts[:,0].mean()
                    if not torch.isfinite(loss): raise RuntimeError('Custo não finito')
                    loss.backward(); opt.step()
                    sums+=parts.detach().sum(0).cpu().numpy(); n+=len(x)
                val=evaluate(model,valid)
                record={'epoch':epoch}
                for prefix,values in [('train',sums/n),('val',val)]:
                    record.update({f'{prefix}_{key}':float(v) for key,v in zip(['total','recon','kl'],values)})
                history.append(record)
                pd.DataFrame(history).to_csv(OUT/f'{name}_historico.csv',index=False)
                print(name,epoch,'val:',round(val[0],3),flush=True)
                if val[0]<best:
                    best=float(val[0]); best_epoch=epoch; stale=0
                    torch.save({'state':model.state_dict(),'hidden':hidden,'latent':latent,'lr':lr,'seed':42,'max_epochs':MAX_EPOCHS},OUT/f'{name}.pt')
                else: stale+=1
                if stale>=4: break
            rows.append(dict(name=name,latent=latent,hidden=str(hidden),lr=lr,val_total=best,best_epoch=best_epoch,epochs=epoch,seconds=time.time()-start))
            pd.DataFrame(rows).to_csv(OUT/'busca.csv',index=False)
results=pd.DataFrame(rows)
print(results.to_string(index=False))

# %% [markdown]
# ## 4. Avaliação final, representações e geração
# PCA é ajustado nas médias latentes de treino e aplicado ao teste. A variância explicada refere-se ao treino. A representação 2D direta não sofre redução. Sobreposições no PCA não provam ausência de separação nas oito dimensões.

# %%
@torch.no_grad()
def embeddings(model,loader):
    model.eval(); zs=[]; ys=[]
    for x,y in loader:
        mu,_=model.encode(x.to(DEVICE)); zs.append(mu.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(zs),np.concatenate(ys)
summary=[]
for latent in [2,8]:
    winner=results[results.latent==latent].sort_values('val_total').iloc[0]
    ck=torch.load(OUT/f'{winner["name"]}.pt',map_location=DEVICE,weights_only=True)
    model=VAE(ck['hidden'],latent).to(DEVICE); model.load_state_dict(ck['state']); model.eval()
    metrics=evaluate(model,testloader)
    summary.append(dict(latent=latent,model=winner['name'],total=metrics[0],recon=metrics[1],kl=metrics[2]))
    hist=pd.read_csv(OUT/f'{winner["name"]}_historico.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,['total','recon','kl']):
        ax.plot(hist.epoch,hist[f'train_{key}'],label='Treino'); ax.plot(hist.epoch,hist[f'val_{key}'],label='Validação'); ax.set_title(key); ax.set_xlabel('Época'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/f'curvas_{latent}.png',dpi=180); plt.show()
    z,y=embeddings(model,testloader)
    np.savez(OUT/f'latentes_{latent}.npz',mu=z,labels=y)
    title=f'Espaço latente {latent}D'
    if latent==8:
        ztrain,_=embeddings(model,DataLoader(Subset(base,tr),batch_size=256))
        pca=PCA(n_components=2).fit(ztrain); coords=pca.transform(z)
        variance=pca.explained_variance_ratio_
        (OUT/'pca.json').write_text(json.dumps({'ratio':variance.tolist(),'sum':float(variance.sum())},indent=2))
        title+=f' — PCA: {variance.sum():.1%} da variância de treino'
    else: coords=z
    fig,ax=plt.subplots(figsize=(9,7))
    for label in range(10):
        mask=y==label; ax.scatter(coords[mask,0],coords[mask,1],s=5,alpha=.35,label=classes[label])
    ax.set_title(title); ax.set_xlabel('PC1' if latent==8 else 'μ1'); ax.set_ylabel('PC2' if latent==8 else 'μ2'); ax.legend(markerscale=3,fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/f'latentes_{latent}.png',dpi=180); plt.show()
    with torch.no_grad():
        x,_=next(iter(testloader)); x=x[:10].to(DEVICE)
        mu,_=model.encode(x); recon=model.decoder(mu).cpu().reshape(-1,28,28)
        seed(); generated=model.decoder(torch.randn(10,latent,device=DEVICE)).cpu().reshape(-1,28,28)
    fig,axes=plt.subplots(3,10,figsize=(15,5))
    for row,images in enumerate([x.cpu().squeeze(1),recon,generated]):
        for col,img in enumerate(images): axes[row,col].imshow(img,cmap='gray',vmin=0,vmax=1); axes[row,col].axis('off')
    fig.suptitle(f'{latent}D: originais / reconstruções pela média / geração aleatória')
    fig.tight_layout(); fig.savefig(OUT/f'imagens_{latent}.png',dpi=180); plt.show()
pd.DataFrame(summary).to_csv(OUT/'teste.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
print('Resultados salvos em',OUT)

# %% [markdown]
# ## 5. Discussão
# Compare os custos e os termos de reconstrução e KL. Há agrupamentos? Quais classes se sobrepõem? O PCA preserva quanto da variância? Diferencie estrutura visual de separação comprovada. Registre a limitação de uma semente e da busca pequena. Não use o teste para escolher hiperparâmetros.

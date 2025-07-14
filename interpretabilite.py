# Modules prédéfinis et tiers
from typing import Tuple

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#Modules créés pour le projet
from datapreparation import SOS, EOS, PAD
from modele import filename,TransformerConfig, LanguageModel,train_dataset, test_dataset, tokenizer

import webbrowser
from pathlib import Path

def print_color(text: str, values: list[float], text_color: str = "white", bck_color: int = 0, font_size: int | str = "18px", output_file: str = "colored_text.html") -> None:
    assert len(text) == len(values), f"lengths of text ({len(text)}) and values ({len(values)}) must be equal."
    assert bck_color in [0, 1, 2], f"bck_color ({bck_color}) must be in [0, 1, 2]"

    def shade_of_value(value, c):
        rgb_color = [
            lambda i: f'rgb({i}, 0, 0)',
            lambda i: f'rgb(0, {i}, 0)',
            lambda i: f'rgb(0, 0, {i})'
        ]
        intensity = int(value * 255)
        return rgb_color[c](intensity)

    def html_span(text_color, background_color, char):
        return f'<span style="font-size:{font_size}; color:{text_color}; background-color:{background_color}">{char}</span>'

    html_text = [html_span(text_color, shade_of_value(values[i], bck_color), char) for i, char in enumerate(text)]
    all_html_text = ''.join(html_text)
    
    # Wrap it in basic HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>Colored Text</title></head>
    <body style="font-family:monospace">{all_html_text}</body>
    </html>
    """

    path = Path(output_file).resolve()
    path.write_text(html_content, encoding="utf-8")
    webbrowser.open(f"file://{path}")

# Retirer la limite du nombre maximal de lignes affichées dans un tableau pandas
pd.set_option('display.max_rows', None)
# Configurer le thème de seaborn
sns.set_theme(style="whitegrid")
pth_rnd_gen_cpu = torch.manual_seed(42)

###CREATION DU SAE
class AutoEncoder(nn.Module):
    def __init__(self, act_size: int, num_features: int, l1_coeff: float) -> None:
        super().__init__()

        self.l1_coeff = l1_coeff
        self.num_features = num_features

        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(act_size, num_features))) 
        self.b_enc = nn.Parameter(torch.zeros(num_features))

        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(num_features, act_size))) 
        self.b_dec = nn.Parameter(torch.zeros(act_size)) 

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec
        return F.relu(x_cent @ self.W_enc + self.b_enc) # calcul des activations des concepts

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_dec + self.b_dec # calcul de la reconstruction

    def reconstruct_loss(self, x: torch.Tensor, acts: torch.Tensor, x_reconstruct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0) # loss de reconstruction MSE
        l1_loss = self.l1_coeff * (acts.float().abs().sum()) # penalité L1 sur les activations des concepts
        return l1_loss, l2_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] :
        """
        Args :
            x : input (B, S, act_size = d_model) ou (B*S, act_size = d_model)
        """
        hidden_acts = self.encode(x) 
        x_reconstruct = self.decode(hidden_acts)

        l1_loss, l2_loss = self.reconstruct_loss(x, hidden_acts, x_reconstruct)
        loss = l2_loss + l1_loss # lasso loss

        return loss, x_reconstruct, hidden_acts, l2_loss, l1_loss

    # permet de stabiliser l'entraînement
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
        
###Il faut modifier la méthode de propagation `forward()` de la classe `AutoEncoder` afin de récupérer les activations de la
# couche cachée si le paramètre `act`vaut `True`. Pour cela, on va créer une classe `AutoEncoderForSAE` qui héritera de
# l'ensemble des éléments de la classe parent, et qui disposera de la méthode `forward()` avec des modifications à apporter,

class LanguageModelForSAE(LanguageModel):
    def __init__(self, model_config: TransformerConfig) -> None:
        super().__init__(model_config)

    def forward(self, tokens: torch.Tensor, act: bool = False, stop_at_layer: int = -1) -> torch.Tensor:
        """
        Args :
            - tokens (torch.Tensor) : input shaped (B, s, vocab_size) with s in [1; S]
            - act (bool) : return hidden activations if True
            - stop_at_layer (int) : the indice of the DecoderLayer module to take output
        """
        x = self.embedding(tokens)
        x = self.core(x, stop_at_layer=stop_at_layer)

        if act:
            return x #(B, S, d_model)

        logits = self.get_logits_(x)

        return logits #(B, S, vocab_size)
    
d_model = 32 # dimension du modèle
n_heads = 4 # nombre de têtes pour l'attention
n_layers = 1 # nombre de couches
dropout = 0.

epochs = 5
batch_size = 16
print_every = 100

config = TransformerConfig(
    vocab_size=tokenizer.vocabulary_size(),
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout,
    max_len=max(train_dataset.max_len, test_dataset.max_len) - 1  # Because X and y : sequence[:-1] and sequence[1:] in dataset
)

model = LanguageModelForSAE(model_config=config)
model.load_state_dict(torch.load(filename, weights_only=True))

act_size = config.d_model
num_features = 4 * config.d_model
sae = AutoEncoder(act_size=act_size, num_features=num_features, l1_coeff=3e-4)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

filename2="./weights/sae"

do_train=False
if do_train:
    optim = torch.optim.Adam(sae.parameters(), lr=3e-4)
    sae.train()

    for epoch in trange(epochs, desc="apprentissage"):
        train_loss = 0
        for X, y in tqdm(train_dataloader, total=len(train_dataloader), desc=f"epoch #{epoch+1}"):

            # on les fait passer dans le modèle et on récupère les activations de la couche cachée du modèle
            # (grâce à act=True)
            hidden_acts_transfo = model(X, act=True).view(-1, config.d_model) # (B*S, d_model)
            loss, _, _, _, _ = sae(hidden_acts_transfo)

            train_loss += loss

            loss.backward()
            # opération explicite de normalisation avant propagation dans la chaîne de calcul
            sae.make_decoder_weights_and_grad_unit_norm()
            optim.step()
            optim.zero_grad()

        train_loss /= len(train_dataloader)

        print(f"\tperte entrainement: {loss.item():.2f}")

    sae.eval()
    torch.save(sae.state_dict(), filename2)
else:
    sae.load_state_dict(torch.load(filename2,map_location=torch.device('cpu'), weights_only=True))    
    sae.eval()
    
    
######### frequence d'activations des features sur les données d'entrainement
def freq_activations_rate(
        model: LanguageModel,
        sae: AutoEncoder,
        dataloader: torch.utils.data.DataLoader
        ) -> torch.Tensor:
    """
    Args :
        - dataloader (torch.utils.data.Dataloader) : jeu de données

    Returns :
        - (torch.Tensor) le taux d'activation de chaque neurone interne du sae
        basé sur le nombre total d'éléments de séquence, par séquence, sur l'ensemble
        du jeu de données
    """
    acts_count = torch.zeros((num_features,))
    for X, _ in tqdm(dataloader, total=len(dataloader), desc="données de test"):
        hidden_acts_transfo = model(X, act=True).view(-1, config.d_model)
        _, _, features, _, _ = sae(hidden_acts_transfo) # (B*S, d_model)
        features = features.to("cpu")
        acts_count += ((features > 0).int()).sum(0)

    return acts_count / (len(dataloader.dataset) * dataloader.dataset.max_len)
def least_most_activated(n=20):
    freq_rate = freq_activations_rate(model, sae, train_dataloader)
    values, indices = freq_rate.topk(n)
    print(f"Les {n} neurones s'activant le plus fréquemment :")
    print(pd.DataFrame({"neurones": indices, "Freq (%)": values*100}))

    values, indices = freq_rate.topk(n, largest=False)
    print(f"Les {n} neurones s'activant le moins fréquemment :")
    print(pd.DataFrame({"neurones": indices, "Freq (%)": values*100}))
#least_most_activated(20)

def update_top_k(
        top_values: torch.Tensor,
        top_indices: torch.Tensor,
        new_values: torch.Tensor,
        new_indices: torch.Tensor,
        k: int) -> Tuple[: torch.Tensor, : torch.Tensor]:
    """
    Args:
        -
    """
    combined_values = torch.cat([top_values, new_values])
    combined_indices = torch.cat([top_indices, new_indices])
    new_top_values, topk_indices = torch.topk(combined_values, k)
    new_top_indices = combined_indices[topk_indices]

    return new_top_values, new_top_indices

topK = 2 # Nombre maximum de valeurs à préserver par série
N = 3 # Nombre d'élements distincts dans une série

# Initialisation d'un batch de 4 séries
test_batch = torch.tensor([
    # Elt 0, 1, 2
    [1, 2000, 30],      # serie 0
    [10, 20, 300],      # serie 1
    [1000, 2, 3],       # serie 2
    [100, 200, 3000],   # serie 3
])

# Recherche des topK max par élément distinct, sur les 4 séries du batch, et des indiques des séries
# dans lesquelles la valeur max est présente
top_values = torch.zeros((N, topK), dtype=torch.int32)
top_indices = torch.full((N, topK), -1, dtype=torch.long)
for n in range(N):
    dim_values = test_batch[:, n]
    dim_indices = n + torch.arange(test_batch.size(0))
    print(dim_values)
    top_values[n], top_indices[n] = \
            update_top_k(top_values[n], top_indices[n], dim_values, dim_indices, topK)

expected_top_values = torch.tensor([
    [1000,  100],   # topK valeurs max pour l'élément 0
    [2000,  200],   # topK valeurs max pour l'élément 1
    [3000,  300]    # topK valeurs max pour l'élément 2
    ])
expected_top_indices = torch.tensor([
    [2, 3],         # indices des séries correspondant aux topK de l'élément 0
    [1, 4],         # indices des séries correspondant aux topK de l'élément 1
    [5, 3]          # indices des séries correspondant aux topK de l'élément 2
    ])

print("top_values", top_values)
print("top_indices", top_indices)

assert torch.equal(expected_top_values,  top_values), f"Valeurs inattendues dans top_values"
assert torch.equal(expected_top_indices,  top_indices), f"Valeurs inattendues dans top_indices"
top_k = 20 # on garde les 20 exemples qui font s'activer le plus chaque neurone caché du Transformer

# Les plus petites valeurs possibles
top_values = torch.full((config.d_model, top_k), -float('inf'))
# indices à -1 (le dernier) du fait que les plus petites valeurs seront classées en dernier dans top_vales
top_indices = torch.full((config.d_model, top_k), -1, dtype=torch.long)

#Pour chaque élément du jeu de données d'entrainement
for X, _ in tqdm(train_dataloader, total=len(train_dataloader)):
    # par lot
    # récupération des activations cachées du Transformer
    hidden_acts_transfo = model(X, act=True) # (B, L, d_model)
    # transfert vers le "cpu"
    hidden_acts_transfo = hidden_acts_transfo.to("cpu")
    # récupération la valeur maximale d'activation de chaque neurone cachée dans le Transformer, par batch
    max_act = hidden_acts_transfo.max(dim=1).values # (B, d_model)

    # pour chaque neurone caché du Transformer
    for id_neuron in range(config.d_model):
        # récupérer l'ensemble des valeurs d'activation de ce neurone
        dim_values = max_act[:, id_neuron]
        # calcul des indices pour chaque valeur d'activation (rappel: une par élément de batch)
        dim_indices = id_neuron + torch.arange(train_dataloader.batch_size)

        # compilation des valeurs d'activations maximum selon l'indice du nom de commune dans le jeu de données d'entrainement
        top_values[id_neuron], top_indices[id_neuron] = \
            update_top_k(top_values[id_neuron], top_indices[id_neuron], dim_values, dim_indices, top_k)

max_activations, indices = top_values[:, 0].topk(k=25)
def interpretabilite_transfo():
    print("valeurs maximales : ", max_activations.tolist())
    print("indices des neurones correspondants : ", indices.tolist())
    max_activations_indices = top_indices[indices][0].tolist()
    print("indices des noms de communes dans le jeu d'entrainement", max_activations_indices)
    id_hidden_activation = indices[0]
    for i in top_indices[id_hidden_activation]:
        data_i, _ = train_dataset[i.item()]
        city_name = tokenizer.to_string(data_i.tolist())
        data_i = data_i.unsqueeze(0)
        hidden_acts_transfo = model(data_i, act=True) # (B, L, n_neurones)
        print_color(city_name, hidden_acts_transfo[0, :, id_hidden_activation].tolist()[:len(city_name)])

top_k = 20 # on garde les 20 exemples qui font s'activer le plus chaque neurone caché du Transformer

# Les plus petites valeurs possibles
top_values_sae = torch.full((sae.num_features, top_k), -float('inf'))
# indices à -1 (le dernier) du fait que les plus petites valeurs seront classées en dernier dans top_values_sae
top_indices_sae = torch.full((sae.num_features, top_k), -1, dtype=torch.long)

#Pour chaque élément du jeu de données d'entrainement
for X, _ in tqdm(train_dataloader, total=len(train_dataloader)):
    # par lot
    # récupération des activations cachées du Transformer
    hidden_acts_transfo = model(X, act=True) # (B, S, d_model)
    # récupération des activations de caractéristiques (features) correspondant
    _, _, features, _, _ = sae(hidden_acts_transfo)
    # récupération la valeur maximale d'activation de chaque caractéristique dans le SAE, par batch
    max_features = features.max(dim=1).values # (B, S, num_features)
    # transfert vers le "cpu"
    max_features = max_features.to("cpu")

     # pour chaque caractéristique gérée par le SAE
    for id_feature in range(sae.num_features):
        # récupérer l'ensemble des valeurs d'activation de cette caractéristique
        dim_values = max_features[:, id_feature]
        # calcul des indices pour chaque valeur d'activation (rappel: une par élément de batch)
        dim_indices = id_feature + torch.arange(train_dataloader.batch_size)

        # compilation des valeurs d'activations maximum selon l'indice du nom de commune dans le jeu de données d'entrainement
        top_values_sae[id_feature], top_indices_sae[id_feature] = \
            update_top_k(top_values_sae[id_feature], top_indices_sae[id_feature], dim_values, dim_indices, top_k)

max_activations_sae, indices_sae = top_values_sae[:, 0].topk(k=20)
def interpretabilite_sae():
    print("valeurs maximales : ", max_activations_sae.tolist())
    print("indices des caractéristiques correspondantes : ", indices_sae.tolist())
    max_activations_indices_sae = top_indices_sae[indices][0].tolist()
    print("indices des noms de communes dans le jeu d'entrainement", max_activations_indices_sae)
    idx_feature = indices_sae[5]
    for i in top_indices_sae[idx_feature]:
        data_i, _ = train_dataset[i.item()]
        city_name = tokenizer.to_string(data_i.tolist())
        data_i = data_i.unsqueeze(0)
        hidden_acts_transfo = model(data_i, act=True) # (B, S, n_features)
        _, _, features, _, _ = sae(hidden_acts_transfo)
        print_color(city_name, features[0, :, idx_feature].tolist()[:len(city_name)])


# Modules prédéfinis et tiers
import math
import datetime
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#Modules créés pour le projet
from datapreparation import get_datasets, SOS, EOS, PAD,CharTokenizer,pth_rnd_gen

@dataclass
class TransformerConfig:
    """
    """
    vocab_size: int
    d_model: int # D or d_model in comments
    n_layers: int
    n_heads: int
    max_len: int # maximum sequence length (for positional embedding)
    dropout: float = 0.1
    bias: bool = False
    norm_eps: float = 1e-5
    super_attn: bool = False # overwrites flash to False
    flash: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be a multiple of n_heads"

        self.d_head = self.d_model // self.n_heads
class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, dim: int, eps: float) -> None:
        """
        Args :
            dim (int) :
            eps (float) :
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
class SelfAttentionMultiHead(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """
        Args :
            config (TransformerConfig) :
        """
        super().__init__()

        self.config = config

        # key, query, value projections for all heads
        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False) # d_query = n_heads*d_head as in the Transformer paper
        self.key_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        if not config.flash:
            # compute the mask once and for all here
            # registrer treats it like a parameter (device, state_dict...) without training
            mask = torch.full((1, 1, config.max_len, config.max_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer('mask', mask)

        # LxL super attention params
        if config.super_attn:
            self.k_in_v_proj = nn.Linear(in_features=config.max_len, out_features=config.max_len, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args :
            x (torch.Tensor) : input shaped (B, S, d_model)
        """
        B, S, _ = x.size()

        Q = self.query_proj(x).view(B, S, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, S, d_query)
        K = self.key_proj(x).view(B, S, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, S, d_key)
        V = self.value_proj(x).view(B, S, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, S, d_head=d_value)

        if self.config.flash and not self.config.super_attn:
            attention = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True
                )
        else:
            QK_T = Q @ torch.transpose(K, 2, 3) # (B, n_heads, S, S)
            QK_T = QK_T + self.mask[:, :, :S, :S]

            attention_scores = torch.softmax(QK_T / math.sqrt(self.config.d_head), dim=3) # (B, n_heads, S, S)

            if self.config.super_attn:
                attention = self.attn_dropout(attention_scores) @ self.k_in_v_proj.weight @ V # (B, n_h, L, d_value=d_head)
            else:
                attention = self.attn_dropout(attention_scores) @ V # (B, n_h, S, d_value=d_head)

        attention = attention.transpose(1, 2) # (B, S, n_heafs, d_head)
        y = attention.contiguous().view(B, S, self.config.d_model) # n_heads * d_head = d_model

        y = self.resid_dropout(self.c_proj(y))

        return y # (B, S, d_model)
class MLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """
        Args :
            config (TransformerConfig) : configuration settings
        """
        super().__init__()

        self.fc_1 = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.fc_2 = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.fc_3 = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args :
            x (torch.Tensor) : input data  shaped (B, S, d_model)
        """
        x = self.dropout(self.fc_2(F.silu(self.fc_1(x)) * self.fc_3(x)))
        return x # (B, S, d_model)
class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """
        Args :
            config (TransformerConfig) :
        """
        super().__init__()

        self.config = config

        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.sa = SelfAttentionMultiHead(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
         Args :
            x (torch.Tensor) : input data shaped (B, S, d_model)
        """
        x = x + self.sa(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))

        return x # (B, S, d_model)
class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """
        """
        super().__init__()

        self.config = config

        # Positional Embedding
        self.PE = nn.Embedding(config.max_len, config.d_model)
        self.in_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])

    def forward(self, x: torch.Tensor, stop_at_layer: int = -1) -> torch.Tensor:
        """
        Args :
            x (torch.Tensor) : input data shaped (B, S, d_model)
            stop_at_layer (int) : return the ouput (activations) after the specified {layer}-th layer (1 -> n_layers)
        """
        if stop_at_layer < 0:
            stop_at_layer += len(self.layers) + 1
        elif stop_at_layer == 0:
            stop_at_layer = 1
        assert stop_at_layer <= len(self.layers), \
            f"stop_at_layer ({stop_at_layer}) should be in [{-len(self.layers)}, {len(self.layers)}]"
        _, S, _ = x.size()

        # Add positional embedding
        pos_emb = self.PE(torch.arange(0, S, dtype=torch.long, device=x.device))
        x = self.in_dropout(x + pos_emb)

        for i, layer in enumerate(self.layers):
            x = layer(x) # (B, S, d_model)

            if stop_at_layer == i+1:
                return x

        return x # (B, S, d_model)
class LanguageModel(nn.Module):
    def __init__(self, model_config: TransformerConfig) -> None:
        super().__init__()

        self.config = model_config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model, padding_idx=0) ### EXERCICE : remplacer None par les bonnes instructions

        self.core = Transformer(self.config) ### EXERCICE : remplacer None par les bonnes instructions
        self.out_norm = RMSNorm(self.config.d_model, self.config.norm_eps) ### EXERCICE : remplacer None par les bonnes instructions
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False) ### EXERCICE : remplacer None par les bonnes instructions
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)
        self._init_normal()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_normal(self):
        for pn, p in self.named_parameters():
            if pn.endswith('fc_3.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layers))

    def get_logits_(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out_norm(x)
        return self.lm_head(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args :
            tokens (torch.Tensor) : input shaped (B, s, vocab_size) with s in [1; S]
        """

        x = self.embedding(tokens)
        x = self.core(x)

        logits = self.get_logits_(x) 
        return logits #(B, S, vocab_size)
d_model = 32 # dimension du modèle
n_heads = 4 # nombre de têtes pour l'attention
n_layers = 1 # nombre de couches
dropout = 0.

lr = 0.00075  # leraning rate
batch_size = 2

epochs = 75
print_each = 1000
# Charger les données et tokenizer
train_dataset, test_dataset, tokenizer, _ = get_datasets("./villes.txt")



do_train = False        # mettre True pour entraîner

def train():
    global do_train
    do_train = True
    print("Entraînement du modèle...")
config = TransformerConfig(
    vocab_size=tokenizer.vocabulary_size(),
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout,
    max_len=max(train_dataset.max_len, test_dataset.max_len) - 1  # Car X et y : sequence[:-1] et sequence[1:] retourné par les "dataset“
    )
filename = f"./weights/model_{d_model}__{n_heads}_heads__{n_layers}_layers"
model = LanguageModel(config)

if do_train:
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in trange(epochs, desc="Entrainement"):
        train_loss = 0
        for X, y in tqdm(train_dataloader, total=len(train_dataloader), desc=f"epoch #{epoch+1:2d} / {epochs}"):

            logits = model(X) # (B, S, vocab_size)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1).long(),
                ignore_index=tokenizer.char_to_int[PAD]
                )
            train_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for j, (X, y) in enumerate(test_dataloader):
                logits = model(X) # (B, S, vocab_size)
                val_loss += F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1).long(),
                    ignore_index=tokenizer.char_to_int[PAD]
                    ).item()
            val_loss /= len(test_dataloader)

        print(f"\tperte entrainement: {loss.item():.2f} | perte de validation : {val_loss:.2f}")
        model.train()

    torch.save(model.state_dict(), filename)
else:
    model.load_state_dict(torch.load(filename, map_location=torch.device("cpu")))
    model.eval()

    

# Fonction pour échantillonner du texte à partir du modèle
def sample(model: LanguageModel, tokenizer: CharTokenizer, prompt: str = "", device="cpu", g = None) -> str:
    global do_train
    do_train=False
    idx = torch.tensor(
        [tokenizer.char_to_int[SOS]] + tokenizer(prompt),
        dtype=torch.int32,
        device=device
        ).unsqueeze(0)

    next_id = -1

    while next_id != tokenizer.char_to_int[EOS]:
        logits = model(idx) # (1, len_s, d_model)

        # calcul des probas pour chaque élément du vocabulaire
        probs = F.softmax(logits[:, -1, :], dim=-1)
        # tirage au sort en prenant en compte ces probas
        next_id = torch.multinomial(probs, num_samples=1, generator=g).item()
        # concaténation
        idx = torch.cat([idx, torch.tensor(next_id, device=device).view(1, 1)], dim=1)

        if idx.shape[1] > model.config.max_len:
            break

    return tokenizer.to_string(idx[0].tolist())




def freq_distribution(
        serie: pd.Series,
        xlabel: str = 'Longueur des chaînes de caractères',
        ylabel: str = 'Fréquence',
        title: str = 'Distribution de la longueur des chaînes de caractères',
        palette: str = "coolwarm"
        ) -> pd.Series:
    """
    Affichage du graphique (historigramme) de la distribution des fréquences des
    valeurs fournies dans la série, et retourne la distribution sous forme de 
    Args :
        len_serie (pd.Series) : séries de chaînes de caractères
        xlabel
        ylabel
        title
        palette
    Returns :
        (pd.Series) distribution des longueurs des chaines de carctères
    """
    # Calculer la longueur des chaînes de caractères dans la colonne "nom"
    lengths = serie.apply(len)

    # Afficher la distribution de la longueur des chaînes de caractères
    distribution = lengths.value_counts().sort_index()
    #print(length_distribution)

    # Afficher la distribution sous forme d'un histogramme
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=distribution.index,
        y=distribution.values,
        hue=distribution.values,
        palette=palette
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.xticks(rotation=90)
    plt.show()

    return lengths


def freq_char(serie: pd.Series) -> None:
    """
    """
    # Concaténer toutes les chaînes de caractères de la colonne "nom"
    all_chars = ''.join(serie)

    # Compter les occurrences de chaque caractère
    char_counts = Counter(all_chars)

    # Convertir le résultat en dataframe pour une meilleure lisibilité
    char_freq_df = pd.DataFrame(
        char_counts.items(), columns=['Caractère', 'Fréquence']
        ).sort_values(by='Fréquence', ascending=False)
    char_freq_df["Ratio Freq (%)"] = char_freq_df["Fréquence"] / char_freq_df["Fréquence"].sum() * 100

    print("Nombre de caractères distincts :", len(char_freq_df))

    # If not in a Jupyter notebook
    if print == None:
        print(char_freq_df)
    else:
        print(char_freq_df)


def rate_freq_by_position(
        serie: pd.Series,
        trunc_length: int = 10,
        topN: int = 15
        ) -> None:
    """
    Affiche sous forme de carte de chaleur les taux de fréquence des topN 
    caractères les plus fréquemment présents dans les trunc_length premières
    positions des chaînes de caractères présentes dans serie.
    Args :
        serie (pd.Series) :
        trunc_length (int) :
        topN (int) :
    """
    # Limiter la longueur des noms à trunc_length caractères
    truncated = serie.str[:trunc_length]

    # Initialiser un dictionnaire pour stocker les fréquences des caractères par position
    position_char_freq = {i: Counter() for i in range(trunc_length)}

    # Remplir le dictionnaire avec les fréquences des caractères par position
    for name in truncated:
        for i, char in enumerate(name):
            position_char_freq[i][char] += 1

    # Convertir le dictionnaire en dataframe pour une meilleure lisibilité
    position_char_freq_df = pd.DataFrame(position_char_freq).fillna(0).astype(int)

    # Limiter aux topN premiers caractères les plus fréquents
    top_chars = position_char_freq_df.sum(axis=1).sort_values(ascending=False).head(topN).index
    position_char_freq_df = position_char_freq_df.loc[top_chars]

    # Calculer le taux de présence par position
    position_char_rate_df = position_char_freq_df.div(position_char_freq_df.sum(axis=0), axis=1) * 100

    # Visualiser les taux de présence avec une carte de chaleur
    plt.figure(figsize=(12, 8))
    sns.heatmap(position_char_rate_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.xlabel('Position')
    plt.ylabel('Caractère')
    plt.title(f'Taux de présence des caractères par position (limité aux {trunc_length} premières positions)')
    plt.yticks(rotation=0)
    plt.show()


Default_Element_Separator = "-' "
def freq_composition_element(
        serie: pd.Series,
        element_separator: str = Default_Element_Separator,
        n_e: int = 15,
        n_p: int = 15
        ) -> None:
    all_elements = ' '.join(serie.str.replace(f"[{element_separator}]", " ", regex=True).values).split()

    # Compter les occurrences de chaque élément
    element_counts = Counter(all_elements)

    # Convertir le résultat en dataframe pour une meilleure lisibilité
    element_freq_df = pd.DataFrame(
        element_counts.items(),
        columns=['Élément', 'Fréquence']
        ).sort_values(by='Fréquence', ascending=False)

    print(f"Nombre total de composants distincts : {len(element_freq_df)}")
    print(f"dont {n_e} exemples :")
    print(element_freq_df.sample(n_e).sort_values(by='Fréquence', ascending=False))

    # Filtrer les éléments dont la fréquence est strictement supérieure à 1
    element_freq_sup_1_df = element_freq_df[element_freq_df['Fréquence'] > 1]

    # Afficher le nombre total d'éléments associés
    print(f"Nombre total de composants présents plus d'une fois : {len(element_freq_sup_1_df)}")
    print(f"dont les {n_p} premiers :")
    print(element_freq_sup_1_df.head(n_p))


def composition_distribution(
        serie: pd.Series,
        element_separator: str = Default_Element_Separator,
        xlabel: str = 'Nombre de composants',
        ylabel: str = 'Fréquence',
        title: str = 'Distribution de la fréquence du nombre de composants',
        palette: str ="coolwarm"
        ) -> None:
    # Calculer le nombre de composants pour chaque nom
    num_components = serie.apply(lambda x: len([comp for comp in x if comp in element_separator]) + 1)

    # Afficher la distribution du taux de fréquence du nombre de composants
    component_distribution = num_components.value_counts().sort_index()
    component_distribution_rate = component_distribution / len(serie) * 100
    print(title + " (en valeur) :")
    print(component_distribution.to_frame())

    # Afficher la distribution sous forme d'un histogramme
    plt.figure(figsize=(12, 8))
    sns.barplot(x=component_distribution_rate.index, y=component_distribution_rate.values, hue=component_distribution_rate.values, palette=palette)
    for i in range(len(component_distribution_rate)):
        plt.text(i, component_distribution_rate.values[i] + 0.5, f'{component_distribution_rate.values[i]:.3f}%', ha='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + " (en %)")
    #plt.xticks(rotation=90)
    plt.show()

###### ANALYSE DU MODÈLE ######
# Génération d'une liste de noms de commune
def analyse():
    communes = []
    N = 1000

    for _ in trange(N):
        communes.append(sample(model, tokenizer, prompt="", g=pth_rnd_gen))

    df = pd.DataFrame(communes, columns=["nom"])
    df["nom"].describe()
    print(df["nom"].sample(20))



    lengths = freq_distribution(df['nom'])
    print(lengths.describe())
    freq_char(df['nom'])
    rate_freq_by_position(df['nom'])
    freq_composition_element(df['nom'])
    composition_distribution(df['nom'])

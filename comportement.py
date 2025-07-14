# Modules prédéfinis et tiers
import seaborn as sns
import pandas as pd
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
#Modules créés pour le projet
from datapreparation import get_datasets, SOS, EOS, PAD, CityNameDataset
from modele import  TransformerConfig,sample, CharTokenizer,train_dataset,test_dataset,tokenizer,d_model,n_heads,n_layers, dropout, batch_size
from interpretabilite import LanguageModelForSAE , AutoEncoder 
# Retirer la limite du nombre maximal de lignes affichées dans un tableau pandas
pd.set_option('display.max_rows', None)
# Configurer le thème de seaborn
sns.set_theme(style="whitegrid")
pth_rnd_gen_cpu = torch.manual_seed(42)
config = TransformerConfig(
    vocab_size=tokenizer.vocabulary_size(),
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout,
    max_len=max(train_dataset.max_len, test_dataset.max_len) - 1  # Because X and y : sequence[:-1] and sequence[1:] in dataset
)

filename = "./weights/model_32__4_heads__1_layers" # A modifier selon le contexte
model = LanguageModelForSAE(config)
model.load_state_dict(torch.load(filename, map_location=torch.device("cpu")))
model.eval()


act_size = config.d_model
num_features = 4 * config.d_model
filename = "./weights/sae" # A modifier selon le contexte
sae= AutoEncoder(act_size=act_size, num_features=num_features, l1_coeff=3e-4)
sae.load_state_dict(
        torch.load(filename, map_location=torch.device('cpu'), weights_only=True)
        )


###diriger le comportement
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model.eval()
sae.eval()
#Récupération des valeurs maximum des activations de chaque caractéristique dans le SAE par rapport au jeu de données
# d'entrainement, afin de bénéficier d'un réfénrenciel pour y appliquer un facteur multiplicateur :
#min_values_sae = torch.full((sae.num_features, 1), +float('inf'))
max_values_sae = torch.full((sae.num_features, ), -float('inf'))

#Pour chaque élément du jeu de données d'entrainement
for X, _ in tqdm(train_dataloader, total=len(train_dataloader)):
    # récupération des activations cachées du Transformer
    hidden_acts_transfo = model(X, act=True) # (B, S, d_model)
    # récupération des activations de caractéristiques (features) correspondant
    _, _, features, _, _ = sae(hidden_acts_transfo) # (B, S, n_features)

    features = features.to("cpu")

    max_features, _ = features.max(dim=0)
    max_features, _ = max_features.max(dim=0)
    max_values_sae = torch.max(max_values_sae, max_features)
    
    #print(max_values_sae)
for i, v in enumerate(max_values_sae):
    print(f"{i:3d}: {v:.2f}", end="   " if (i+1)%11 else "\n")

####GENERATION CONTROLEE
###Il s'agit d'appliquer la méthode de contrôle consistant à modifier les valeurs d'activations des neurones cachés dans le SAE,
# en connaissant les concepts qu'ils représentent, selon ce que l'on en a interprété.


def steered_sample(
        model: LanguageModelForSAE,
        sae: AutoEncoder,
        tokenizer: CharTokenizer,
        steering_vector: torch.Tensor,
        prompt: str = "",
        max_values_sae: torch.Tensor = None,
        device="cpu",
        g = torch.Generator(),
        ) -> str:
    """
    Args:
        - model (LanguageModelForSAE) :
        - sae (AutoEncoder,) :
        - tokenizer (CharTokenizer) :
        - steering_vector (torch.Tensor) :
        - prompt (str = "") :
        - max_values_sae (torch.Tensor) :
        - device (str) :
        - g (torch.Generator) :
    """

    idx = torch.tensor(
        [tokenizer.char_to_int[SOS]] + tokenizer(prompt),
        dtype=torch.int32,
        device=device
        ).unsqueeze(0)
    next_id = -1

    while next_id != tokenizer.char_to_int[EOS]:
        # activations cachées dans le Transformer
        hidden_act = model(idx, act=True) # (1, l, d_model)

        # encodage et decodage du SAE
        features = sae.encode(hidden_act) # (1, l, num_features)
        act_reconstruct_1 = sae.decode(features) # (1, l, d_model) # reconstruction sans modification

        # decodage du SAE avec l'encodage (les caractéristiques) forcé
        features[:, :, steering_vector[:, 0]] = \
            max_values_sae[steering_vector[:, 0]] * steering_vector[:, 1].float() # forçage des concepts sur chaque lettre
        act_reconstruct_2 = sae.decode(features) # reconstruction avec modification

        # correction de l'erreur de reconstruction
        error = hidden_act - act_reconstruct_1
        final_act = act_reconstruct_2 + error

        # génération des logits
        logits = model.get_logits_(final_act)

        # calcul des probas pour chaque élément du vocabulaire
        probs = F.softmax(logits[:, -1, :], dim=-1)
        # tirage au sort en prenant en compte ces probas
        next_id = torch.multinomial(probs, num_samples=1, generator=g).item()
        # concaténation
        idx = torch.cat([idx, torch.tensor(next_id, device=device).view(1, 1)], dim=1)

        if idx.shape[1] > model.config.max_len:
            break

    return tokenizer.to_string(idx[0].tolist()) 


max_activation_index = max_values_sae.argmax().item()

steering_vector = torch.tensor([[max_activation_index, 1]], dtype=int, device="cpu")

for i in range(15):
    print(steered_sample(
        model,
        sae,
        tokenizer,
        prompt="ouled",
        steering_vector=steering_vector,
        max_values_sae=max_values_sae,
        g=pth_rnd_gen_cpu
        )
    )


#neutraliser ce neurone
steering_vector = torch.tensor([[max_activation_index, 0]], dtype=int, device="cpu")

for i in range(15):
    print(steered_sample(
        model,
        sae,
        tokenizer,
        prompt="ouled",
        steering_vector=steering_vector,
        max_values_sae=max_values_sae,
        g=pth_rnd_gen_cpu
        )
    )

#generation sans contrôle
    
for i in range(15):
    print(sample(
        model,
        tokenizer,
        prompt="ouled",
        g=pth_rnd_gen_cpu
        )
    )
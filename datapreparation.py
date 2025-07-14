# Modules prédéfinis et tiers
import random
from collections import Counter
from typing import Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import trange, tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Retirer la limite du nombre maximal de lignes affichées dans un tableau pandas
pd.set_option('display.max_rows', None) 
# Configurer le thème de seaborn
sns.set_theme(style="whitegrid")
# Paramétrer les graines aléatoires
pth_rnd_gen = torch.manual_seed(42)


df = pd.read_table("./villes.txt", header=None, names=["nom"])


def longeur():
    # Calculer la longueur des chaînes de caractères dans la colonne "nom"
    df['length'] = df['nom'].apply(len)
    # Afficher la distribution de la longueur des chaînes de caractères
    length_distribution = df['length'].value_counts().sort_index()
    #print(length_distribution)
    # Afficher la distribution sous forme d'un histogramme
    plt.figure(figsize=(12, 8))
    sns.barplot(x=length_distribution.index, y=length_distribution.values, hue=length_distribution.values, palette="coolwarm")
    plt.xlabel('Longueur des chaînes de caractères')
    plt.ylabel('Fréquence')
    plt.title('Distribution de la longueur des chaînes de caractères')
    plt.show()
    print(df['length'].describe())
def frequence():
    # Concaténer toutes les chaînes de caractères de la colonne "nom"
    all_chars = ''.join(df['nom'])
    # Compter les occurrences de chaque caractère
    char_counts = Counter(all_chars)
    # Convertir le résultat en dataframe pour une meilleure lisibilité
    char_freq_df = pd.DataFrame(
        char_counts.items(), columns=['Caractère', 'Fréquence']
        ).sort_values(by='Fréquence', ascending=False)
    char_freq_df["Ratio Freq (%)"] = char_freq_df["Fréquence"] / char_freq_df["Fréquence"].sum() * 100
    print("Nombre de caractères distincts :", len(char_freq_df))
    print(char_freq_df)
def frequence_10():
    # Limiter la longueur des noms à 10 caractères
    df['nom_limited'] = df['nom'].str[:10]

    # Initialiser un dictionnaire pour stocker les fréquences des caractères par position
    position_char_freq = {i: Counter() for i in range(10)}

    # Remplir le dictionnaire avec les fréquences des caractères par position
    for name in df['nom_limited']:
        for i, char in enumerate(name):
            position_char_freq[i][char] += 1

    # Convertir le dictionnaire en dataframe pour une meilleure lisibilité
    position_char_freq_df = pd.DataFrame(position_char_freq).fillna(0).astype(int)

    # Limiter aux 15 premiers caractères les plus fréquents
    top_chars = position_char_freq_df.sum(axis=1).sort_values(ascending=False).head(15).index
    position_char_freq_df = position_char_freq_df.loc[top_chars]

    # Calculer le taux de présence par position
    position_char_rate_df = position_char_freq_df.div(position_char_freq_df.sum(axis=0), axis=1) * 100

    # Visualiser les taux de présence avec une carte de chaleur
    plt.figure(figsize=(12, 8))
    sns.heatmap(position_char_rate_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.xlabel('Position')
    plt.ylabel('Caractère')
    plt.title('Taux de présence des caractères par position (limité aux 10 premières positions)')
    plt.yticks(rotation=0)
    plt.show()
def compose():
        # Calculer le nombre de composants pour chaque nom
    element_separator = "-' "

    df['num_components'] = df['nom'].apply(lambda x: len([comp for comp in x if comp in element_separator]) + 1)

    # Afficher la distribution du taux de fréquence du nombre de composants
    component_distribution = df['num_components'].value_counts().sort_index()
    component_distribution_rate = component_distribution / len(df) * 100
    print("Distribution de la fréquence du nombre de composants (en valeur) :")
    print(component_distribution.to_frame())

    # Afficher la distribution sous forme d'un histogramme
    plt.figure(figsize=(12, 8))
    sns.barplot(x=component_distribution_rate.index, y=component_distribution_rate.values, hue=component_distribution_rate.values, palette="coolwarm")
    for i in range(len(component_distribution_rate)):
        plt.text(i, component_distribution_rate.values[i] + 0.5, f'{component_distribution_rate.values[i]:.3f}%', ha='center')
    plt.xlabel('Nombre de composants')
    plt.ylabel('Fréquence')
    plt.title('Distribution de la fréquence du nombre de composants (en %)')
    plt.xticks(rotation=90)
    plt.show()


SOS = "<SOS>" # Start Of Sequence
EOS = "<EOS>" # End Of Sequence
PAD = "<PAD>" # Padding
class CharTokenizer():
    def __init__(self, corpus: list[str]) -> None:
        self.special_tokens =  [PAD, SOS, EOS] # PAD en premier pour indice 0

         # liste des différents caractères distincts
        chars = sorted(list(set(''.join(corpus))))
        # ajout des trois jetons de contrôle
        chars = self.special_tokens + chars
        # tables de correspondances
        self.char_to_int = {}
        self.int_to_char = {}

        # indexation des tables de correspondances
        for (c, i) in tqdm(
            zip(chars, range(len(chars))),
            desc="creating vocabulary",
            total=len(chars)
            ):
            self.char_to_int[c] = i
            self.int_to_char[i] = c

    def __call__(self, string: str) -> list[int]:
        return self.to_idx(string)

    def vocabulary_size(self) -> int:
        return len(self.char_to_int)

    def to_idx(self, sequence: str) -> list[int]:
        """
        Translate a sequence of chars to its conterparts of indexes in the vocabulary
        """
        return [self.char_to_int[c] for c in sequence]

    def to_tokens(self, sequence: list[int]) -> list[str]:
        """
        Translate a sequence of indexes to its conterparts of chars in the vocabulary
        """
        return [self.int_to_char[i] for i in sequence]

    def to_string(self, sequence: list[int]) -> str:
        """
        Return the string corresponding to the sequence of indexes in the vocabulary
        """
        return "".join([self.int_to_char[i] for i in sequence if i > 2])



class CityNameDataset(Dataset):
    def __init__(self, names: list[str], tokenizer: CharTokenizer) -> None:
        """
        Args:
            - names : collection of string
            - vocabulary : maps of "char to index" and "index to char" based on names
        """
        super().__init__()

        self.tokenizer = tokenizer

        # création des séquences encodées
        num_sequences = len(names)
        self.max_len = max([len(name) for name in names]) + 2 # <SOS> et <EOS>
        self.X = torch.zeros((num_sequences, self.max_len), dtype=torch.int32)
        for i, name in tqdm(enumerate(names), total=num_sequences, desc="creatind dataset"):
            # encodage de la séquence : "SOS s e q u e n c e EOS PAD PAD ... PAD"
            self.X[i] = torch.tensor(
                self.tokenizer([SOS]) +
                self.tokenizer(name) +
                self.tokenizer([EOS]) +
                self.tokenizer([PAD] * (self.max_len - len(name) - 2))
            )

    def __len__(self):
        """
        """
        return self.X.size(0)

    def __getitem__(self, idx: int):
        """
        """
        return self.X[idx, :-1], self.X[idx, 1:] #sans dernier puis sans premier


import random
from typing import Tuple

def get_datasets(
        filename: str,
        split_rate: float = 0.85
        ) -> Tuple[CityNameDataset, CityNameDataset, CharTokenizer, int]:
    """
    Return train and test datasets, and the max length in the processed string collection

    Args:
        - filename (str) : path and file name of string data
        - split_rate (float) : rate of the split of the train data, in [0.; 1.]

    Returns:
        - train_dataset, test_dataset, tokenizer, max_len
    """
    # chargement des données
    with open(filename, "r") as file:
        raws = file.read()
    names = raws.replace('\n', ',').split(',')

    # filtrage des noms trop courts et calcul du max_len
    names_ = []
    max_len = 0
    for n in names:
        if len(n) > 2:
            names_.append(n)
            max_len = max(max_len, len(n))

    # création du tokenizer
    tokenizer = CharTokenizer(names_)

    # mélange aléatoire des noms
    random.shuffle(names_)

    # split train/test
    n_split = int(split_rate * len(names_))
    train_dataset = CityNameDataset(names_[:n_split], tokenizer)
    test_dataset = CityNameDataset(names_[n_split:], tokenizer)

    return train_dataset, test_dataset, tokenizer, max_len

train_dataset, test_dataset, tokenizer, max_len = get_datasets("./villes.txt")
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
batch_X, batch_y = next(iter(train_dataloader))
print("Dimensions du batch", batch_X.size(), end="\n\n")

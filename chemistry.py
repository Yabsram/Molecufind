from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import Crippen 
import pandas as pd
import numpy as np
from admet_ai import ADMETModel

model = ADMETModel()
def test_admet_model():
    test_smiles = ["CCO"]
    predictions = model.predict(smiles=test_smiles)
    print(f"Number of properties predicted: {len(predictions.columns)}")
    print(f"\nAvailable properties:\n{list(predictions.columns)}")
    print(f"\nPredictions:\n{predictions}")
    return predictions


if __name__ == "__main__":
    test_admet_model()
    

def calculate_lipophilicity(smiles: str) -> float:
    if not isinstance(smiles, str) or not smiles.strip():
        return np.nan
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.nan
    try:
        return float(Crippen.MolLogP(m))
    except Exception:
        return np.nan
    
#Data loading and conversion to dataframe
def load_molecule_database(csv_path: str) -> pd.DataFrame:
    """Read CSV, clean data, calculate properties for each molecule"""
    df = pd.read_csv('original.csv', sep = ';', usecols = ['ChEMBL ID', 'Name', 'Synonyms', 'Molecular Weight','Polar Surface Area', 'Aromatic Rings', 'Heavy Atoms','Molecular Formula', 'Smiles'])
    df['Lipophilicity'] = df['Smiles'].apply(calculate_lipophilicity)
    df.dropna(how='any', inplace=True)
    print(f"Loaded {len(df)} molecules")
    print(df.head())  # Show first 5 rows
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    return df


def calculate_similarity_score(selected_properties: dict, property_row: dict) -> float:
    """Compare selected properties to a molecule's properties"""
    # Returns numerical similarity score (0-1 or 0-100)


def get_top_similar_molecules(
    dataframe: pd.DataFrame, selected_properties: dict, top_n: int = 30) -> pd.DataFrame:
    """Filter and rank molecules by similarity to selected properties"""
    # Apply calculate_similarity_score to each row
    # Sort by score
    # Return top N

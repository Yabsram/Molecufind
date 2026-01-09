from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import Crippen 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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

# Data loading and conversion to dataframe
def load_molecule_database(csv_path: str) -> pd.DataFrame:
    """Read CSV, clean data, calculate properties for each molecule"""
    df = pd.read_csv('original.csv', sep = ';', usecols = ['ChEMBL ID', 'Name', 'Synonyms', 'Molecular Weight','Polar Surface Area', 'Aromatic Rings', 'Heavy Atoms','Molecular Formula', 'Smiles'])
    df['Lipophilicity'] = df['Smiles'].apply(calculate_lipophilicity)
    print(f"Loaded {len(df)} molecules")
    print(df.head())  # Show first 5 rows
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    return df


def calculate_properties(smiles: str) -> dict:
    """Calculate molecular properties (solubility, toxicity, etc.)"""
    # Returns dict like {'soluble': True, 'toxic': False, 'mw': 234.5, ...}


def calculate_similarity_score(selected_properties, db_properties):
    """Compare selected properties to a molecule's properties"""
    # Molecular Weight, Polar Surface Area, Aromatic Rings, Heavy Atoms, Lipophilicity
    property_cols = ['Molecular Weight' , 'Polar Surface Area', 'Aromatic Rings', 'Heavy Atoms', 'Lipophilicity']

    # extract desired cols
    db_extracted_values = db_properties[property_cols]

    # ensure selected_properties is a df, make into df
    if not isinstance(selected_properties, pd.DataFrame):
        selected_properties = pd.DataFrame([selected_properties])
    selected_extracted = selected_properties[property_cols]

    # normalize data frames
    scaler = StandardScaler()
    db_normalized = scaler.fit_transform(db_extracted_values)

    # normalize selected_properties using same scaler
    selected_normalized = scaler.transform(selected_extracted)

    # calculate distances using euclidean distance
    distances = np.linalg.norm(db_normalized - selected_normalized, axis=1)

    # convert distance to show similarity scores using exponential decay
    # creates more separation between similar and different molecules
    similarity_scores = np.exp(-distances / 2)

    db_results = db_properties.copy()
    db_results['similarity_score'] = similarity_scores

    return db_results


def get_top_similar_molecules(db_properties, selected_properties, n):
    """Filter and rank molecules by similarity to selected properties"""
    # Apply calculate_similarity_score to each row
    results = calculate_similarity_score(selected_properties, db_properties)

    # Sort by score
    sorted_top_results = results.sort_values(by='similarity_score', ascending=False)

    # return top n results
    return sorted_top_results.head(n) 

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

import pandas as pd
from sklearn.preprocessing import StandardScaler

EXAMPLE_COMPOUNDS = [
    # smiles, substructure fingerprint
    "CCC(Cl)C(N)C1=CC=CC=C1",
    "CCC(Cl)C(F)C1=CC=CC=C1",
    "CCC(Cl)C(F)C1CCCCC1",
    "CCC(Cl)C(N)C1CCCCC1",
    "CCC(F)C(Cl)CC",
    "CCC(F)C(N)CC",
    "CCC(Cl)C(N)C1CCC2CCCCC2C1",
]

# Data loading & preparation (run once at module load or server startup)
def load_molecule_database(csv_path: str) -> pd.DataFrame:
    """Read CSV, clean data, calculate properties for each molecule"""
    # Read CSV
    # Clean up values
    # Calculate properties
    # Return dataframe


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
    
    # convert distance to show similarity scores
    similarity_scores = 1 / (1 + distances)

    db_results = db_properties.copy()
    db_results['similarity_score'] = similarity_scores

    return db_results

def get_top_similar_molecules(
    dataframe, selected_properties, n
):
    """Filter and rank molecules by similarity to selected properties"""
    # Apply calculate_similarity_score to each row
    # Sort by score
    # Return top N

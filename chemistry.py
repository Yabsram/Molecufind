from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import os, shutil
import pandas as pd

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


def calculate_similarity_score(selected_properties: dict, property_row: dict) -> float:
    """Compare selected properties to a molecule's properties"""
    # Returns numerical similarity score (0-1 or 0-100)


def get_top_similar_molecules(
    dataframe: pd.DataFrame, selected_properties: dict, top_n: int = 30
) -> pd.DataFrame:
    """Filter and rank molecules by similarity to selected properties"""
    # Apply calculate_similarity_score to each row
    # Sort by score
    # Return top N

def show_images(dataframe):
    folder_path = "images"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)

    #d = {'col1': ["Paracetamol", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "two"], 'col2': ["Nicotine", "CN1CCCC1C2=CN=CC=C2", "two"]}
    #df = pd.DataFrame(data=d)
    df = dataframe

    # Second column (index 1)
    for col in df.columns:
        name = df[col][0] #find out what index the name is stored at
        structure = df[col][1] #find out what index the structure is stored at

        mol = Chem.MolFromSmiles(structure)

        # Save to file
        filename = name + ".png"
        img = Draw.MolToImage(mol)
        filepath = os.path.join(folder_path, filename)
        img.save(filepath)       
    return
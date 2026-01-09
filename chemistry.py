from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import Crippen
import pandas as pd
import os, shutil
import numpy as np
from admet_ai import ADMETModel
from sklearn.preprocessing import StandardScaler
from math import pi
import matplotlib.pyplot as plt


model = ADMETModel()
def test_admet_model():
    test_smiles = ["CCOC(=O)C1(c2ccccc2)CCCN(C)CC1.O=C(O)CC(O)(CC(=O)O)C(=O)O", "CCOC(=O)C1(c2ccccc2)CCCN(C)CC1.O=C(O)CC(O)(CC(=O)O)C(=O)O"]
    predictions = model.predict(smiles=test_smiles)
    print(f"Number of properties predicted: {len(predictions.columns)}")
    print(f"\nAvailable properties:\n{list(predictions.columns)}")
    print(f"\nPredictions:\n{predictions}")
    return predictions

def plot_admet_radar_clean(smiles: str):
    predictions = model.predict(smiles=[smiles])
    
    print(f"\n\nType: {type(predictions)}")
    print(f"\nShape: {predictions.shape}")
    print(f"\nContent check: {not predictions.empty}")
    if hasattr(predictions, 'head'):
        print(f"\nContent Head:\n{predictions.head()}")
    else:
        print(f"\nContent: {predictions}")

    if hasattr(predictions, 'keys'):
        print(f"Keys: {list(predictions.keys())}")
        
    properties = {
        'hERG Safe': 100 - (predictions['hERG'].values[0]*100),
        'Lipinski': predictions['Lipinski'].values[0],
        'Soluble': (predictions['Solubility_AqSolDB'].values[0]+5)*20,
        'Non-Toxic': 100 - (predictions['AMES'].values[0] * 100),
        'Blood Brain Barrier Safe': 100 - (predictions['BBB_Martins'].values[0]*100),
    }
    
    categories = list(properties.keys())
    values = list(properties.values())
    num_vars = len(categories)
    #compute angle for each axis
    angles = [n / float(num_vars)*2*pi for n in range(num_vars)]
    values += values[:1]
    angles += angles[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.plot(angles, values, 'o-', linewidth=2.5, color='#E74C3C', markersize=8)
    ax.fill(angles, values, alpha=0.3, color='#E74C3C')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, weight='bold')
    
    # Set y-axis (radial) limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100'], size=9, color='gray')
    
    ax.grid(True, linewidth=0.5, color='gray', alpha=0.3)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_linewidth(1.5)
    plt.tight_layout()
    return fig

def save_radar_chart(smiles: str, name: str, output_folder: str = "static/images"):
    """Generate and save radar chart for a molecule."""
    try:
        fig = plot_admet_radar_clean(smiles)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = os.path.join(output_folder, f"radar_{name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error generating radar chart for {name}: {e}")

def test_radar_chart():
    smile = "CC[C@H]1C[C@H]2[C@@H]3CCC4=CC(=O)CC[C@@H]4[C@H]3CC[C@]2(C)[C@H]1O"
    fig = plot_admet_radar_clean(smile)
    output_file = "radar_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
# if __name__ == "__main__":
#     test_radar_chart()
#     test_admet_model()
    

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
    df = pd.read_csv(
        "original.csv",
        sep=";",
        usecols=[
            "ChEMBL ID",
            "Name",
            "Synonyms",
            "Molecular Weight",
            "Polar Surface Area",
            "Aromatic Rings",
            "Heavy Atoms",
            "Molecular Formula",
            "Smiles",
        ],
    )
    df["Lipophilicity"] = df["Smiles"].apply(calculate_lipophilicity)
    df.dropna(how="any", inplace=True)
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
    property_cols = [
        "Molecular Weight",
        "Polar Surface Area",
        "Aromatic Rings",
        "Heavy Atoms",
        "Lipophilicity",
    ]

    # extract desired cols
    db_extracted_values = db_properties[property_cols]

    # ensure selected_properties is a df, make into df
    if not isinstance(selected_properties, pd.DataFrame):
        selected_properties = pd.DataFrame([selected_properties])

    # make sure all expected columns exist
    # fill missing with NaN so we can still run
    selected_extracted = selected_properties.reindex(columns=property_cols)

    # Only use properties that user actually provided (not NaN)
    # Get mask of which properties were provided
    provided_props = ~selected_extracted.isna().iloc[0]
    provided_cols = [col for col, provided in provided_props.items() if provided]

    if not provided_cols:
        print("No properties provided!")
        return db_properties.assign(similarity_score=0.0)

    # Only use the columns the user provided for similarity calculation
    db_for_calc = db_extracted_values[provided_cols]
    selected_for_calc = selected_extracted[provided_cols].dropna()

    # normalize data frames
    scaler = StandardScaler()
    db_normalized = scaler.fit_transform(db_for_calc)

    # normalize selected_properties using same scaler
    selected_normalized = scaler.transform(selected_for_calc)

    print(f"Selected normalized values: {selected_normalized}")  # Debug

    # calculate distances using euclidean distance
    distances = np.linalg.norm(db_normalized - selected_normalized, axis=1)

    # convert distance to show similarity scores using exponential decay
    # creates more separation between similar and different molecules
    similarity_scores = np.exp(-distances / 2)

    db_results = db_properties.copy()
    db_results["similarity_score"] = similarity_scores

    return db_results


def get_top_similar_molecules(db_properties, selected_properties, n):
    """Filter and rank molecules by similarity to selected properties"""
    # Apply calculate_similarity_score to each row
    results = calculate_similarity_score(selected_properties, db_properties)

    # Sort by score
    sorted_top_results = results.sort_values(by="similarity_score", ascending=False)

    # return top n results
    return sorted_top_results.head(n)


def show_images(dataframe):
    folder_path = "static/images"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)

    # d = {'cogl1': ["Paracetamol", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "two"], 'col2': ["Nicotine", "CN1CCCC1C2=CN=CC=C2", "two"]}
    # df = pd.DataFrame(data=d)
    df = dataframe

    # Second column (index 1)
    for idx, row in df.iterrows():
        name = row.get(
            "Name", f"molecule_{idx}"
        )  # find out what index the name is stored at
        structure = row.get(
            "Smiles", None
        )  # find out what index the structure is stored at
        if not isinstance(structure, str) or not structure.strip():
            continue

        mol = Chem.MolFromSmiles(structure)

        # Save molecule structure image to file
        filename = name + ".png"
        img = Draw.MolToImage(mol)
        filepath = os.path.join(folder_path, filename)
        img.save(filepath)
        
        # Generate and save radar chart
        save_radar_chart(structure, name, folder_path)
    return

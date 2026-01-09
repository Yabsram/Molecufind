from flask import Flask, render_template
import pandas as pd, os, shutil
from rdkit import Chem
from rdkit.Chem import Draw
import chemistry

app = Flask(__name__)

#molecule_db = load_molecule_database("original.csv")

@app.route("/", methods=["GET"])
def select_property():
    return
    #show the form to select properties

@app.route("/results", methods=["POST"])
def show_results():
    return
    #display top results

if __name__ == '__main__':
    d = {
    'col1': ["Paracetamol", "CC(=O)NC1=CC=C(C=C1)O", "two"],
    'col2': ["Nicotine", "CN1CCCC1C2=CN=CC=C2", "two"],
    'col3': ["Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", "three"],
    'col4': ["Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "three"],
    'col5': ["Morphine", "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O", "five"],
    'col6': ["Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "four"],
    'col7': ["Dopamine", "NCCC1=CC=C(O)C(O)=C1", "two"]
    }
    df = pd.DataFrame(data=d)
    chemistry.show_images(df)
    
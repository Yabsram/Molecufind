from flask import Flask, render_template
import pandas as pd, os, shutil
from rdkit import Chem
from rdkit.Chem import Draw
import chemistry
from chemistry import load_molecule_database

app = Flask(__name__)

#molecule_db = load_molecule_database("original.csv")

@app.route("/", methods=["GET"])
def select_property():
    #show the form to select properties
    return render_template(
        "selection.html"
    )

@app.route("/results", methods=["POST"])
def show_results():
    #display top results 
   return render_template(
        "base.html"
    )
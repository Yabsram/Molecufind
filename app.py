from flask import Flask, render_template, request
import pandas as pd, os, shutil
from rdkit import Chem
from rdkit.Chem import Draw
import chemistry
from chemistry import load_molecule_database
import os

app = Flask(__name__)

# molecule_db = load_molecule_database("original.csv")


@app.route("/", methods=["GET", "POST"])
def select_property():
    # This will hold the selected filters and user input
    selected = {}

    # Map filter display names to form field names
    filters = {
        "Molecular Weight": ("amw", "amw_value"),
        "Lipophilicity": ("clogp", "clogp_value"),
        "Polar Surface Area": ("tpsa", "tpsa_value"),
        "Heavy Atoms": ("numHeavyAtoms", "numHeavyAtoms_value"),
        "Aromatic Rings": ("numAromaticRings", "numaromaticrings_value"),
    }

    if request.method == "POST":
        for display_name, (checkbox_name, value_name) in filters.items():
            # Check if checkbox was checked
            if request.form.get(checkbox_name):
                # Only store if checkbox checked and convert to float
                selected[display_name] = float(request.form.get(value_name, 0))

        if selected:
            molecule_db = load_molecule_database("original.csv")
            results = chemistry.get_top_similar_molecules(molecule_db, selected, n=20)
            chemistry.show_images(results)
            return render_template("base.html", selected=selected, results=results)

    return render_template("base.html", selected={})


@app.route("/results", methods=["POST"])
def show_results():
    # display top results
    return render_template("base.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8006)))

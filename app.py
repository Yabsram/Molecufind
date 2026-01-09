from flask import Flask, render_template
from chemistry import load_molecule_database

app = Flask(__name__)

molecule_db = load_molecule_database("original.csv")

@app.route("/", methods=["GET"])
def select_property():
    return render_template("base.html")

@app.route("/results", methods=["POST"])
def show_results():
    #display top results
    return "TODO"
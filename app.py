from flask import Flask, render_template

app = Flask(__name__)

molecule_db = load_molecule_database("original.csv")

@app.route("/", methods=["GET"])
def select_property():
    #show the form to select properties

@app.route("/results", methods=["POST"])
def show_results():
    #display top results 
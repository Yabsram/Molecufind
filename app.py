from flask import Flask, render_template

app = Flask(__name__)

#molecule_db = load_molecule_database("original.csv")

@app.route("/", methods=["GET"])
def select_property():
    return render_template(
        "selection.html"
    )
@app.route("/results", methods=["POST"])
def show_results():
   return render_template(
        "base.html"
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8004)))

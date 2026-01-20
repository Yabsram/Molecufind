# MolecuFind 🔍

MolecuFind is a web application that helps chemists and researchers quickly identify molecules that best match a desired set of chemical properties. It is designed to support early stage drug discovery by narrowing down candidate molecules using property based similarity.

Built as a hackathon project, MolecuFind focuses on clarity, speed, and practical usability.

## Key Features ✨
- Select up to five core molecular properties
- Input desired target values for any subset of properties
- Retrieve the top 20 closest matching molecules
- View similarity scores for each match
- Visualize and compare molecular property profiles
- Simple and intuitive web interface

## Supported Molecular Properties ⚛️
- Molecular Weight
- Crippen CLogP (Lipophilicity)
- Polar Surface Area (PSA)
- Number of Heavy Atoms
- Number of Aromatic Rings
Users may specify values for as many or as few properties as they choose.

## Dataset 📊
- Source: ChEMBL
- Properties extracted and computed using RDKit
- Intended for exploratory analysis and prototyping

## Tech Stack 🛠️
### Backend
- Python
- Flask
- RDKit
- scikit-learn
- NumPy
- Pandas

### Frontend
- HTML
- CSS
- Jinja templates

## Running Locally 🚀
- Clone the repository
- Create and activate a virtual environment
- Install dependencies
- Start the app
- Open

## Future Improvements 🔮
- Additional molecular descriptors
- Property weighting controls
- More advanced similarity metrics
- Enhanced visual analytics
- Expanded dataset support

RDKit for molecular computation tools

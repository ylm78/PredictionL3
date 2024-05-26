from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from L3_Transverse import *

 # Assurez-vous d'importer votre module de prédiction ici

app = Flask(__name__)

# Créez une instance de votre modèle de prédiction
model = RandomForestClassifier(
    n_estimators=100,    # Nombre d'arbres dans la forêt
    criterion='gini',    # Fonction pour mesurer la qualité d'une scission
    max_depth=None,      # Profondeur maximale de l'arbre, None signifie que les nœuds sont développés jusqu'à ce que toutes les feuilles soient pures
    min_samples_split=2, # Le nombre minimal d'échantillons requis pour scinder un nœud interne
    min_samples_leaf=1,  # Le nombre minimal d'échantillons requis pour être à un nœud feuille
    random_state=42      # Une graine pour la reproductibilité des résultats
)

@app.route('/')
def index():
    teams = get_team_names()
    return render_template('index.html', teams=teams)  # Page HTML pour l'entrée utilisateur


@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']
    prediction = model.predict_match(home_team, away_team)
    return jsonify(result=prediction)

if __name__ == '__main__':
    app.run(debug=True)

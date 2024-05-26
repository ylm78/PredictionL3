from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/results')
def results():
    with open('matches.json', 'r') as f:
        matches = json.load(f)
    return render_template('results.html', matches=matches)

if __name__ == '__main__':
    app.run(debug=True)

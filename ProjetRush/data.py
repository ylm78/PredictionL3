import json

def save_output(data, key, filename="output.json"):
    try:
        with open(filename, 'r+') as file:
            # Chargement des données existantes
            file_data = json.load(file)
            # Mise à jour des données
            file_data[key] = data
            # Réinitialisation de la position du curseur au début du fichier
            file.seek(0)
            # Écriture des données mises à jour
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        # Si le fichier n'existe pas, il sera créé
        with open(filename, 'w') as file:
            json.dump({key: data}, file, indent=4)

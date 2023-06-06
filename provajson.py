import json
import numpy as np
"""
with open("simulation.json", "r") as file_json:
    # Leggi il contenuto del file JSON
    contenuto = json.load(file_json)
"""
Qdet = 10.51662321197508
z_deviation= [  8.62222189, -30.7984888 ]
Q = [[5.16666667e-01, 4.76654034e-13], [4.76654285e-13, 5.16666667e-01]]
print(np.linalg.inv(Q))
Q = Qdet**(-1/2)*np.exp(-1/2*np.transpose(z_deviation) @ np.linalg.inv(Q) @ z_deviation)
print
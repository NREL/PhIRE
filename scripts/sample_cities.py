import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    cities = pd.read_csv('worldcities.csv')

    filtered = cities[cities.population >= 1e6]
    p, edges_lng, edges_lat = np.histogram2d(filtered.lng, filtered.lat, bins=[30, 15], density=False)

    edges_lng[-1] += 1e5
    edges_lat[-1] += 1e5
    a = np.digitize(filtered.lng, edges_lng) - 1
    b = np.digitize(filtered.lat, edges_lat) - 1

    sampled = filtered.sample(150, weights=1/p[a,b]).sort_values(['lng', 'lat'])
    sampled.to_csv('cities.csv', index=False)

    plt.subplots(figsize=(10,7))
    plt.hist2d(filtered.lng, filtered.lat, bins=[30, 15])
    plt.colorbar(orientation="horizontal")
    plt.gca().set_aspect('equal')
    plt.show()

    plt.subplots(figsize=(10,7))
    plt.hist2d(sampled.lng, sampled.lat, bins=[30, 15])
    plt.colorbar(orientation="horizontal")
    plt.gca().set_aspect('equal')
    plt.show()
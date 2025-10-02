import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MPLCircle
from shapely import get_parts


def visualize_chordal_env(valid_polygons):
    valid_faces = list(get_parts(valid_polygons))
    valid_areas = [face.area for face in valid_faces]
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_faces)))
    for face, color in zip(valid_faces, colors):
        x, y = face.exterior.xy
        ax.fill(x, y, color=color, alpha=0.6, edgecolor='black', linewidth=1)
        centroid = face.centroid
        ax.text(centroid.x, centroid.y, f'{face.area:.3f}', 
                ha='center', va='center', fontsize=8, weight='bold')

    # Draw the chords
    for chord in chords:
        x, y = chord.xy
        ax.plot(x, y, 'k-', linewidth=2, alpha=0.7)

    circle_patch = MPLCircle((0, 0), radius, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(circle_patch)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Circle with {n_chords} chords (angle + bias parameterization)')

    plt.tight_layout()
    plt.show()
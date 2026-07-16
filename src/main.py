"""Rede de Hopfield para recuperação das letras A, E e X."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# Configurações
# ==========================================================

SEED = 42
NOISE_FRACTION = 0.30
MAX_UPDATES = 5_000
IMAGE_SHAPE = (7, 7)

# Pasta na qual as figuras serão armazenadas.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Funções da Rede de Hopfield
# ==========================================================

def train_hebb(patterns):
    """Calcula os pesos da rede usando a regra de Hebb."""
    number_of_neurons = patterns.shape[1]
    weights = patterns.T @ patterns / number_of_neurons

    # Um neurônio não deve possuir conexão consigo mesmo.
    np.fill_diagonal(weights, 0)

    return weights


def calculate_energy(weights, state):
    """Calcula a energia de um estado da rede."""
    return -0.5 * state @ weights @ state


def update_neuron(weights, state, rng):
    """Atualiza aleatoriamente um único neurônio."""
    updated_state = state.copy()
    index = rng.integers(len(updated_state))

    local_field = weights[index] @ updated_state
    new_value = 1 if local_field >= 0 else -1
    changed = new_value != updated_state[index]

    updated_state[index] = new_value

    return updated_state, changed


def recover_pattern(weights, corrupted, target, rng):
    """Executa atualizações assíncronas até a estabilização."""
    state = corrupted.copy()

    states = [state.copy()]
    energies = [calculate_energy(weights, state)]
    errors = [np.count_nonzero(state != target)]

    unchanged_updates = 0
    updates = 0

    while unchanged_updates < len(state) and updates < MAX_UPDATES:
        state, changed = update_neuron(weights, state, rng)

        states.append(state.copy())
        energies.append(calculate_energy(weights, state))
        errors.append(np.count_nonzero(state != target))

        unchanged_updates = 0 if changed else unchanged_updates + 1
        updates += 1

    return states, energies, errors


# ==========================================================
# Padrões 7 × 7
# ==========================================================

def create_pattern(rows):
    """Converte uma representação textual em valores -1 e 1."""
    return np.array([
        [1 if pixel == "#" else -1 for pixel in row]
        for row in rows
    ]).flatten()


PATTERNS = {
    "A": create_pattern([
        "..###..", ".#...#.", "#.....#", "#######",
        "#.....#", "#.....#", "#.....#",
    ]),
    "E": create_pattern([
        "#######", "#......", "#......", "#####..",
        "#......", "#......", "#######",
    ]),
    "X": create_pattern([
        "#.....#", ".#...#.", "..#.#..", "...#...",
        "..#.#..", ".#...#.", "#.....#",
    ]),
}


# ==========================================================
# Gráficos
# ==========================================================

def plot_recovery(names, originals, corrupted, recovered):
    """Exibe os padrões originais, corrompidos e recuperados."""
    figure, axes = plt.subplots(3, 3, figsize=(8, 8))

    for column, title in enumerate(
        ["Original", "Corrompida", "Recuperada"]
    ):
        axes[0, column].set_title(title, fontsize=15)

    for row, name in enumerate(names):
        images = [originals[row], corrupted[row], recovered[row]]

        for column, image in enumerate(images):
            axes[row, column].imshow(
                image.reshape(IMAGE_SHAPE),
                cmap="gray_r",
                vmin=-1,
                vmax=1,
            )
            axes[row, column].axis("off")

        axes[row, 0].set_ylabel(
            name,
            fontsize=18,
            fontweight="bold",
            rotation=0,
            labelpad=25,
        )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "recuperacao_AEX.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_metrics(names, energy_curves, error_curves):
    """Gera os gráficos de energia e distância de Hamming."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["green", "red", "blue"]
    markers = ["o", "s", "^"]

    for name, energies, errors, color, marker in zip(
        names, energy_curves, error_curves, colors, markers
    ):
        axes[0].plot(
            energies, color=color, marker=marker,
            markevery=5, markersize=4, label=name,
        )
        axes[1].plot(
            errors, color=color, marker=marker,
            markevery=8, markersize=4, label=name,
        )

    descriptions = [
        ("Energia", "(a)"),
        ("Distância de Hamming", "(b)"),
    ]

    for axis, (ylabel, title) in zip(axes, descriptions):
        axis.set_xlabel("Atualizações de neurônios", fontsize=13)
        axis.set_ylabel(ylabel, fontsize=13)
        axis.set_title(title, fontsize=16, loc="left")
        axis.grid(color="0.85", linewidth=0.8)
        axis.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "energia_hamming_AEX.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# ==========================================================
# Execução principal
# ==========================================================

def main():
    """Treina a rede, adiciona ruído e recupera os padrões."""
    rng = np.random.default_rng(SEED)

    names = list(PATTERNS.keys())
    patterns = np.array(list(PATTERNS.values()))
    weights = train_hebb(patterns)

    corrupted_patterns = []
    recovered_patterns = []
    energy_curves = []
    error_curves = []

    # Usa posições de ruído diferentes para cada letra.
    number_of_noisy_pixels = int(NOISE_FRACTION * patterns.shape[1])

    for name, pattern in zip(names, patterns):
        corrupted = pattern.copy()
        noisy_indexes = rng.choice(
            len(pattern),
            number_of_noisy_pixels,
            replace=False,
        )
        corrupted[noisy_indexes] *= -1

        states, energies, errors = recover_pattern(
            weights, corrupted, pattern, rng
        )

        corrupted_patterns.append(corrupted)
        recovered_patterns.append(states[-1])
        energy_curves.append(energies)
        error_curves.append(errors)

        # Compara o resultado com todos os padrões armazenados.
        distances = {
            stored_name: np.count_nonzero(states[-1] != stored_pattern)
            for stored_name, stored_pattern in zip(names, patterns)
        }

        print(f"{name}: distância final = {errors[-1]}")
        print(f"Distâncias para os padrões armazenados: {distances}")

    plot_recovery(
        names, patterns, corrupted_patterns, recovered_patterns
    )
    plot_metrics(names, energy_curves, error_curves)

    print(f"Figuras salvas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
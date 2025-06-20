import numpy as np
from math import sqrt
import pandas as pd
import json

# =================== Funções ===================

def normalize_decision_matrix_fixed(data):
    lv = data["linguistic_variables_alternatives"]
    decision_matrix = data["decision_matrix"]
    criteria_type = data["criteria_type"]
    normalized_matrix = {}
    for criterion, values in decision_matrix.items():
        fuzzy_values = [lv[val] for val in values]
        fuzzy_array = np.array(fuzzy_values)
        if criteria_type[criterion] == "Benefit":
            d_star = np.max(fuzzy_array[:, 3])
            normalized = np.array([[v / d_star for v in row] for row in fuzzy_array])
        else:  # Cost
            a_min = np.min(fuzzy_array[:, 0])
            normalized = np.array([[a_min / v if v != 0 else 0 for v in row] for row in fuzzy_array])
        normalized_matrix[criterion] = normalized.tolist()
    return normalized_matrix

def construct_weighted_normalized_matrix(normalized_matrix, weights, linguistic_weights):
    weighted_matrix = {}
    for criterion, norm_values in normalized_matrix.items():
        weight_label = weights[criterion][0]
        w = linguistic_weights[weight_label]
        weighted_matrix[criterion] = [
            [round(r * w_i, 4) for r, w_i in zip(row, w)] for row in norm_values
        ]
    return weighted_matrix

def get_positive_ideal_solutions(profile_matrix, linguistic_variables):
    return {
        profile: {
            criterion: linguistic_variables[label]
            for criterion, label in criteria.items()
        }
        for profile, criteria in profile_matrix.items()
    }

def fuzzy_distance(a, b):
    return sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)) / 4)

def get_negative_ideal_solutions(profile_matrix, linguistic_variables):
    profiles = list(profile_matrix.keys())
    fuzzy_profiles = {
        profile: {
            criterion: linguistic_variables[label]
            for criterion, label in profile_matrix[profile].items()
        }
        for profile in profiles
    }
    negative_ideal_solutions = {}
    for p in profiles:
        farthest_profile = max(
            (p2 for p2 in profiles if p2 != p),
            key=lambda p2: sum(
                fuzzy_distance(fuzzy_profiles[p][c], fuzzy_profiles[p2][c])
                for c in profile_matrix[p]
            )
        )
        negative_ideal_solutions[p] = fuzzy_profiles[farthest_profile]
    return negative_ideal_solutions

def apply_weights_to_profiles(profile_solutions, weights, linguistic_weights):
    weighted_profiles = {}
    for profile, criteria in profile_solutions.items():
        weighted_profiles[profile] = {}
        for criterion, values in criteria.items():
            weight = linguistic_weights[weights[criterion][0]]
            weighted_profiles[profile][criterion] = [round(v * w, 4) for v, w in zip(values, weight)]
    return weighted_profiles

def calculate_distances_to_ideal_solutions(weighted_matrix, positive_ideal, negative_ideal, alternatives):
    distances = {}
    for profile in positive_ideal:
        distances[profile] = {"positive": {}, "negative": {}}
        for i, alt in enumerate(alternatives):
            d_pos = sum(fuzzy_distance(weighted_matrix[crit][i], positive_ideal[profile][crit]) for crit in weighted_matrix)
            d_neg = sum(fuzzy_distance(weighted_matrix[crit][i], negative_ideal[profile][crit]) for crit in weighted_matrix)
            distances[profile]["positive"][alt] = round(d_pos, 4)
            distances[profile]["negative"][alt] = round(d_neg, 4)
    return distances

def calculate_closeness_coefficients(distance_results, alternatives):
    cc = {}
    for profile in distance_results:
        cc[profile] = {}
        for alt in alternatives:
            d_pos = distance_results[profile]["positive"][alt]
            d_neg = distance_results[profile]["negative"][alt]
            cc[profile][alt] = round(d_neg / (d_pos + d_neg), 4) if (d_pos + d_neg) else 0.0
    return cc

def classify_alternatives(closeness_coeffs):
    classification = {}
    profiles = list(closeness_coeffs.keys())
    for alt in closeness_coeffs[profiles[0]]:
        best_profile = max(profiles, key=lambda p: closeness_coeffs[p][alt])
        classification[alt] = best_profile
    return classification

# =================== Execução ===================

if __name__ == "__main__":
    with open("input.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_matrix = normalize_decision_matrix_fixed(data)
    weighted_matrix = construct_weighted_normalized_matrix(
        normalized_matrix, data["weights"], data["linguistic_variables_weights"]
    )
    positive_ideal = get_positive_ideal_solutions(data["profile_matrix"], data["linguistic_variables_alternatives"])
    negative_ideal = get_negative_ideal_solutions(data["profile_matrix"], data["linguistic_variables_alternatives"])
    positive_ideal_weighted = apply_weights_to_profiles(positive_ideal, data["weights"], data["linguistic_variables_weights"])
    negative_ideal_weighted = apply_weights_to_profiles(negative_ideal, data["weights"], data["linguistic_variables_weights"])

    distance_results = calculate_distances_to_ideal_solutions(
        weighted_matrix, positive_ideal_weighted, negative_ideal_weighted, data["alternatives"]
    )
    closeness_coeffs = calculate_closeness_coefficients(distance_results, data["alternatives"])
    classification = classify_alternatives(closeness_coeffs)

    # Montar tabela final
    result_table = {}
    for alt in data["alternatives"]:
        result_table[alt] = {
            f"{p}_d_pos": distance_results[p]["positive"][alt] for p in data["profile_matrix"]
        }
        result_table[alt].update({
            f"{p}_d_neg": distance_results[p]["negative"][alt] for p in data["profile_matrix"]
        })
        result_table[alt].update({
            f"{p}_cc": closeness_coeffs[p][alt] for p in data["profile_matrix"]
        })
        result_table[alt]["Classificação"] = classification[alt]

    df_result = pd.DataFrame.from_dict(result_table, orient="index")
    df_result.to_csv("resultado_ftopsis.csv")
    print(df_result)

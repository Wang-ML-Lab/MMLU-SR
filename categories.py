"Code originally copied from MMLU https://github.com/hendrycks/test/blob/master/categories.py"

import os
import pandas as pd
import json

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other": ["other", "business", "health"],
}

# Function to load results and calculate average
def load_and_process_results(results_file):
    with open(results_file, 'r') as file:
        results_data = json.load(file)

    # Normalize categories for case-insensitivity issues
    normalized_categories = {k.lower(): v for k, v in categories.items()}
    category_scores = {k: [] for k in normalized_categories}
    total_scores = []
    unmatched_subjects = []

    for subject, score in results_data.items():
        if subject == "all":
            continue
        matched = False

        # Normalize subcategories access
        subject_lower = subject.lower()  # Normalize subject for matching
        if subject_lower in subcategories:
            for subcat in subcategories[subject_lower]:
                for category, members in normalized_categories.items():
                    if subcat in members:
                        category_scores[category].append(score)
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                unmatched_subjects.append(subject)
        else:
            unmatched_subjects.append(subject)

    if unmatched_subjects:
        print("Unmatched subjects:", unmatched_subjects)

    # Calculate averages
    average_scores = {cat: sum(scores) / len(scores) if scores else 0 for cat, scores in category_scores.items()}
    total_scores = [score for score in results_data.values() if isinstance(score, (int, float))]  # Ensure 'all' or other non-score values are excluded
    total_average = sum(total_scores) / len(total_scores) if total_scores else 0

    return average_scores, total_average

# Main function to generate the table
def main():
    results_file = "/results/question_and_answer_gpt-3.5-turbo_5-shot_accs.json"  # Path to your results JSON file
    average_scores, total_average = load_and_process_results(results_file)

    # Print the table (can be improved with a proper table formatting library like tabulate)
    print("Model\tHumanities\tSocial Sciences\tSTEM\tOther\tAverage")
    model_info = "GPT-3.5-turbo (5-shot)"
    print(f"{model_info}\t"
          f"{average_scores.get('humanities', 0):.3f}\t"
          f"{average_scores.get('social sciences', 0):.3f}\t"
          f"{average_scores.get('stem', 0):.3f}\t"
          f"{average_scores.get('other', 0):.3f}\t"
          f"{total_average:.3f}")

if __name__ == "__main__":
    main()
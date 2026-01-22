import pandas as pd
import os

def generate_sample_data(output_file="data/medquad.csv"):
    """
    Creates a sample dataset if the real one doesn't exist, to allow immediate testing.
    """
    if os.path.exists(output_file):
        print(f"{output_file} already exists.")
        return

    data = [
        {
            "Focus": "Influenza",
            "Question": "What are the symptoms of the flu?",
            "Answer": "Symptoms of the flu can include fever, cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue (tiredness). Some people may have vomiting and diarrhea, though this is more common in children than adults.",
            "Source": "cdc_gov_flu.xml"
        },
        {
            "Focus": "Influenza",
            "Question": "How does the flu spread?",
            "Answer": "Most experts believe that flu viruses spread mainly by tiny droplets made when people with flu cough, sneeze or talk. These droplets can land in the mouths or noses of people who are nearby. Less often, a person might get flu by touching a surface or object that has flu virus on it and then touching their own mouth, nose, or possibly their eyes.",
            "Source": "cdc_gov_flu.xml"
        },
        {
            "Focus": "Diabeties",
            "Question": "What is type 2 diabetes?",
            "Answer": "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). With type 2 diabetes, your body either doesn't produce enough insulin, or it resists insulin.",
            "Source": "niddk_nih_gov_diabetes.xml"
        },
        {
            "Focus": "Hypertension",
            "Question": "What is high blood pressure?",
            "Answer": "High blood pressure (hypertension) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
            "Source": "nhlbi_nih_gov_hbp.xml"
        }
    ]
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Created sample data at {output_file}")

if __name__ == "__main__":
    generate_sample_data()

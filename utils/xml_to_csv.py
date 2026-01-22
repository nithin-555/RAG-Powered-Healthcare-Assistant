import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def process_xml_files(input_dir="data", output_file="data/medquad.csv"):
    """
    Parses MedQuAD XML files and converts them to a CSV.
    Assumes XML files are directly in input_dir or subdirectories.
    """
    data = []
    
    # Walk through all XML files
    xml_files = glob.glob(os.path.join(input_dir, "**", "*.xml"), recursive=True)
    
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return False

    print(f"Found {len(xml_files)} XML files. Processing...")

    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract basic info
            focus = root.findtext("Focus")
            if not focus: 
                focus = "General Health"

            # Iterate over QAPairs
            qa_pairs = root.find("QAPairs")
            if qa_pairs is not None:
                for pair in qa_pairs.findall("QAPair"):
                    question = pair.findtext("Question")
                    answer = pair.findtext("Answer")
                    
                    if question and answer:
                        data.append({
                            "Focus": focus,
                            "Question": question,
                            "Answer": answer,
                            "Source": os.path.basename(file_path)
                        })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(df)} QA pairs to {output_file}")
        return True
    else:
        print("No valid QA pairs extracted.")
        return False

if __name__ == "__main__":
    process_xml_files()

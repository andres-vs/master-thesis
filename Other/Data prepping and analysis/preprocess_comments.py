import pandas as pd
import re
import csv

# Step 1: Data Preparation

def read_comments(file_path):
    comments = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into folder, file, and comment parts using the new delimiter
            parts = line.strip().split('|||')
            if len(parts) == 3:
                folder = parts[0].replace('Folder: ', '')
                file_name = parts[1].replace('File: ', '')
                comment = parts[2].replace('Comment: ', '')
                comments.append({
                    'folder': folder,
                    'file': file_name,
                    'comment': comment
                })
            else:
                # Debugging statement for lines that do not match the expected format
                print(f"Skipping line: {line.strip()}")
    return comments

def extract_metadata(file_name):
    # Extract layer and head from the filename
    layer_match = re.search(r'blocks\.(\d+)', file_name)
    head_match = re.search(r'head_(\d+)', file_name)
    layer = int(layer_match.group(1)) if layer_match else None
    head = int(head_match.group(1)) if head_match else None
    return layer, head

def extract_prompt_and_type(folder_name):
    # Extract prompt number and input type from the folder name
    prompt_match = re.search(r'prompt-(\d+)', folder_name)
    type_match = re.search(r'type (\d+)', folder_name)
    prompt = int(prompt_match.group(1)) if prompt_match else None
    input_type = int(type_match.group(1)) if type_match else None
    return prompt, input_type

# Tags comment by matching it with all found matching patterns by checking whether a keyword of the pattern is also in the comment
def tag_pattern_category(comment, patterns):
    matching_categories = []
    for category, keywords in patterns.items():
        for keyword in keywords:
            if keyword in comment:
                matching_categories.append(category)
                break  # Avoid adding the same category multiple times
    return matching_categories if matching_categories else ['other']

def preprocess_comments(comments):
    data = []
    for comment in comments:
        layer, head = extract_metadata(comment['file'])
        prompt, input_type = extract_prompt_and_type(comment['folder'])
        data.append({
            'Layer': layer,
            'Head': head,
            'Prompt': prompt,
            'Input_Type': input_type,
            'Comment': comment['comment']
        })
    return pd.DataFrame(data)

# Example usage
comments = read_comments('comments.txt')
print(f"Total comments read: {len(comments)}")  # Debugging statement

df = preprocess_comments(comments)
print(f"Total comments processed: {len(df)}")  # Debugging statement

# Sort the DataFrame hierarchically by Input_Type, Layer, and Head
df_sorted = df.sort_values(by=['Input_Type', 'Layer', 'Head'])

# Print the sorted DataFrame
print(df_sorted)

# Save the sorted DataFrame to a CSV file with only the Comment column quoted
df_sorted.to_csv('comments_analysis_sorted.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
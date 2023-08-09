import argparse
import pandas as pd
import numpy as np

# parser = argparse.ArgumentParser(description='Data Preprocessing of GoE file')
# parser.add_argument('-i', '--input', type=str, help='Input file path')
# args = parser.parse_args()
input_file_path = 'data/GOE_Content.xlsx'

def preprocess_mockup_data():
    # Read the data from the Excel file into a DataFrame
    mockup_data = pd.read_excel(input_file_path, sheet_name='Mockup Content')

    # Strip the "" from the titles and remove extra spaces at the end
    mockup_data['Title'] = mockup_data['Title'].apply(lambda x: x[1:-1].strip())

    # Sort the DataFrame by Title
    mockup_data = mockup_data.sort_values('Title')

    # Remove duplicate titles
    mockup_data.drop_duplicates(subset='Title', inplace=True)

    # Convert Mind-Beauty to Beauty
    mockup_data['Pillar'] = mockup_data['Pillar'].apply(lambda x: ','.join(['Beauty' if val.strip() == 'Mind-Beauty' else val for val in x.split(',')]))

    # Change the name of the columns to match the real dataset
    mockup_data.rename(columns={'Type of Content': 'Type', 'Category': 'Categories', 'Require Movement': 'Require_Movement'}, inplace=True)

    # Add a new column to include Source (Mockup/Real)
    mockup_data['Source'] = 'Mockup'

    # Remove spaces, extra characters, and duplicates from Tags
    mockup_data['Tags'] = mockup_data['Tags'].str.rstrip(', ')
    mockup_data['Tags'] = mockup_data['Tags'].apply(lambda x: ', '.join(set([tag.strip() for tag in str(x).split(',')])) if pd.notnull(x) else np.nan)

    return mockup_data

def preprocess_real_data():
    # Read the data from the Excel file into a DataFrame
    real_data = pd.read_excel(input_file_path, sheet_name='Real')

    # Sort the DataFrame by Title
    real_data = real_data.sort_values('Title')

    # Remove extra spaces at the ends of titles
    real_data['Title'] = real_data['Title'].apply(lambda x: x.strip())

    # Remove duplicate titles
    real_data.drop_duplicates(subset='Title', inplace=True)

    # Concatenate Series to Tags, remove spaces, extra characters, and duplicates from Tags
    real_data['Tags'] = real_data['Tags'].str.rstrip(', ')
    real_data['Tags'] = real_data['Tags'] + ', ' + real_data['Series']
    real_data['Tags'] = real_data['Tags'].apply(lambda x: ', '.join(set([tag.strip() for tag in str(x).split(',')])) if pd.notnull(x) else np.nan)

    # Remove unnecessary columns
    real_data.drop(['Unnamed: 9'], axis=1, inplace=True)

    # Add a new column to include Source (Mockup/Real)
    real_data['Source'] = 'Real'

    return real_data

def data_combine(mockup, real):
    # Combine both the dataframes
    combined_df = pd.concat([mockup, real])

    # Create the final dataframe structure
    final_df = pd.DataFrame(columns=['ID', 'Title', 'Pillar', 'Instructor', 'Categories', 'Series', 'Difficulty', 'Difficulty_Num',
                                     'Duration', 'Duration_Num', 'Require_Movement', 'Require_Movement_Num', 'Tags',
                                     'Source', 'Type'])

    # Fill the dataframe from combined dataframe
    for col in final_df.columns:
        if col in combined_df.columns:
            final_df[col] = combined_df[col]

    # Assign ids randomly but with consistency
    np.random.seed(123)
    final_df['ID'] = np.random.choice(range(1, len(final_df) + 1), size=len(final_df), replace=False)
    
    return final_df

def encode_columns(df):
    # Encoding difficulty
    difficulty_mapping = {"Beginner": 1, "Intermediate": 2, "Expert": 3}
    df['Difficulty_Num'] = df['Difficulty'].map(difficulty_mapping).fillna(0).astype(int)

    # Encoding duration
    duration_mapping = {"<1 min (Shorts)": 1, "1-5 min": 2, "6-15 min": 3, "15 - 30 min": 4, "30+ min": 5}
    df['Duration_Num'] = df['Duration'].map(duration_mapping).fillna(0).astype(int)

    # Encoding require movement
    movement_mapping = {"Yes": 1, "No": 0}
    df['Require_Movement_Num'] = df['Require_Movement'].map(movement_mapping).fillna(2).astype(int)
    df['Require_Movement_Num'] = df['Require_Movement_Num'].apply(lambda x: np.nan if x == 2 else x)

    # Make the separate as ',' for the Type column
    df['Type'] = df['Type'].str.replace('/', ', ')

    # One-hot encoding for Type
    # Replace missing values with a placeholder value
    df['Type'].fillna('Unknown', inplace=True)

    # Remove duplicates and strip spaces from values
    df['Type'] = df['Type'].apply(lambda x: ', '.join(sorted(set(x.strip() for x in x.split(',')))))

    # Split the comma-separated values into separate columns and remove duplicates
    split_values = df['Type'].str.get_dummies(sep=', ')
    split_values = split_values.loc[:, ~split_values.columns.duplicated()]

    # Concatenate the one-hot encoded columns with the original DataFrame
    df = pd.concat([df, split_values], axis=1)
    
    # Encoding for Pillar
    pillars = ["Active", "Beauty", "Nutrition", "Rest"]
    for pillar in pillars:
        df[pillar] = df['Pillar'].str.contains(pillar).astype(int)

    return df

def error_corrections(df_encoded):
    
    # Clean the data by removing double spaces, spaces before commas, and adding spaces after commas
    df_encoded = df_encoded.replace('\s*,\s*', ', ', regex=True)  # Remove leading and trailing spaces before commas
    df_encoded = df_encoded.replace(' +', ' ', regex=True)  # Remove double spaces
    df_encoded = df_encoded.replace(',(?=\S)', ', ', regex=True)  # Add space after comma if not already present
    
    # Fix the spelling errors for columns
    df_encoded = df_encoded.rename(columns={'Intructor': 'Instructor', 'Recepie': 'Recipe'})
    df_encoded['Pillar'] = df_encoded['Pillar'].replace({'Activity': 'Active'})
    df_encoded['Pillar'] = df_encoded['Pillar'].replace({'Nutrition ': 'Nutrition'})
    df_encoded['Categories'] = df_encoded['Categories'].replace({'Take abreath, Mind and Body':'Take a breath, Mind and Body', 
                                                                'Take abreath, Mind and Body, Relax': 'Take a breath, Mind and Body, Relax'})
    df_encoded['Instructor'] = df_encoded['Instructor'].replace({'Julie Chinmayi ':'Julie Chinmayi', 'Jess and Iain': 'Jess Vergeer, Iain Mcbride'})
    
    return df_encoded

# Function to capitalize the first letter and make the rest lowercase
def capitalize_text(text):
    if pd.isnull(text):  # Check if the value is missing or null
        return text  # Return the missing value as is
    words = text.split()
    capitalized_words = [word.capitalize() if i == 0 or (i > 0 and words[i-1][-1] in ['.', ',']) else word.lower() for i, word in enumerate(words)]
    return ' '.join(capitalized_words)

def lowercase_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase the column in the dataframe
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.lower()

    return df

def replace_tags(tags_column: pd.Series, tags_dict: dict) -> pd.Series:
    """
    Replace tags in the tags column
    """

    for base_term, variations in tags_dict.items():
        for variation in variations:
            tags_column = tags_column.str.replace(variation, base_term)
    return tags_column

def clean_and_tokenize_tags(tags_column: pd.Series) -> pd.Series:
    """
    Clean and tokenize the tags column
    """

    # Lowercase and remove punctuation
    tags_column = tags_column.str.lower().str.replace('[^\w\s]', '', regex=True)
    
    # Split tags by commas and expand each tag into a new row
    tags_column = tags_column.str.split(',').apply(pd.Series, 1).stack()
    
    # Remove index level introduced by stack()
    tags_column.index = tags_column.index.droplevel(-1)

    # Remove first whitespace instance if it exists
    tags_column = tags_column.str.strip()

    return tags_column

def capitalize_final_data(lowercase_df):
    
    # Apply the capitalize_text function to the 'Title' and 'Categories' columns
    lowercase_df['Title'] = lowercase_df['Title'].apply(capitalize_text)
    lowercase_df['Categories'] = lowercase_df['Categories'].apply(capitalize_text)
    lowercase_df['Tags'] = lowercase_df['Tags'].apply(capitalize_text)
    lowercase_df['Pillar'] = lowercase_df['Pillar'].apply(capitalize_text)
    lowercase_df['Difficulty'] = lowercase_df['Difficulty'].apply(capitalize_text)
    lowercase_df['Source'] = lowercase_df['Source'].apply(capitalize_text)
    lowercase_df['Type'] = lowercase_df['Type'].apply(capitalize_text)

    return lowercase_df


if __name__ == '__main__':
    mock = preprocess_mockup_data()
    real = preprocess_real_data()
    df = data_combine(mock, real)
    df_encoded = encode_columns(df)
    df_corrected = error_corrections(df_encoded)
    lowercase_df = lowercase_column(df_corrected)
    
    tags_dict = {
        'affirmation': ['affirmation', 'affirmations'],
        'ambient': ['ambient', 'ambient sounds'],
        'anxiety': ['anxiety', 'anxiety relief'],
        'balance': ['balance', 'balanced', 'balanced diet'],
        'body weight': ['body weight', 'bodyweight', 'bodyweight exercises library', 'bodyweight series'],
        'breathing': ['breathing', 'breathing exercises', 'breathing techniques for health'],
        'building': ['building', 'building blocks'],
        'calm': ['calm', 'calming'],
        'core': ['core', 'core exercises library', 'core series'],
        'deep': ['deep', 'relaxation'],
        'drink': ['drink', 'drinks'],
        'education': ['education', 'educational'],
        'emotional': ['emotional', 'emotional release'],
        'empowerment': ['empowerment', 'empowering'],
        'energy': ['energ', 'energize', 'energizing', 'energetic'],
        'essential': ['essential', 'essentials'],
        'guide': ['guide', 'guided'],
        'healthy': ['healthy', 'healthy eating'],
        'improved': ['improved', 'improving'],
        'inner': ['inner', 'inner peace'],
        'life': ['life', 'lifestyle', 'living'],
        'meal': ['meal', 'meals', 'meal ideas', 'meal planning'],
        'mindf_DSul': ['mindf_DSul', 'mindf_DSulness'],
        'motivation': ['motivation', 'motivational'],
        'nature': ['nature', 'nature sounds'],
        'night': ['night', "night's"],
        'nutrition': ['nutrition', 'nutrients', 'nutritional', 'nutritious'],
        'peace': ['peace', 'peaceful', 'peacefully'],
        'positive': ['positive', 'positivity'],
        'practice': ['practice', 'practices', 'practicing'],
        'preparation': ['prep', 'preparation', 'preparing', 'prepping'],
        'protein': ['protein', 'proteins'],
        'quick': ['quick', 'quick and easy'],
        'recipe': ['recipe', 'recipes'],
        'relax': ['relax', 'relaxation'],
        'serene': ['serene', 'serenity'],
        'sounds': ['sounds', 'soundly'],
        'strength': ['strength', 'strenght'],
        'stress': ['stress', 'stress relief', 'stress management'],
        'thai stretch': ['thai stretching', 'thai stretching classes'],
        'tranquil': ['tranquil', 'tranquility'],
        'vegan': ['vegan', 'veggie'],
        'yoga': ['yoga', 'yoga foundations', 'yoga meditation', 'yoga office workout', 'yoga poses library', 'yoga series']
    }
    
    tags_column = lowercase_df['Tags']
    lowercase_df['Tags'] = replace_tags(tags_column, tags_dict)

    new_tags_column = lowercase_df['Tags']
    new_clean_tags_column = clean_and_tokenize_tags(new_tags_column)
    
    # Remove duplicates within each row of Tags column
    lowercase_df['Tags'] = lowercase_df['Tags'].apply(lambda x: ', '.join(sorted(set(str(x).split(',')), key=str(x).split(',').index)) if pd.notnull(x) else x)
    new_clean_tags_column = set(new_clean_tags_column)
    
    # Apply the capitalize_text function to the 'Text' column
    final_df = capitalize_final_data(lowercase_df)
    
    # Save the final dataframe to a csv file
    final_df.to_csv('data/GoE_Dataset.csv', index=False)
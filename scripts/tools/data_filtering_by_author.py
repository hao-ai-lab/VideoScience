import csv

AUTHOR = 'Murray' # change this to author you want to review

def filter_author_entries(input_file, output_file):
    """
    Filter CSV entries where Author is Murray and extract Prompts and Expected phenomenon columns.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    author_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if row.get('Author', '').strip() == 'AUTHOR' and row.get('finalized?', '').strip() == 'Done':
                filtered_row = {
                    'Prompts': row.get('Prompts', ''),
                    'Expected phenomenon': row.get('Expected phenomenon', '')
                }
                author_entries.append(filtered_row)
    
    # Write filtered entries to new CSV
    if author_entries:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['Prompts', 'Expected phenomenon']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(author_entries)
        
        print(f"Successfully filtered {len(author_entries)} {AUTHOR} entries to {output_file}")
    else:
        print(f"No entries found for author {AUTHOR}")

if __name__ == '__main__':
    input_file = '/workspace/ScienceAtlas/whole_database.csv'
    output_file = '/workspace/ScienceAtlas/test.csv'
    
    filter_author_entries(input_file, output_file)


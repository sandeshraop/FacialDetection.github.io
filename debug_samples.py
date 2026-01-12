import pandas as pd
from pathlib import Path

# Load Excel data
df = pd.read_excel('data/processed/Cropped/CASME2-coding-20140508.xlsx')
valid_emotions = {'happiness', 'disgust', 'surprise', 'repression'}
filtered = df[df['Estimated Emotion'].isin(valid_emotions)]

# Get ROI data samples
roi_dir = Path('data/processed/roi_optical_flow')
roi_samples = []
for subject_dir in roi_dir.iterdir():
    if subject_dir.is_dir() and subject_dir.name.startswith('sub'):
        for video_dir in subject_dir.iterdir():
            if video_dir.is_dir() and (video_dir / 'stacked_flow.npy').exists():
                roi_samples.append(f'{subject_dir.name}/{video_dir.name}')

print(f'ROI samples found: {len(roi_samples)}')
print('First 10 ROI samples:', roi_samples[:10])

# Check matches
excel_samples = []
for _, row in filtered.iterrows():
    filename = str(row['Filename'])
    if not filename.endswith('f'):
        filename = filename + 'f'
    sample_name = f'sub{row["Subject"]:02d}/{filename}'
    excel_samples.append(sample_name)

print(f'Excel samples: {len(excel_samples)}')
print('First 10 Excel samples:', excel_samples[:10])

# Find matches
matches = set(roi_samples) & set(excel_samples)
print(f'Matches: {len(matches)}')
print('Sample matches:', list(matches)[:5])

# Check emotions in matches
match_emotions = {}
for sample in matches:
    subject, video = sample.split('/')
    # Find corresponding emotion
    for _, row in filtered.iterrows():
        filename = str(row['Filename'])
        if not filename.endswith('f'):
            filename = filename + 'f'
        if f'sub{row["Subject"]:02d}/{filename}' == sample:
            match_emotions[row['Estimated Emotion']] = match_emotions.get(row['Estimated Emotion'], 0) + 1
            break

print('Emotions in matches:', match_emotions)

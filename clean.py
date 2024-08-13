import pandas as pd

data = pd.read_csv('data_utf8.csv', on_bad_lines='skip', delimiter=';')

# Drop columns that are not needed
data = data.drop(columns=['Group', 'KTAS_RN', 'Disposition', 'Error_group', 'KTAS duration_min', 'mistriage', 'Length of stay_min'])

# Decode columns with number values
data['Sex'] = data['Sex'].replace({1: 0, 2: 1})
data['Arrival mode'] = data['Arrival mode'].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6})
data['Injury'] = data['Injury'].replace({1: 0, 2: 1})
data['Mental'] = data['Mental'].replace({1: 0, 2: 1, 3: 2, 4: 3})
data['Pain'] = data['Pain'].replace({2: 0})

# Convert relevant columns to numeric values
data['SBP'] = pd.to_numeric(data['SBP'], errors='coerce')
data['DBP'] = pd.to_numeric(data['DBP'], errors='coerce')
data['HR'] = pd.to_numeric(data['HR'], errors='coerce')
data['RR'] = pd.to_numeric(data['RR'], errors='coerce')
data['BT'] = pd.to_numeric(data['BT'], errors='coerce')
data['Saturation'] = pd.to_numeric(data['Saturation'], errors='coerce')
data['NRS_pain'] = pd.to_numeric(data['NRS_pain'], errors='coerce')

# Replace NaN values with "normal" values for each category
data['SBP'].fillna(110, inplace=True)  # Normal SBP
data['DBP'].fillna(70, inplace=True)   # Normal DBP
data['HR'].fillna(75, inplace=True)    # Normal HR
data['RR'].fillna(16, inplace=True)    # Normal RR
data['BT'].fillna(37, inplace=True)    # Normal BT
data['Saturation'].fillna(98, inplace=True)  # Normal Saturation
data['NRS_pain'].fillna(0, inplace=True)  # No pain

# Making brackets for Blood Pressure
def get_bp_category(row):
    sbp = row['SBP']
    dbp = row['DBP']
    if sbp < 90 and dbp < 60:
        return 0 # Hypotension
    if sbp < 120 and dbp < 80:
        return 0  # Normal
    elif 120 <= sbp <= 129 and dbp < 80:
        return 1  # Elevated
    elif 130 <= sbp <= 139 or 80 <= dbp <= 89:
        return 2  # High Blood Pressure (Hypertension) Stage 1
    elif sbp >= 140 or dbp >= 90:
        if sbp > 180 or dbp > 120:
            return 4  # Hypertensive Crisis
        return 3  # High Blood Pressure (Hypertension) Stage 2
    else:
        return 4  # Hypertensive Crisis
data['BP'] = data.apply(get_bp_category, axis=1)
sbp_index = data.columns.get_loc('SBP')
data.drop(['SBP', 'DBP'], axis=1, inplace=True)
data.insert(sbp_index, 'BP', data.pop('BP'))

# Making brackets for Heart Rate
def replace_hr(row):
    hr = row['HR']
    if hr < 40:
        return 0  
    elif 40 <= hr <= 50:
        return 1  
    elif 51 <= hr <= 60:
        return 2  
    elif 61 <= hr <= 100:
        return 3  # Normal
    elif 101 <= hr <= 120:
        return 4
    elif 121 <= hr <= 140:
        return 5
    elif 141 <= hr <= 160:
        return 6
    else:
        return 7
data['HR'] = data.apply(replace_hr, axis=1)

# Making brackets for Respiration Rate
def replace_rr(row):
    rr = row['RR']
    if rr < 8:
        return 0  
    elif 8 <= rr <= 11:
        return 1  
    elif 12 <= rr <= 20:
        return 2  # Normal
    elif 21 <= rr <= 24:
        return 3  
    elif 25 <= rr <= 30:
        return 4
    else:
        return 5
data['RR'] = data.apply(replace_rr, axis=1)

# Making brackets for Body Temperature
def replace_bt(row):
    bt = row['BT']
    if bt < 35:
        return 0  # Hypothermia
    elif 35 <= bt <= 36.4:
        return 1  
    elif 36.5 <= bt <= 37.5:
        return 2  # Normal
    elif 37.6 <= bt <= 38.3:
        return 3  # Low-grade Fever
    elif 38.4 <= bt <= 40:
        return 4 # Hyperthermia
    else:
        return 5 # Hyperpyrexia
data['BT'] = data.apply(replace_bt, axis=1)

# Making brackets for Oxygen Saturation
def replace_o2(row):
    o2 = row['Saturation']
    if o2 < 90:
        return 0  # Hypoxemia
    elif 90 <= o2 <= 94:
        return 1  
    elif (95 <= o2 <= 100) or o2 == 'NaN':
        return 2  # Normal
data['Saturation'] = data.apply(replace_o2, axis=1)

# Making brackets for Pain Scale
def replace_nrs(row):
    nrs = row['NRS_pain']
    if nrs == 0:
        return 0  # No pain
    elif 1 <= nrs <= 3:
        return 1  # Mild pain
    elif 4 <= nrs <= 6:
        return 2  # Moderate pain
    elif 7 <= nrs <= 10:
        return 3  # Severe pain
data['NRS_pain'] = data.apply(replace_nrs, axis=1)

print(data.head(10))
data.to_csv('data_cleaned.csv', index=False)
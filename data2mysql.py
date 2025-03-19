with open('/Volumes/PASSPORT/FinalYearProject/MCANETRUN/Davis.txt', 'r') as f:
    lines = f.readlines()

with open('/Volumes/PASSPORT/FinalYearProject/MCANETRUN/KIBA.txt', 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    parts = line.strip().split(' ', 4)
    if len(parts) == 5:
        compound_id, protein_name, smiles, rest = parts[0], parts[1], parts[2], parts[3] + ' ' + parts[4]
        sequence, label = rest.rsplit(' ', 1)
        data.append({
            'compound_id': compound_id,
            'protein_name': protein_name,
            'smiles': smiles,
            'sequence': sequence,
            'label': int(label)
        })

import pymysql

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='root',
                             db='DTI',
                             charset='utf8mb4')

cursor = connection.cursor()

cursor.executemany('INSERT INTO dataset (compound_id, protein_name, smiles, sequence, label) '
                   'VALUES (%(compound_id)s, %(protein_name)s, %(smiles)s, %(sequence)s, %(label)s)',
                   data)

connection.commit()
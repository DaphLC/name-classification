import requests
import json
import time
import bs4

def scrap(current_page):
    '''
    Parse & scrap one webpage
    '''
    response = requests.get(f"https://adoption.com/baby-names/browse?page={current_page}")
    soup = bs4.BeautifulSoup(response.text, "lxml")
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")
    headers = [th.text for th in rows[0].find_all("th")]
    names = [
        {
        h: td.text
        for td, h in zip(row.find_all("td"), headers)
        }
    for row in rows[1:] if len(row.find_all('td')) == 5
    ]
    return names

headers = ['Name', 'Meaning', 'Gender', 'Origin', 'Similar']

# parse & scrap webpages
names = list()
for current_page in range(1, 1442):
    time.sleep(3) 
    names.extend(scrap(current_page))
    if current_page % 50 == 0:
        print('Page:', current_page)

# clean data
new_names = list()
for name in names:
    origins = name['Origin'].split(', ')
    for o in origins:
        new_names.append({
            'Name': name['Name'],
            'Meaning': name['Meaning'],
            'Gender': name['Gender'],
            'Origin': o
        })
    
# save dataset as json file    
with open('cleaned_names.jsonl', mode='w', encoding='utf8') as f:
    for name in new_names:
        f.write(f'{json.dumps(name)}\n')
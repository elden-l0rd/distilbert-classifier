################################################
## Extracts mapping of CAPEC to MITRE ATT&CKS ##
################################################

import xml.etree.ElementTree as ET
tree = ET.parse('data/external/raw/658.xml')
root = tree.getroot()

print("Root tag:", root.tag)

with open('data/results/capec-mitre.txt', 'w') as file:
    for attack_pattern in root.findall('.//ns:Attack_Pattern', namespaces={'ns': "http://capec.mitre.org/capec-3"}):
        capec_id = attack_pattern.get('ID')
        if not capec_id:
            print("No CAPEC ID found")
        else:
            print(f"Found CAPEC ID: {capec_id}")

        mappings = attack_pattern.findall(".//ns:Taxonomy_Mapping[@Taxonomy_Name='ATTACK']", namespaces={'ns': "http://capec.mitre.org/capec-3"})
        for mapping in mappings:
            entry_id = mapping.find('ns:Entry_ID', namespaces={'ns': "http://capec.mitre.org/capec-3"}).text
            if entry_id:
                file.write(f"CAPEC ID: {capec_id}, Entry ID: {entry_id}\n")
            else:
                print("No Entry ID found for CAPEC ID:", capec_id)

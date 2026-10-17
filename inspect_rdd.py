import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

base = Path(
    "data/raw/RDD2022/"
    "RDD2022_released_through_CRDDC2022/"
    "RDD2022/India"
)

counts = Counter()
annotated_files = 0

xml_files = list(base.rglob("*.xml"))

print(f"Scanning {len(xml_files)} XML files...")

for xml_file in xml_files:
    root = ET.parse(xml_file).getroot()
    objects = root.findall("object")

    if objects:
        annotated_files += 1

    for obj in objects:
        name = obj.findtext("name")

        if name:
            counts[name] += 1

print()
print("XML files with annotations:", annotated_files)
print("Total annotated objects:", sum(counts.values()))

print()
print("Damage classes:")

for name, count in sorted(counts.items()):
    print(f"{name}: {count}")
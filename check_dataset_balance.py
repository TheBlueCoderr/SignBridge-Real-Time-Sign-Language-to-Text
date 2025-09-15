import os

base_dir = "../data/ISL_Dataset"

print("🧾 Class-wise image counts:\n")
for label in sorted(os.listdir(base_dir)):
    label_path = os.path.join(base_dir, label)
    if os.path.isdir(label_path):
        count = len(os.listdir(label_path))
        print(f"{label}: {count} images")

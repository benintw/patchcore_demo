An implementation of paper:
https://arxiv.org/pdf/2106.08265v2
from basically from someone's github

link to entire mvtec data:
https://www.mvtec.com/company/research/datasets/mvtec-ad

# HOW TO RUN

1. Download the mvtec dataset from the above link
2. Place the folder in the same directory as the main.py, so that the project directory looks like:
   .
   ├── README.md
   ├── config.py
   ├── evaluation.py
   ├── main.py
   ├── memory_bank.py
   ├── mvtec ## dataset
   ├── prediction.py
   ├── requirements.txt
   ├── resnet_feature_extractor.py
   └── utils.py

3. Pick a SUBJECT (eg. transistor, carpet, pill, .. etc)
4. Pick a DEFECT_TYPE (eg. "bent_lead" in SUBJECT carpet)
   \*\* the code can be later modified to detect all defect types

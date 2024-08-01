# GNN for Mixture Similarity (GMS)
This is a GNN that can take two mixtures of odors and predict how similar they are to humans.

## To train a model
Update the src/GMS/models/config.json training parameters if desired
```
git clone https://github.com/GrantMcConachie/GMS.git
cd GMS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.GMS.scripts.train
```
Model parameters, config, and loss will save in the folder specified in src/GMS/scripts/train.py

## To test a trained model
go to src/GMS/scripts/test.py and change filepath to desired model
```
python -m src.GMS.scripts.test
```
a `preds.csv` file will populate in the filepath specified

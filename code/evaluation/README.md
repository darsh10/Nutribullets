# KG Evaluation

We use the relation extraction models to compare the knowledge structures of the model outputs with respect to the human written summaries.

The entity tagging dataset is used to train an entity extraction model using this repository. https://github.com/jiesutd/NCRFpp

We share the relation extraction model ckpts trained using BERT https://arxiv.org/abs/1810.04805.
The causes relation model is here https://drive.google.com/file/d/19EDWDDmDxOK-mVfyxw8nC27SQL9rc2H-/view?usp=sharing and the contains relation model is here https://drive.google.com/file/d/1WKm5WfVR79AiCvH13y983-luBLiKahAI/view?usp=sharing.
The bert based model to train and run this model is run_re.py.

The file which calls these models and runs this evaluation is create_kg.py

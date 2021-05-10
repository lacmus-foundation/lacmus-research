# Neural Network Research

Section with ml researches

### Table of contents

ModelsMetrics.md - summary of training with different settings 


test_torch_SDD.ipynb - 
	re-implementation of solution with torchvision based on torchvision.models.detection.retinanet and tutorial for SDD dataset (pre-training)

test_torch_LADD.ipynb - 
	training for Lacmus dataset (dataset is merged from seasonal datasets spring_korolev_2019, summer_moscow_2019, summer_tambov_2019, winter_moscow_2018)


functions_torch.py
	This contains classes, taken from tutorial at https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html without any serious adaptation (copied, not to import tutorial git repo, which is not garanteed to sustain reverse compatibility). Probably significant part of code is redundant

functions.py, compute_overlap.pyx - evaluation mAp the same way as in original retinanet (to keep models comparable)



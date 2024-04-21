# BN-Refinement

This repository contains code and supplementary material for the paper "Modeling multiple adverse pregnancy outcomes:\\ Learning from diverse data sources" AIME (2024).

- `Eliciting initial BN structure from LLM.pdf` contains the prompt and the response from Google Gemini that was used to construct the initial BN.
- `black_list.csv` contains the list of temporally impossible edges. These edges are excluded from the local search used to refine the initial BN.
- `structure_learning.py` is a module that implements the BN refinement procedure.
- `refine.py` reads an initial BN structure and a data set from disk and calls the refinement procedure to obtain a refined BN.

## Requirements 
- pgmpy==0.1.24
- scikit-learn==1.1.3


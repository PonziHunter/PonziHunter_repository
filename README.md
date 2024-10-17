## PonziHunter

This repository contains the source code of the paper "PonziHunter: Hunting Ethereum Ponzi Contract via Static Analysis and Contrastive Learning on the Bytecode Level". PonziHunter is a contrastive learning method based on smart contract bytecode, which models the control flows and state variable dependencies of the programs and locates the key basic blocks related to the Ponzi scheme. 


### 🚀 Getting started

PonziHunter leverages Gigahorse to lift the contract bytecode to TAC-based IR. Run on this command as follows to obtain TAC files:
```
cd ./data
python3 run_gigahorse.py
```

Then, PonziHunter constructs cross-function CFGs based on TAC. Run on these commands to generate CFG files (In this step, PoniHunter also applies code slicing algorithm and calculates the important scores):
```
cd ./algorithm
python3 train_doc2vec.py
python3 dataset.py
```

Next, PonziHunter uses a contrastive learning mechanism to pre-train the graph encoder:
```
python3 pretrain_graph_encoder.py
```

Last, run on the following command for the downstream classification task:
```
python3 train_downstream_classifier.py --split=split1
```


### 📚 Dataset
The dataset files will be made public after the paper is accepted. Waiting for updates.

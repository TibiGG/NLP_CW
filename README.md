# NLP_CW

## Generate Virtual Environment
This task is useful both for running the main script on either your computer or the departmental GPU cluster.

1. Install anaconda3 on the machine of your choice.
For DoC GPU, ensure you install anaconda in `/vol/bitbucket/<your_username>`

2. Generate the new environment:

```bash
conda create -n nlp python=3.8
conda activate nlp
conda install -c pytorch pytorch 
conda install cudatoolkit=11.0
conda install pandas tqdm
```

3. Install rest of dependencies with pip
```bash
pip install simpletransformers
pip install tensorboardx
```

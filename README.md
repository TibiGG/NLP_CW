# NLP_CW

## File to run:
As mentioned in the coursework specification, the entire project should be runnable from one file.
Please, then, run the following file as a JuPyter notebook:
```
submission_file.ipynb
```

## Generate Virtual Environment

This task is useful both for running the main script on either your computer or
the departmental GPU cluster.

0. SSH into a DoC machine:

```bash
ssh -XY <username>@gpuXX.doc.ic.ac.uk
```

1. Install anaconda3 on the machine of your choice. For DoC GPU, ensure you
   install anaconda in `/vol/bitbucket/<your_username>`.

    You may follow this link for DoC [tutorial](https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/support/applications/conda/) for how-to.


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

4. Clone this repo into your `/vol/bitbucket` directory:

```bash
cd /vol/bitbucket/<username>
git clone <url_of_this_dir>
```

5. SSH into the gpu cluster
```bash
ssh gpucluster.doc.ic.ac.uk
```

6. Run shell script from repo in your home directory. This will allow you to see the output of the shell script in your home directory after the end of the program's runtime.
```bash
sbatch /vol/bitbucket/<your_username>/NLP_CW/trainer.sh 
squeue # see if your job is running
```


And that's about it!

For any other queries, you should find your answer on the [DoC GPU guide](https://www.imperial.ac.uk/computing/csg/guides/hpcomputing/gpucluster/).

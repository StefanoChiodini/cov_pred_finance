import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Function to run a notebook
def run_notebook(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})

# Paths to your notebooks
notebooks = ['experiments/rwPerformance.ipynb']

for notebook in notebooks:
    run_notebook(notebook)

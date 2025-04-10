# Using Jupyter Notebook with Virtual Environments

This guide explains how to configure Jupyter Notebook to use a specific virtual environment.

## Prerequisites

- Python installed on your system
- Basic knowledge of virtual environments
- Jupyter Notebook installed (can be installed via `pip install notebook`)

## Step-by-Step Instructions

### 1. Activate Your Virtual Environment

#### For venv/virtualenv:

```bash
# On Linux/Mac
source myenv/bin/activate

# On Windows
myenv\Scripts\activate
```

#### For Conda:

```bash
conda activate myenv
```

### 2. Install ipykernel

Once your virtual environment is activated, install the `ipykernel` package:

```bash
pip install ipykernel
```

### 3. Register Your Environment as a Jupyter Kernel

Run the following command to make your virtual environment available as a kernel in Jupyter:

```bash
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
```

**Note:** Replace `myenv` with your virtual environment's name and `Python (myenv)` with whatever display name you want to show in Jupyter.

### 4. Start Jupyter Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

### 5. Select Your Kernel

When creating a new notebook, select your virtual environment from the "New" dropdown menu.

To switch kernels in an existing notebook:
1. Click on "Kernel" in the menu
2. Select "Change kernel"
3. Choose your virtual environment from the list

## Additional Tips

- To list all available kernels: `jupyter kernelspec list`
- To remove a kernel: `jupyter kernelspec uninstall myenv`
- If you update packages in your virtual environment, they'll automatically be available in your Jupyter kernel

## Troubleshooting

- If your kernel doesn't appear, ensure the ipykernel installation completed successfully
- If you receive import errors, verify the package is installed in the correct virtual environment
- For permission errors during kernel installation, try running the command with administrator privileges
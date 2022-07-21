# Introduction to Version Control with Git

## What do you need to run your Python code?
- Your code: Some files with .py extension
- Python package: Libraries that contains many functions related to each other (e.g. numpy, scipy, pandas, scikit-learn)
- To run your code, you need to define its environment: Version of Python + Some packages + Version of packages

## When do you need to know your environment?
- You may have on your computer different Python codes with different versions of packages
- You give your code to a friend
- Some of your packages may depend on other packages, with a specific version. How to make sure you have the right version of everything?

## How to deal with this?
Install [anaconda](https://www.anaconda.com/products/individual).

* User interface + command line
* Tools for developing code in Python: JupyterLab, Spyder
* Jupyter notebook (more on this later)
* Tools for managing environment

### Basic conda commands
`conda info`

Check conda version to make sure its installed.

`env list`

List out available environments (the starred * environment is the current activate environment).

`conda env create --file environment.yml`

Create conda environment from environment file.

`conda env remove --yes --name myenv`

Removing the conda environment.

`conda activate myenv`

Activate a conda enivronment by name.

`conda deactivate`

Deactivate the current cona environment.

<br>

### Example of .yml file
    name: MLlabs
        channels:
          - conda-forge
          - defaults
        dependencies:
          - python=3.9
          - jupyter
          - matplotlib
          - numpy
          - pandas
          - scipy
          - scikit-learn
          - pytorch

To learn more about Anaconda environments, visit [this site](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Jupyter notebooks

### How to explain what your code is doing?
A Jupyter notebook allows you to merge text, images, code, and code output.

### What is a Jupyter notebook made of?
* Markdown cells for the text
* Code cells for the code
* Kernel:
  - You can run the code cells one by one
  - Run all the cells until the end
  - Restart the kernel i.e. delete all the variables that you've created so far

## Markdown

### What is Markdown?
A simple language for text formatting.

Used for:
* Text in Jupyter notebooks
* Text on .md files on GitHub (e.g. README.md in a GitHub repo)
* Text on RStudio files

### Basic Markdown commands
Headings <br>
***
\# Heading level 1 <br>
\#\# Heading level 2 <br>
\#\#\# Heading level 3 <br>
***

Paragraphs: Leave a blank line
***
This is my first paragraph.

This is my second paragraph.
***

Line break: use `<br>`<br>
***
This is my first line.<br>
This is my second line.
***

Bold text<br>
***
\*\*This is my bold text\*\*
***

Italic Text<br>
***
\*This is my italic text\*
***

Bold and italic text
***
\*\*\*This is my bold, italic text\*\*\*
***

Ordered list
***
1. First item
2. Second item
3. Third item
4. Fourth item
***

Unordered list
***
\- First item <br>
\- Second item <br>
\- Third item <br>
\- Fourth item <br>
***

Images: use `<img src="images/glass.png" width="200"/>`

<img src="glass.png" width="200"/>
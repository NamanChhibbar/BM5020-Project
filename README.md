# AI in Biomedicine and Healthcare [BM5020]

## Tissue Recognition during Third-Space Endoscopy

### Team Members

- **Pradeep Mundlik** [AI21BETCH11022](mailto:ai21btech11022@iith.ac.in)
- **Naman Chhibbar** [MA21BTECH11011](mailto:ma21btech11011@iith.ac.in)

[Final Presentation](https://docs.google.com/presentation/d/1PxlyDp7aWtlnVhDewr-FX4nmOz7pDEPEOOLXaoyQ4wA/edit?usp=sharing)

[Github](https://github.com/NamanChhibbar/BM5020-Project)

## Repository structure

1. **main.ipynb**: Notebook containing model training and evaluation.
2. **annotations.ipynb**: Notebook used for annotating images to generate pixel labels for segmentation.
3. **utils.py**: Python file containing utility functions.
4. **Report**: This folder contains `latex` code and PDF file for report. Refer to `main.pdf` and `main.tex`.
5. **requirements.txt**: File which includes technical specfications of all modules used.

## Installing the dependencies

A Python virtual environment is highly recommended for installing the dependencies. Create one by running

```sh
python3 -m venv .venv
```

Activate the environment

```sh
source .venv/bin/activate
```

Install the dependencies

```sh
pip3 install -r requirements.txt
```

To deactivate the environment

```sh
deactivate
```

## Data structure

Images must be stored in jpg format and the labels in a pickle file containing a python dictionary which maps the image file names to a `torch.tensor` of size `H x W`, where `H` and `W` the height and width of the image respectively. The `i, j`th value of the tensors should correspond to the label of `i, j`th pixel in the image.

Both the images and pickle file should be stored in a directory whose location is stored in the variable `im_dir` in [main.ipynb](main.ipynb). Update these before training.

## Generating labels using Segment Anything Model (SAM)

[This file](annotation.ipynb) can be used to generate pixel labels via the SAM model. Make sure SAM checkpoint is downloaded in your system; its path is stored in `sam_checkpoint`.

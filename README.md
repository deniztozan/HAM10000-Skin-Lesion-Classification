# HAM10000 Skin Lesion Analysis and Classification

I'm using this repository to explore how well different machine learning approaches can classify skin lesions from dermoscopic images. The plan is to start with the basics—understanding the dataset, checking data quality, and looking at the images—before moving on to image preprocessing, HOG feature extraction, and a Random Forest baseline.

The project is a work in progress. I'll keep the notebooks, code, and results updated as I experiment with new methods.

> **For research and learning only.** This is not a clinical diagnostic tool, and its predictions should not be used to make medical decisions.

## The dataset

I'm working with the **HAM10000 Segmentation and Classification** dataset from Kaggle:

**[Download the dataset on Kaggle](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)**

It contains **10,015 dermoscopic images** across seven diagnostic categories. The version used here brings together three things:

- **`images/`** — the original skin lesion images.
- **`masks/`** — segmentation masks outlining the lesion areas.
- **`GroundTruth.csv`** — the image IDs and their diagnostic labels, stored in one-hot format.

For example, a row with `NV = 1` and all other class columns set to `0` is labelled as a melanocytic nevus. Before converting those columns into a single target label, the data loader checks that every row has exactly one valid class.

### Classes and image counts

| Class | Description | Images | Share |
| --- | --- | ---: | ---: |
| NV | Melanocytic nevi | 6,705 | 66.95% |
| MEL | Melanoma | 1,113 | 11.11% |
| BKL | Benign keratosis-like lesions | 1,099 | 10.97% |
| BCC | Basal cell carcinoma | 514 | 5.13% |
| AKIEC | Actinic keratoses / intraepithelial carcinoma | 327 | 3.27% |
| VASC | Vascular lesions | 142 | 1.42% |
| DF | Dermatofibroma | 115 | 1.15% |

One of the first things I noticed during EDA was the **class imbalance**: about two-thirds of the images belong to the NV class, while DF accounts for just over 1%. A classifier that always predicts NV would already reach roughly 67% accuracy, so accuracy on its own won't tell the whole story. I'll also look at **macro F1, per-class recall, and confusion matrices**—especially errors involving melanoma.

## Where the project stands

So far, I've:

- Set up reusable Python modules and separate notebooks.
- Added checks for missing or duplicate image IDs, invalid labels, and missing image files.
- Linked image IDs to their corresponding images and available segmentation masks.
- Examined class frequencies and displayed sample images from all seven classes.

Next, I'll finish exploring image sizes, visual variation, and mask coverage. After that, the first modelling experiment will use **HOG features with Random Forest**. I also want to compare results from original images and images where the background has been masked out.

### Roadmap

- [x] Organize the project into shared modules and notebooks
- [x] Load and validate `GroundTruth.csv`
- [x] Explore class distribution and sample images
- [ ] Finish EDA: image dimensions, mask coverage, and image quality
- [ ] Create reproducible train/validation/test splits
- [ ] Build and compare preprocessing options
- [ ] Extract HOG features
- [ ] Train and evaluate a Random Forest baseline
- [ ] Compare original images with ground-truth-masked images
- [ ] Explore other models or segmentation approaches

Completed items reflect the work done so far, not claims about model performance.

## Project structure

The repository is organized around a simple idea: **notebooks are for exploration; `src/` is for code I want to reuse.** Each notebook loads what it needs through shared modules rather than relying on variables left in another notebook's memory.

```text
Health/
├── images/                       # Downloaded dataset; not tracked by Git
├── masks/                        # Downloaded segmentation masks; not tracked
├── GroundTruth.csv               # Downloaded labels; not tracked
├── src/
│   ├── __init__.py
│   ├── config.py                  # Shared paths and class names
│   ├── data_loader.py             # Dataset loading and validation
│   ├── preprocessing.py           # Planned
│   ├── features.py                # Planned: HOG feature extraction
│   └── evaluation.py              # Planned: model evaluation helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb      # Planned
│   ├── 04_hog_features.ipynb      # Planned
│   └── 05_random_forest.ipynb     # Planned
├── outputs/                       # Local generated files; not tracked
├── README.md
└── .gitignore
```

This is the **intended structure**; some files are still on the roadmap.

## Getting started

1. Clone the repository and open the project in VS Code.
2. Download the dataset from the [Kaggle page](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification).
3. Place `images/`, `masks/`, and `GroundTruth.csv` in the project root as shown above.
4. Select your Python environment (I use a Conda environment) and the matching Jupyter kernel.
5. Install the libraries used for the current stage:

   ```bash
   python -m pip install numpy pandas matplotlib scikit-learn scikit-image opencv-python jupyter ipykernel
   ```

6. Open `notebooks/02_eda.ipynb` to explore the dataset.

The image files, masks, and CSV are excluded via `.gitignore`, so you'll need to download them locally. I may add a pinned environment file once the dependencies settle.

### Loading data in a notebook

From a notebook running with `Health/notebooks` as its working directory:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSES
from src.data_loader import load_dataset

df = load_dataset()
display(df.head())
```

If the import fails, check `Path.cwd()` first—the working directory may differ depending on how you launched the notebook.

## A few things I'm keeping in mind

- **Avoiding data leakage:** Related images of the same lesion or patient shouldn't end up on both sides of a train/test split if identifiers are available. Preprocessing steps that learn parameters must be fitted on training data only.
- **Class imbalance:** Overall accuracy can hide poor results on smaller classes. The class-by-class errors matter.
- **Segmentation masks:** Using provided ground-truth masks makes for a useful experiment, but it assumes the mask is available at prediction time. A real end-to-end workflow would need to generate it.
- **Reproducibility:** I'll record random seeds, splitting decisions, preprocessing settings, model parameters, and metrics as the experiments progress.

## Sources and acknowledgements

This work builds on the dataset and research made available by the original authors and contributors:

- **Kaggle dataset used in this project:** [HAM10000 — Segmentation and Classification](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)
- **Original HAM10000 dataset:** [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* [Scientific Data](https://doi.org/10.1038/sdata.2018.161)
- Tschandl, P., Rinner, C., Apalla, Z., et al. (2020). *Human–computer collaboration for skin cancer recognition.* [Nature Medicine](https://doi.org/10.1038/s41591-020-0942-0)
- [HAM10000 lesion segmentation masks](https://www.kaggle.com/tschandl/ham10000-lesion-segmentations)

Please refer to the source datasets for their licensing and attribution requirements before redistributing any data.

---

*Current stage: exploratory data analysis. I'll update this README as the project grows.*

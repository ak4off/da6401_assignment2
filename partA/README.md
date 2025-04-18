## **Setup**

### **Clone the repository:**  
```bash
   git clone https://github.com/ak4off/da6401_assignment2.git
   cd da6401_assignment2/partA
```

### **Requirements**
```bash
pip install wandb
pip install numpy
pip intall torch torchvision
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

## 📁 Code Organization

```bash
partA/
├── model_cnn.py         # Defines the 5-layer CNN architecture
├── data_loader.py       # Loads iNaturalist data, applies transforms, splits into train/val/test
├── config.yaml          # Stores hyperparameters and W&B sweep config
├── train.py             # Main training loop and validation logic
├── evaluate.py          # Evaluates the best saved model on test data
├── visualize.py         # (Optional) Visualizations: filters, guided backprop, etc.
├── wandb_sweep.py       # Sets up and launches W&B hyperparameter sweeps
├── utils.py             # Helper functions (e.g., accuracy, parameter count)
├── requirements.txt     # Dependency list
└── README.md            # You're here!
```


## **Dataset**
- The dataset used is inaturalist_12k
- The data should be downloaded and present in the machine 
- create a symbolic link (ln -s <path-to-dataset> data) in the pwd

- 20% of Train set is set aside for Validation set ( Question2 ) 
## **Usage**

## **Running the Model**
To train the network
```bash
python train.py --data_dir data/inaturalist_12K 
```
To train and log on wandb
```bash
python wandb_sweep.py
```

## **Results & Logging**
- Training and validation **loss** and **accuracy** are logged per epoch.
- Final **test accuracy** is reported on the separate `val/` folder (used as test set).
- W&B dashboards include accuracy trends, parameter correlations, and sweep summaries.



## **License**
This project is licensed under the MIT License.


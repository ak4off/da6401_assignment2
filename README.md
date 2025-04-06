# da6401_assignment2
A Repository for Assignment2 - CNNs: train from scratch and finetune a pre-trained model as it is.

- [Assignment Link](https://wandb.ai/sivasankar1234/DA6401/reports/DA6401-Assignment-2--VmlldzoxMjAyNjgyNw)  
- [Report Link](https://api.wandb.ai/links/ns24z274-iitm-ac-in/)

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
```
## **Usage**

## **Dataset**
- The dataset used is inaturalist_12k
- The data should be downloaded and present in the machine 
- create a symbolic link (ln -s <path-to-dataset> data) in the pwd

-20% of Train set is set aside for Validation set ( Question2 ) 

## **Running the Model**
To train the network
```bash
python train.py --data_dir data/inaturalist_12K 
```
## **Results & Logging**
- Training loss and accuracy are recorded.
- Validation loss and accuracy are recorded/
- **Test accuracy** is evaluated on the test(val folder) set.

## **License**
This project is licensed under the MIT License.

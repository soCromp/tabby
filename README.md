Based off the [Great](https://github.com/kathrinse/be_great) repository.

## Setup
```python
conda create --name env python=3.11.4
conda activate env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tqdm matplotlib jupyter pandas scikit-learn jupyter
pip install transformers==4.39.3 accelerate
```

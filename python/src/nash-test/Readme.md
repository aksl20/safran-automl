# Neural Architecture Search by Hill Climbing (NASH)

In this directory, three notebooks testing the NASH method can be found. The algorithm is described in: [*Thomas Elsken, Jan-Hendrik Metzen, Frank Hutter, **Simple And Efficient rchitecture Search for Convolutional Neural Networks**, arXiV:1711.04528*](https://arxiv.org/abs/1711.04528).

### Requirements
The following libraries are used:
- **PyTorch**
- **NNI** : a neural architecture search package that includes a crude support for function-preserving graph morphisms
- **Torchviz** : a utility that displays the architecture of a model

This is was a quick test: some functions are hardcoded to use CUDA, so a GPU is also required to run the code without modification. `torch.cuda.is_available()` should return `True`.

### Dataset
Two of the notebooks apply the method on a a problem of textile orientation and defects classification. Data should be downloaded from  [Kaggle](https://www.kaggle.com/dataset/6e98e7fa2ed4166bb10c375c19df2ffb8bbd636687be7ec7338e2fc7fe2ac15e).
Kaggele requires a login to access its website and there is no convenient way to automatically download the data. This has to be done manually; the directory where the files are located should be given to the dataloader, e.g.: `TextureDataset(directory="./textures")`

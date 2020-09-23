# An implementation of NeuReduce models.
1. All models is implemented with Tensorflow and Keras 2.0.
2. The files in the Baseline folder is peer tools used for comparison to our NeuReduce.

## Requirments

- Python==3.x (better 3.7+)
- Tensorflow==1.14.0
- Keras==2.3.1
- Better if you have GPUs

## Dataset

Datasets is uploaded in another zip file. We offered 1 million train data and 10k test data.

## Training

You can find the *train.py* from the folder corresponding to the model you are interested in. Please run the command below in termianl to train the model.

`python train.py`

The training process may take 20 to 40 hours, depending on whether you have a GPU to speed it up.

If you want to change the superparamaters or dataset, you can do it in the top of *main()* function in *train.py*. All trained result will be stored in xxxx_based_result folder.

To perform a prediction, you can do this.

`python predict.py`

The time cost of prediction for one obfuscated expression is small than 1 second.
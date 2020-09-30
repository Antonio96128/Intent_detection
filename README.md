# Intent_detection
We built an attention model to detect intent.

In this notebook we will discuss one of the classic applications of NLP, intent detection. 
We will try to solve this problem using one of the most powerful ideas available for natural language tasks, an attention mechanism on top of a recurrent neural network, RNN.
The dataset we encounter is imbalanced, like most real datasets, but we will encounter that we can fix this issue. 
First that all, we will preprocess the text, because as we know neural networks deal with vectors and matrices, so we need to convert our word into vectors.
How are we going to do that? Well, there are a few approaches but we will use something called word embeddings because these have proved to
encapsulate the meaning of words in a superior manner. In fact, it was discovered that this technique could even handle analogies! This is kind of a proof that
real relations between word are being learned.
Later, we will train an bidirectional RNN (GRU) using Pytorch. You may discover from reading the class "Attention_Model" that you can 
use LSTM RNN's changing less than a line of code. Also, you can see how the loss function was weighted in order to deal with the imbalanced dataset.
Finally, we implement the attention mechanism on the outputs of the RNN. One thing to notice is that the implementation uses a GPU, but you can turn it off setting 
is_GPU_ON = False. I do not recommend this, rather, use Google Colab if you cannot use a local GPU.

In the last part we analyze the results. First, we create a confusion matrix for both the training set and the validation set, and notice a pretty decent f1-score for the 
8 different classes. Also, we achieve a high accuracy, althought this metric is not the most relevant for imabalanced datasets.
Finally, we focus on one of the most exciting things related to attention models. We can actually see what words are the relevant ones to make a classification of a given sentence!

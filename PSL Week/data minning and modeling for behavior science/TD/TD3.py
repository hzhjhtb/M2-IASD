####################################################################
# This TD is using glove to explore :
#  - Word embeddings
#  - euclidian distance / cosine similarity
#  - Similarity matrix
#  - Hierarchical clustering
#  - Multi dimensional scaling
#  - PCA
#  - k-means



################# ################# #################
################# Import packages
################# ################# #################
import numpy as np
import torchtext
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


################# ################# #################
################# Download glove
################# ################# #################
glove = torchtext.vocab.GloVe(name="6B", dim=100)


################# ################# #################
################# Word representation
################# ################# #################
# Q1 Look at the representation of the word 'lamp', 'candle'
A = XXXXXXXXX
B = XXXXXXXXX

# Q2 measure the euclidian dissimilarity between the two words
euclidian  = XXXXXXXXX

# Q3 measure the cosine similarity between the two words
cosine = XXXXXXXXX

# Q4 Redo the same for 'mouse' and 'lamp'
A = XXXXXXXXX
B = XXXXXXXXX
euclidian  = XXXXXXXXX
cosine = XXXXXXXXX

#Q5 What can you conclude ?



# Q6 using cosine similarity, find the 10 closest words to the word lamp
A = XXXXXXXXX
allwords = np.array(glove.itos)
Similarity = []
for word in allwords:
    B = XXXXXXXXX
    cosine = XXXXXXXXX
    Similarity.append(cosine)

max_index = XXXXXXXXX
allwords[max_index]


# Q7 Write a function that takes as arg the embedding of a word and N and return the N closest words.
def returnclosest(word, n):
    XXXXXXXXX
    return(XXXXXXXXX)

# Q8 Use this function to return the 10 closest words to 'lamp', 'rabbit' & 'bottle'
close = returnclosest(XXXXXXXXX)
close = returnclosest(XXXXXXXXX)
close = returnclosest(XXXXXXXXX)


# Q9 Now that words are vectors, we can move into that space. For example find the embedding of 'queen' - 'woman' + 'man'
emb = XXXXXXXXX

# Q10 Find the closest word to this new embeding
XXXXXXXXX

# Q11 Check that the other relation is true = 'king' - 'man' + 'woman' -> 'queen'
XXXXXXXXX


# Q12 Now let's investigating bias in glove embeddings : look at the 3 closest words to : doctor - man + woman
XXXXXXXXX

# Q13 and do in the other direction : doctor - woman + man
XXXXXXXXX
  # -> Embeddings are just reflecting the bias that can be found in the human made text used for training.




################# ################# #################
################# Dimensionality reduction
################# ################# #################
# Q21 Now let's consider the 20 following words = ['rabbit', 'mouse', 'horse', 'bear', 'monkey','whale', 'dolphin', 'tuna', 'swordfish', 'wolf', 'sofa', 'table', 'chair', 'rug', 'lamp','bag','computer', 'phone', 'keyboard', 'screen']
# Compute the similarity matrix (using cosine similarity) between all those words

words = ['rabbit', 'mouse', 'horse', 'bear', 'monkey','whale', 'dolphin', 'tuna', 'swordfish', 'wolf', 'sofa', 'table', 'chair', 'rug', 'lamp','bag','computer', 'phone', 'keyboard', 'screen']
XXXXXXXXX

# Q22 plot this similarity matrix, do you notice something ?
XXXXXXXXX

# Q23 using a function from seaborn, plot the similarity matrix and its hierarchical clustering
sim = pd.DataFrame(sim) # Convert to pandas dataframe
sim = sim.rename(mapper=pd.Series(words),axis=1) # Add word names as labels
ax = XXXXXXXXX

# Q24 What can you say about this clustering ? Look carefully at the word 'mouse' what do you notice ?




# Q24 perform multi-dimensional scaling on this similarity matrix
mds = MDS(random_state=0) # MDS is implemented sklearn.manifold
scaled_df = XXXXXXXXX

# Q25 plot the results of MDS by showing the position of each word in this space
for i in range(np.shape(words)[0]):
    XXXXXXXXX
    plt.annotate(words[i], (XXXXXXXXX)


# Q26 perform PCA with one dimension on the matrix, and plot the result
pca = XXXXXXXXX
fig = plt.figure()
ax = fig.add_subplot(111)
X_reduced = XXXXXXXXX
for i in range(np.shape(words)[0]):
    ax.scatter(XXXXXXXXX)
    ax.text(XXXXXXXXX, rotation = 35)


# Q27 redo the same with a PCA with 2 dimensions
XXXXXXXXX

# Q28 and now with 3 dimensions
XXXXXXXXX




# Q29 Now compute the similarity matrix for the 800 first words of glove.
words = glove.itos[0:800]
XXXXXXXXX

# Q30 And plot the hierarchical clustering
XXXXXXXXX


# Q31 plot the 2D dimension of this matrix
XXXXXXXXX

# Q32 what can you notice ?

# Q33 using KMeans function from sklearn.cluster cluster the similarity matrix in 6 clusters
kmeans = XXXXXXXXX
kmeans.labels_

# Q34 recompute the 2D PCA and plot it with color corresponding to the cluster
pca = XXXXXXXXX
X_reduced = XXXXXXXXX
colors = sns.color_palette("husl", 6)
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(np.shape(words)[0]):
    ax.scatter(XXXXXXXXX)
    ax.text(XXXXXXXXX)


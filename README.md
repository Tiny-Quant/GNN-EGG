# GNN-EGG: Graph Neural Network Explanations via Graph Generation

## Abstract
Abstract— Graph Neural Networks (GNNs) provide a
means for modeling inherently graphical data, such as
transportation, social, and molecular networks, but also
for enhancing and reducing unstructured data, including
text and images. A potential shortcoming is that their pre-
dictions are opaque — a black box — hindering broader
adoption and refinement. In this paper, we propose a novel
architecture agnostic algorithm, GNN-EGG (Graph Neu-
ral Network Explanations via Graph Generation) for GNN
classifiers. As a model-level post-hoc explanation method,
GNN-EGG can learn the data generating distribution for
each class of graph. The primary contribution of this work
is the use of a differentiable approximation to Graph Edit
Distance (GED) in the loss function. This term enables us
to ensure consistency in both the graph space and the
embedding space for our representative examples. It also
reduce the random baseline issue where completely ran-
dom graphs can still yield similar embeddings and strong
predictions as reported in previous work. We benchmark
our algorithm against the current state-of-the-art models
using the mutagenic molecules dataset (MUTAG), and apply
our method to a large-scale GNN for maligancy detection in
digtal pathology tasks.

Index Terms— Explanations, Graph Edit Distance, Graph
Neural Networks, Model-Level

(Link-to-Paper)[coming_soon]

## Reproducibility 

The MUTAG results can be reproduced by cloning the repo and running the chunk below in the terminal. 

```
cd .devcontainer
docker compose build 
docker compose up 
```

Then simply click the jupyter link and re-run the notebook. 

## Demo Explanation 

## Acknowledgement



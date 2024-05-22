# Self-supervised Contrastive Learning for Heterogeneous Graph with Latent Recurring Pattern Embedding


Recurring patterns, which denote semantic relations frequently occurring among nodes within a heterogeneous graph, encapsulate rich semantic information crucial for modeling semantic relations. Previous approaches have relied on manual design or annotation of recurring patterns, such as meta-paths and motif substructures, to discriminate and focus on homogeneous nodes, limiting their utility in unsupervised graph representation learning.To overcome this limitation, we propose a self-supervised Contrastive Learning framework with Latent Recurring Pattern Embedding (CL-LRPE) to learn semantic relations in heterogeneous graphs and facilitate node discrimination.  Initially, we propose a novel Latent Recurring Pattern Embedding (LRPE) module to capture the common semantic relations therein across the subgraphs in an unsupervised manner and integrate them into node representation.
Subsequently, the LRPE module is integrated into a contrastive learning framework featuring an elaborate graph decoder, enabling the learned representations to not only distinguish nodes but also capture as much information as possible to reconstruct the features and structure of the neighborhood comprehensively. Extensive experiments have demonstrated that our CL-LRPE framework significantly outperforms state-of-the-art unsupervised methods in node classification tasks on heterogeneous graphs. Furthermore, it achieves classification accuracy comparable to some supervised methods.

![](https://github.com/zuiaichirouya/CL-LRPE/blob/main/CL-LRPE.jpeg)

# Preparation
# Installation
A suitable conda environment named CL-LRPE can be created and activated with:

'''bash
conda env create -n CL-LRPE python=3.8.1
conda activate mage
'''

Download the code

'''bash
git clone https://anonymous.4open.science/r/CL-LRPE-631C/
cd CL-LRPE 
'''

Install environment
'''bash
pip install -r requirements.txt
'''

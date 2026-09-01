"""
Sample documents for demonstrating hybrid search in Qdrant.
These documents cover various machine learning topics to showcase
semantic vs. keyword matching capabilities.
"""

DOCUMENTS = [
    "Feature scaling in machine learning is crucial for algorithms like gradient descent and SVM. Common techniques include normalization and standardization to ensure all features contribute equally to the model.",
    
    "Unsupervised learning algorithms discover hidden patterns in data without labeled outputs. Popular methods include K-means clustering, hierarchical clustering, and principal component analysis for dimensionality reduction.",
    
    "Data preprocessing steps are essential before training machine learning models. This includes handling missing values, encoding categorical variables, removing outliers, and splitting data into training and test sets.",
    
    "Neural networks consist of layers of interconnected nodes that process information. Deep learning uses multiple hidden layers to learn hierarchical representations of complex data patterns.",
    
    "Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor generalization to new data. Regularization techniques like L1 and L2 help prevent this.",
    
    "Cross-validation is a resampling technique used to evaluate machine learning models. K-fold cross-validation divides data into k subsets, training on k-1 folds and testing on the remaining fold.",
    
    "Natural language processing enables computers to understand and generate human language. Key tasks include sentiment analysis, named entity recognition, machine translation, and text summarization.",
    
    "Ensemble methods combine multiple models to improve prediction accuracy. Random forests use multiple decision trees, while boosting methods like XGBoost sequentially train models to correct previous errors.",
    
    "Gradient descent is an optimization algorithm that minimizes the loss function by iteratively moving in the direction of steepest descent. Learning rate controls the step size in each iteration.",
    
    "Transfer learning leverages pre-trained models on large datasets and fine-tunes them for specific tasks. This approach is especially effective in computer vision and natural language processing with limited labeled data."
]

# Example queries demonstrating different search scenarios
EXAMPLE_QUERIES = [
    "How to prevent models from memorizing training data?",  # Semantic query (relates to overfitting)
    "gradient descent optimization",  # Keyword query
    "preprocessing data for neural networks"  # Hybrid benefit (semantic + keywords)
]
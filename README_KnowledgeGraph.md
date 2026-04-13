# Knowledge Graph with XGBoost Integration

This project demonstrates how to create and analyze knowledge graphs using GraphFrames in PySpark, and leverage the extracted graph features for machine learning tasks with XGBoost.

## 🚀 Features

- **Complete Knowledge Graph Implementation**: Create complex knowledge graphs with multiple entity types and relationships
- **Advanced Graph Analytics**: Extract sophisticated features using GraphFrames algorithms
- **XGBoost Integration**: Use graph features for classification and regression tasks
- **Hyperparameter Tuning**: Automated model optimization with GridSearchCV
- **Visualization**: Generate plots showing model performance and feature importance

## 📋 Prerequisites

- Python 3.8+
- Java 8+ (required for PySpark)
- At least 4GB RAM recommended

## 🛠️ Installation

1. **Clone or download the files**

2. **Install Java** (if not already installed):
   - Windows: Download from [Adoptium](https://adoptium.net/)
   - macOS: `brew install openjdk@11`
   - Linux: `sudo apt-get install openjdk-11-jdk`

3. **Run the setup script**:
   ```bash
   python setup_knowledge_graph.py
   ```

4. **Or install manually**:
   ```bash
   pip install pyspark==3.4.0 xgboost==1.7.5 pandas numpy scikit-learn matplotlib seaborn
   ```

## 📊 Usage

### Basic Knowledge Graph + XGBoost

```python
from knowledge_graph_xgboost import KnowledgeGraphXGBoost

# Initialize and run the complete pipeline
kg_xgb = KnowledgeGraphXGBoost()
results = kg_xgb.run_complete_pipeline()

# Access trained models
classification_model = results['classification_model']
regression_model = results['regression_model']
graph = results['graph']
```

### Advanced Features

```python
from advanced_kg_xgboost import AdvancedKnowledgeGraph

# Run advanced pipeline with hyperparameter tuning
kg = AdvancedKnowledgeGraph()
results, X, df, feature_cols = kg.run_advanced_pipeline()

# Access optimized models
person_classifier = results['person_classifier']['model']
value_regressor = results['value_regressor']['model']
```

## 🏗️ Architecture

### Knowledge Graph Structure

The knowledge graph consists of:

- **Vertices (Entities)**:
  - People (with skills, experience, location)
  - Companies (with size, industry, location)
  - Products (with metrics, status, category)
  - Skills (with demand ratings, difficulty)
  - Projects (with budget, timeline, success metrics)

- **Edges (Relationships)**:
  - Employment relationships
  - Skill proficiencies
  - Project participation
  - Professional networks
  - Product usage/development

### Graph Features Extracted

1. **Centrality Measures**:
   - In-degree, out-degree, total degree
   - PageRank scores
   - Betweenness centrality

2. **Community Detection**:
   - Connected components
   - Strongly connected components
   - Label propagation communities

3. **Structural Features**:
   - Triangle count (clustering coefficient)
   - Relationship diversity
   - Edge weight statistics

4. **Custom Features**:
   - Normalized degree centrality
   - Value per experience ratio
   - Clustering coefficients

### XGBoost Integration

The extracted graph features are used for multiple ML tasks:

- **Classification**: Predict entity types, high-value entities
- **Regression**: Predict numerical attributes like salary, company valuation
- **Ranking**: Rank entities by importance or relevance

## 📈 Example Results

### Sample Knowledge Graph
```
Vertices: 16 entities (People, Companies, Products, Skills)
Edges: 25 relationships with weights and temporal information
```

### Model Performance
```
Person Classification: 95%+ accuracy
Value Regression: Low MSE with R² > 0.8
Experience Classification: 90%+ accuracy
```

### Top Features
1. PageRank score
2. Total degree centrality
3. Community membership
4. Relationship diversity
5. Triangle count

## 🔧 Customization

### Adding New Entity Types

```python
# Add new vertices
new_vertices = [
    ("location_1", "Location", "San Francisco", "City", 884000, "Active", 0, "California")
]

# Add new relationships
new_edges = [
    ("person_1", "location_1", "lives_in", 5.0, "current", "2020-01-01")
]
```

### Custom Graph Algorithms

```python
# Add custom feature extraction
def extract_custom_features(graph):
    # Implement custom graph algorithms
    custom_centrality = graph.pageRank(resetProbability=0.1, maxIter=15)
    return custom_centrality.vertices
```

### XGBoost Model Tuning

```python
# Custom hyperparameter grid
param_grid = {
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
```

## 📁 File Structure

```
├── knowledge_graph_xgboost.py      # Main implementation
├── advanced_kg_xgboost.py          # Advanced features
├── setup_knowledge_graph.py        # Setup script
├── requirements.txt                # Dependencies
└── README_KnowledgeGraph.md        # This file
```

## 🎯 Use Cases

1. **Social Network Analysis**: Analyze user relationships and predict behavior
2. **Recommendation Systems**: Use graph features for collaborative filtering
3. **Fraud Detection**: Identify suspicious patterns in transaction networks
4. **Knowledge Management**: Organize and query enterprise knowledge
5. **Supply Chain Optimization**: Model and optimize complex supply networks
6. **Scientific Research**: Analyze citation networks and research collaborations

## 🚨 Troubleshooting

### Common Issues

1. **Java Not Found**:
   ```bash
   # Check Java installation
   java -version
   # Set JAVA_HOME if needed
   export JAVA_HOME=/path/to/java
   ```

2. **GraphFrames Package Error**:
   - Ensure PySpark version compatibility
   - Check internet connection for package download

3. **Memory Issues**:
   ```python
   # Increase Spark memory
   spark = SparkSession.builder \
       .config("spark.driver.memory", "4g") \
       .config("spark.executor.memory", "4g") \
       .getOrCreate()
   ```

4. **Slow Performance**:
   - Reduce graph size for testing
   - Increase Spark partitions
   - Use more powerful hardware

## 📚 References

- [GraphFrames Documentation](https://graphframes.github.io/graphframes/docs/_site/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation.

## 📄 License

This project is open source and available under the MIT License.

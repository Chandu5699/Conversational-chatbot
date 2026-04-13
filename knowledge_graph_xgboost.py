"""
Complete Knowledge Graph Implementation using GraphFrames and PySpark
Integrated with XGBoost for Machine Learning Tasks
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, avg, max as spark_max
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from graphframes import GraphFrame
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class KnowledgeGraphXGBoost:
    def __init__(self, app_name="KnowledgeGraph-XGBoost"):
        """Initialize Spark session and GraphFrames"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        self.graph = None
        self.vertex_features = None
        
    def create_sample_knowledge_graph(self):
        """Create a comprehensive sample knowledge graph"""
        
        # Define vertices (entities in knowledge graph)
        vertices_data = [
            # People
            ("person_1", "Person", "Alice", "Engineer", 28, "New York"),
            ("person_2", "Person", "Bob", "Manager", 35, "California"),
            ("person_3", "Person", "Charlie", "Analyst", 29, "Texas"),
            ("person_4", "Person", "Diana", "Designer", 31, "Florida"),
            ("person_5", "Person", "Eve", "Developer", 26, "Washington"),
            
            # Companies
            ("company_1", "Company", "TechCorp", "Technology", 1000, "San Francisco"),
            ("company_2", "Company", "DataInc", "Analytics", 500, "Seattle"),
            ("company_3", "Company", "AILabs", "AI/ML", 200, "Boston"),
            
            # Products
            ("product_1", "Product", "CloudPlatform", "Software", 99, "Enterprise"),
            ("product_2", "Product", "DataTool", "Analytics", 49, "Professional"),
            ("product_3", "Product", "MLFramework", "AI/ML", 199, "Enterprise"),
            
            # Skills
            ("skill_1", "Skill", "Python", "Programming", 5, "Technical"),
            ("skill_2", "Skill", "MachineLearning", "AI", 4, "Technical"),
            ("skill_3", "Skill", "Leadership", "Management", 3, "Soft"),
            ("skill_4", "Skill", "DataAnalysis", "Analytics", 4, "Technical"),
        ]
        
        vertices_schema = StructType([
            StructField("id", StringType(), True),
            StructField("type", StringType(), True),
            StructField("name", StringType(), True),
            StructField("category", StringType(), True),
            StructField("value", IntegerType(), True),
            StructField("location", StringType(), True)
        ])
        
        vertices = self.spark.createDataFrame(vertices_data, vertices_schema)
        
        # Define edges (relationships)
        edges_data = [
            # Employment relationships
            ("person_1", "company_1", "works_at", 2.5, "current"),
            ("person_2", "company_1", "works_at", 5.0, "current"),
            ("person_3", "company_2", "works_at", 1.5, "current"),
            ("person_4", "company_3", "works_at", 3.0, "current"),
            ("person_5", "company_2", "works_at", 2.0, "current"),
            
            # Skill relationships
            ("person_1", "skill_1", "has_skill", 4.5, "expert"),
            ("person_1", "skill_2", "has_skill", 3.5, "intermediate"),
            ("person_2", "skill_3", "has_skill", 4.0, "expert"),
            ("person_3", "skill_4", "has_skill", 4.2, "expert"),
            ("person_5", "skill_1", "has_skill", 4.8, "expert"),
            
            # Product usage
            ("person_1", "product_1", "uses", 3.0, "frequent"),
            ("person_2", "product_2", "uses", 4.0, "frequent"),
            ("person_3", "product_2", "uses", 3.5, "moderate"),
            ("company_1", "product_1", "develops", 5.0, "owner"),
            ("company_2", "product_2", "develops", 5.0, "owner"),
            
            # Social connections
            ("person_1", "person_2", "knows", 3.0, "colleague"),
            ("person_1", "person_5", "knows", 4.0, "friend"),
            ("person_2", "person_3", "knows", 2.5, "acquaintance"),
            ("person_3", "person_4", "knows", 3.5, "colleague"),
            
            # Company partnerships
            ("company_1", "company_2", "partners_with", 4.0, "strategic"),
            ("company_2", "company_3", "partners_with", 3.0, "technical"),
        ]
        
        edges_schema = StructType([
            StructField("src", StringType(), True),
            StructField("dst", StringType(), True),
            StructField("relationship", StringType(), True),
            StructField("weight", DoubleType(), True),
            StructField("status", StringType(), True)
        ])
        
        edges = self.spark.createDataFrame(edges_data, edges_schema)
        
        # Create GraphFrame
        self.graph = GraphFrame(vertices, edges)
        print("Knowledge Graph created successfully!")
        print(f"Vertices: {self.graph.vertices.count()}")
        print(f"Edges: {self.graph.edges.count()}")
        
        return self.graph
    
    def extract_graph_features(self):
        """Extract comprehensive features from the knowledge graph"""
        print("Extracting graph features...")
        
        # Basic vertex information
        base_vertices = self.graph.vertices
        
        # 1. Degree centrality measures
        in_degrees = self.graph.inDegrees.withColumnRenamed("inDegree", "in_degree")
        out_degrees = self.graph.outDegrees.withColumnRenamed("outDegree", "out_degree")
        total_degrees = self.graph.degrees.withColumnRenamed("degree", "total_degree")
        
        # 2. PageRank algorithm
        print("Computing PageRank...")
        pagerank_result = self.graph.pageRank(resetProbability=0.15, maxIter=10)
        pagerank_vertices = pagerank_result.vertices.select("id", col("pagerank").alias("pagerank_score"))
        
        # 3. Connected Components
        print("Computing Connected Components...")
        cc_result = self.graph.connectedComponents()
        cc_vertices = cc_result.select("id", col("component").alias("component_id"))
        
        # 4. Triangle Count (for clustering coefficient)
        print("Computing Triangle Count...")
        triangle_count = self.graph.triangleCount()
        triangle_vertices = triangle_count.vertices.select("id", col("count").alias("triangle_count"))
        
        # 5. Shortest Paths (sample)
        print("Computing Shortest Paths...")
        landmarks = ["person_1", "company_1", "skill_1"]
        shortest_paths = self.graph.shortestPaths(landmarks=landmarks)
        
        # Extract shortest path features
        sp_features = shortest_paths.select("id", col("distances").alias("shortest_paths"))
        
        # 6. Edge weight aggregations per vertex
        edge_stats = self.graph.edges.groupBy("src").agg(
            count("*").alias("outgoing_edges"),
            avg("weight").alias("avg_outgoing_weight"),
            spark_sum("weight").alias("total_outgoing_weight"),
            spark_max("weight").alias("max_outgoing_weight")
        ).withColumnRenamed("src", "id")
        
        # Join all features
        print("Combining all features...")
        self.vertex_features = base_vertices \
            .join(in_degrees, "id", "left_outer") \
            .join(out_degrees, "id", "left_outer") \
            .join(total_degrees, "id", "left_outer") \
            .join(pagerank_vertices, "id", "left_outer") \
            .join(cc_vertices, "id", "left_outer") \
            .join(triangle_vertices, "id", "left_outer") \
            .join(edge_stats, "id", "left_outer") \
            .fillna(0)
        
        print("Graph features extracted successfully!")
        return self.vertex_features
    
    def prepare_xgboost_data(self, target_column="type", prediction_task="classification"):
        """Prepare data for XGBoost training"""
        print("Preparing data for XGBoost...")
        
        # Convert to Pandas
        df_pandas = self.vertex_features.toPandas()
        
        # Create target variable based on task
        if prediction_task == "classification":
            # Example: Predict if entity is a Person
            df_pandas['target'] = (df_pandas[target_column] == 'Person').astype(int)
        elif prediction_task == "regression":
            # Example: Predict the 'value' column
            df_pandas['target'] = df_pandas['value']
        
        # Select numerical features for XGBoost
        feature_columns = [
            'value', 'in_degree', 'out_degree', 'total_degree', 
            'pagerank_score', 'component_id', 'triangle_count',
            'outgoing_edges', 'avg_outgoing_weight', 
            'total_outgoing_weight', 'max_outgoing_weight'
        ]
        
        # Handle missing values
        df_pandas[feature_columns] = df_pandas[feature_columns].fillna(0)
        
        X = df_pandas[feature_columns]
        y = df_pandas['target']
        
        return X, y, df_pandas
    
    def train_xgboost_model(self, X, y, task_type="classification"):
        """Train XGBoost model"""
        print(f"Training XGBoost model for {task_type}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if task_type == "classification" else None
        )
        
        if task_type == "classification":
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )
        else:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        else:
            mse = np.mean((y_test - y_pred) ** 2)
            print(f"Mean Squared Error: {mse:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return model, X_test, y_test, y_pred
    
    def run_complete_pipeline(self):
        """Run the complete knowledge graph + XGBoost pipeline"""
        print("=" * 60)
        print("KNOWLEDGE GRAPH + XGBOOST PIPELINE")
        print("=" * 60)
        
        # Step 1: Create knowledge graph
        self.create_sample_knowledge_graph()
        
        # Step 2: Extract features
        self.extract_graph_features()
        
        # Step 3: Prepare data and train models
        X, y, df_full = self.prepare_xgboost_data(prediction_task="classification")
        
        # Step 4: Train classification model
        print("\n" + "=" * 40)
        print("CLASSIFICATION TASK")
        print("=" * 40)
        model_clf, X_test_clf, y_test_clf, y_pred_clf = self.train_xgboost_model(X, y, "classification")
        
        # Step 5: Train regression model (predicting 'value')
        print("\n" + "=" * 40)
        print("REGRESSION TASK")
        print("=" * 40)
        X_reg, y_reg, _ = self.prepare_xgboost_data(target_column="value", prediction_task="regression")
        model_reg, X_test_reg, y_test_reg, y_pred_reg = self.train_xgboost_model(X_reg, y_reg, "regression")
        
        return {
            'classification_model': model_clf,
            'regression_model': model_reg,
            'graph': self.graph,
            'features': self.vertex_features
        }
    
    def close(self):
        """Close Spark session"""
        self.spark.stop()

# Example usage
if __name__ == "__main__":
    # Initialize and run the pipeline
    kg_xgb = KnowledgeGraphXGBoost()
    
    try:
        results = kg_xgb.run_complete_pipeline()
        print("\nPipeline completed successfully!")
        
        # Display some sample predictions
        print("\nSample Knowledge Graph Entities and Predictions:")
        sample_data = kg_xgb.vertex_features.select("id", "name", "type").limit(5).toPandas()
        print(sample_data)
        
    finally:
        kg_xgb.close()

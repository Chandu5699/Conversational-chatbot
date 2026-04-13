"""
Advanced Knowledge Graph with XGBoost
Includes more sophisticated graph algorithms and ML techniques
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, size, explode, collect_list
from pyspark.sql.types import *
from graphframes import GraphFrame
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedKnowledgeGraph:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("Advanced-KG-XGBoost") \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
    def create_complex_knowledge_graph(self):
        """Create a more complex knowledge graph with multiple entity types"""
        
        # Vertices with rich attributes
        vertices_data = [
            # People with detailed attributes
            ("p1", "Person", "Alice Johnson", "Senior Engineer", 85000, "Tech", 5, "San Francisco"),
            ("p2", "Person", "Bob Smith", "Product Manager", 95000, "Business", 7, "New York"),
            ("p3", "Person", "Carol Davis", "Data Scientist", 90000, "Tech", 4, "Seattle"),
            ("p4", "Person", "David Wilson", "UX Designer", 75000, "Design", 6, "Austin"),
            ("p5", "Person", "Eve Brown", "DevOps Engineer", 88000, "Tech", 3, "Denver"),
            
            # Companies with market data
            ("c1", "Company", "TechGiant", "Technology", 50000, "Public", 15, "California"),
            ("c2", "Company", "StartupX", "AI/ML", 150, "Private", 2, "Massachusetts"),
            ("c3", "Company", "DataCorp", "Analytics", 5000, "Public", 10, "Washington"),
            
            # Products with performance metrics
            ("pr1", "Product", "CloudPlatform", "SaaS", 1000000, "Active", 8, "Enterprise"),
            ("pr2", "Product", "MLToolkit", "Software", 50000, "Active", 3, "Developer"),
            ("pr3", "Product", "AnalyticsDash", "Dashboard", 200000, "Active", 5, "Business"),
            
            # Skills with demand ratings
            ("s1", "Skill", "Python", "Programming", 95, "High", 10, "Technical"),
            ("s2", "Skill", "Leadership", "Management", 85, "High", 8, "Soft"),
            ("s3", "Skill", "MachineLearning", "AI", 90, "Very High", 6, "Technical"),
            ("s4", "Skill", "CloudComputing", "Infrastructure", 88, "High", 7, "Technical"),
            
            # Projects with success metrics
            ("proj1", "Project", "AI Initiative", "Research", 500000, "Completed", 12, "Strategic"),
            ("proj2", "Project", "Platform Migration", "Infrastructure", 200000, "Active", 6, "Operational"),
        ]
        
        vertices_schema = StructType([
            StructField("id", StringType(), True),
            StructField("type", StringType(), True),
            StructField("name", StringType(), True),
            StructField("category", StringType(), True),
            StructField("value", IntegerType(), True),
            StructField("status", StringType(), True),
            StructField("experience", IntegerType(), True),
            StructField("location", StringType(), True)
        ])
        
        vertices = self.spark.createDataFrame(vertices_data, vertices_schema)
        
        # Complex edges with temporal and weighted relationships
        edges_data = [
            # Employment with tenure and performance
            ("p1", "c1", "employed_by", 4.5, "current", "2020-01-15"),
            ("p2", "c1", "employed_by", 4.8, "current", "2019-03-01"),
            ("p3", "c3", "employed_by", 4.2, "current", "2021-06-01"),
            ("p4", "c2", "employed_by", 4.0, "current", "2022-01-01"),
            ("p5", "c3", "employed_by", 4.3, "current", "2021-09-15"),
            
            # Skill proficiency with certification levels
            ("p1", "s1", "has_skill", 4.8, "expert", "2018-01-01"),
            ("p1", "s3", "has_skill", 4.2, "advanced", "2020-01-01"),
            ("p1", "s4", "has_skill", 3.8, "intermediate", "2021-01-01"),
            ("p2", "s2", "has_skill", 4.5, "expert", "2017-01-01"),
            ("p3", "s1", "has_skill", 4.9, "expert", "2019-01-01"),
            ("p3", "s3", "has_skill", 4.7, "expert", "2020-01-01"),
            ("p5", "s4", "has_skill", 4.6, "expert", "2020-01-01"),
            
            # Project participation with contribution scores
            ("p1", "proj1", "worked_on", 4.5, "lead", "2022-01-01"),
            ("p3", "proj1", "worked_on", 4.3, "contributor", "2022-01-01"),
            ("p2", "proj2", "worked_on", 4.7, "manager", "2023-01-01"),
            ("p5", "proj2", "worked_on", 4.1, "contributor", "2023-01-01"),
            
            # Product development and usage
            ("c1", "pr1", "develops", 5.0, "owner", "2018-01-01"),
            ("c2", "pr2", "develops", 5.0, "owner", "2021-01-01"),
            ("c3", "pr3", "develops", 5.0, "owner", "2019-01-01"),
            ("p1", "pr1", "uses", 4.2, "power_user", "2020-01-01"),
            ("p3", "pr2", "uses", 4.5, "power_user", "2021-01-01"),
            
            # Professional network with interaction frequency
            ("p1", "p2", "collaborates_with", 3.5, "frequent", "2020-01-01"),
            ("p1", "p3", "mentors", 4.0, "regular", "2021-01-01"),
            ("p2", "p4", "collaborates_with", 3.2, "occasional", "2022-01-01"),
            ("p3", "p5", "collaborates_with", 3.8, "frequent", "2021-01-01"),
            
            # Company partnerships and competitions
            ("c1", "c3", "partners_with", 4.0, "strategic", "2021-01-01"),
            ("c2", "c1", "acquired_by", 5.0, "completed", "2023-01-01"),
        ]
        
        edges_schema = StructType([
            StructField("src", StringType(), True),
            StructField("dst", StringType(), True),
            StructField("relationship", StringType(), True),
            StructField("weight", DoubleType(), True),
            StructField("status", StringType(), True),
            StructField("date", StringType(), True)
        ])
        
        edges = self.spark.createDataFrame(edges_data, edges_schema)
        
        self.graph = GraphFrame(vertices, edges)
        return self.graph
    
    def extract_advanced_features(self):
        """Extract sophisticated graph features"""
        print("Extracting advanced graph features...")
        
        # Basic centrality measures
        in_degrees = self.graph.inDegrees
        out_degrees = self.graph.outDegrees
        total_degrees = self.graph.degrees
        
        # PageRank with different parameters
        pagerank = self.graph.pageRank(resetProbability=0.15, maxIter=20)
        pagerank_vertices = pagerank.vertices.select("id", col("pagerank").alias("pagerank_score"))
        
        # Connected components
        cc = self.graph.connectedComponents()
        cc_vertices = cc.select("id", col("component").alias("component_id"))
        
        # Triangle count for clustering coefficient
        triangles = self.graph.triangleCount()
        triangle_vertices = triangles.vertices.select("id", col("count").alias("triangle_count"))
        
        # Strongly connected components
        scc = self.graph.stronglyConnectedComponents(maxIter=10)
        scc_vertices = scc.select("id", col("component").alias("scc_id"))
        
        # Label propagation for community detection
        lpa = self.graph.labelPropagation(maxIter=5)
        lpa_vertices = lpa.select("id", col("label").alias("community_label"))
        
        # Edge-based features
        edge_features = self.graph.edges.groupBy("src").agg(
            count("*").alias("out_edge_count"),
            avg("weight").alias("avg_out_weight"),
            sum("weight").alias("total_out_weight"),
            max("weight").alias("max_out_weight")
        ).withColumnRenamed("src", "id")
        
        # Relationship diversity (number of unique relationship types)
        rel_diversity = self.graph.edges.groupBy("src").agg(
            countDistinct("relationship").alias("relationship_diversity")
        ).withColumnRenamed("src", "id")
        
        # Combine all features
        base_vertices = self.graph.vertices
        
        feature_df = base_vertices \
            .join(in_degrees, "id", "left_outer") \
            .join(out_degrees, "id", "left_outer") \
            .join(total_degrees, "id", "left_outer") \
            .join(pagerank_vertices, "id", "left_outer") \
            .join(cc_vertices, "id", "left_outer") \
            .join(triangle_vertices, "id", "left_outer") \
            .join(scc_vertices, "id", "left_outer") \
            .join(lpa_vertices, "id", "left_outer") \
            .join(edge_features, "id", "left_outer") \
            .join(rel_diversity, "id", "left_outer") \
            .fillna(0)
        
        return feature_df
    
    def prepare_ml_data(self, feature_df, target_type="Person"):
        """Prepare data for machine learning with advanced preprocessing"""
        
        # Convert to pandas
        df = feature_df.toPandas()
        
        # Create multiple target variables for different tasks
        df['is_person'] = (df['type'] == target_type).astype(int)
        df['high_value'] = (df['value'] > df['value'].median()).astype(int)
        df['high_experience'] = (df['experience'] > df['experience'].median()).astype(int)
        
        # Encode categorical variables
        le_type = LabelEncoder()
        le_category = LabelEncoder()
        le_status = LabelEncoder()
        le_location = LabelEncoder()
        
        df['type_encoded'] = le_type.fit_transform(df['type'])
        df['category_encoded'] = le_category.fit_transform(df['category'])
        df['status_encoded'] = le_status.fit_transform(df['status'])
        df['location_encoded'] = le_location.fit_transform(df['location'])
        
        # Feature engineering
        df['degree_centrality'] = df['degree'] / (len(df) - 1)  # Normalized degree
        df['clustering_coefficient'] = df['triangle_count'] / (df['degree'] * (df['degree'] - 1) / 2 + 1)
        df['value_per_experience'] = df['value'] / (df['experience'] + 1)
        
        # Select features for ML
        feature_cols = [
            'value', 'experience', 'inDegree', 'outDegree', 'degree',
            'pagerank_score', 'component_id', 'triangle_count', 'scc_id',
            'community_label', 'out_edge_count', 'avg_out_weight',
            'total_out_weight', 'max_out_weight', 'relationship_diversity',
            'type_encoded', 'category_encoded', 'status_encoded', 'location_encoded',
            'degree_centrality', 'clustering_coefficient', 'value_per_experience'
        ]
        
        X = df[feature_cols].fillna(0)
        
        return X, df, feature_cols
    
    def train_advanced_models(self, X, df):
        """Train multiple XGBoost models with hyperparameter tuning"""
        
        results = {}
        
        # Task 1: Person classification
        print("Training Person Classification Model...")
        y_person = df['is_person']
        
        # Hyperparameter tuning for classification
        param_grid_clf = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0]
        }
        
        xgb_clf = xgb.XGBClassifier(random_state=42)
        grid_clf = GridSearchCV(xgb_clf, param_grid_clf, cv=3, scoring='accuracy', n_jobs=-1)
        grid_clf.fit(X, y_person)
        
        results['person_classifier'] = {
            'model': grid_clf.best_estimator_,
            'best_params': grid_clf.best_params_,
            'best_score': grid_clf.best_score_
        }
        
        # Task 2: Value regression
        print("Training Value Regression Model...")
        y_value = df['value']
        
        param_grid_reg = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200]
        }
        
        xgb_reg = xgb.XGBRegressor(random_state=42)
        grid_reg = GridSearchCV(xgb_reg, param_grid_reg, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_reg.fit(X, y_value)
        
        results['value_regressor'] = {
            'model': grid_reg.best_estimator_,
            'best_params': grid_reg.best_params_,
            'best_score': -grid_reg.best_score_  # Convert back to positive MSE
        }
        
        # Task 3: High experience classification
        print("Training High Experience Classification Model...")
        y_exp = df['high_experience']
        
        xgb_exp = xgb.XGBClassifier(**grid_clf.best_params_, random_state=42)
        exp_scores = cross_val_score(xgb_exp, X, y_exp, cv=5, scoring='accuracy')
        xgb_exp.fit(X, y_exp)
        
        results['experience_classifier'] = {
            'model': xgb_exp,
            'cv_scores': exp_scores,
            'mean_cv_score': exp_scores.mean()
        }
        
        return results
    
    def visualize_results(self, results, X, feature_cols):
        """Create visualizations of model results"""
        
        plt.figure(figsize=(15, 10))
        
        # Feature importance for person classifier
        plt.subplot(2, 2, 1)
        person_model = results['person_classifier']['model']
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': person_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Top 10 Features - Person Classification')
        plt.xlabel('Feature Importance')
        
        # Feature importance for value regressor
        plt.subplot(2, 2, 2)
        value_model = results['value_regressor']['model']
        feature_importance_reg = pd.DataFrame({
            'feature': feature_cols,
            'importance': value_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        plt.barh(feature_importance_reg['feature'], feature_importance_reg['importance'])
        plt.title('Top 10 Features - Value Regression')
        plt.xlabel('Feature Importance')
        
        # Model performance comparison
        plt.subplot(2, 2, 3)
        models = ['Person Classifier', 'Experience Classifier']
        scores = [results['person_classifier']['best_score'], 
                 results['experience_classifier']['mean_cv_score']]
        
        plt.bar(models, scores)
        plt.title('Classification Model Accuracy')
        plt.ylabel('Accuracy Score')
        plt.ylim(0, 1)
        
        # Cross-validation scores for experience classifier
        plt.subplot(2, 2, 4)
        cv_scores = results['experience_classifier']['cv_scores']
        plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-')
        plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
        plt.title('Experience Classifier - CV Scores')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('kg_xgboost_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance, feature_importance_reg
    
    def run_advanced_pipeline(self):
        """Run the complete advanced pipeline"""
        print("=" * 60)
        print("ADVANCED KNOWLEDGE GRAPH + XGBOOST PIPELINE")
        print("=" * 60)
        
        # Create complex knowledge graph
        graph = self.create_complex_knowledge_graph()
        print(f"Created graph with {graph.vertices.count()} vertices and {graph.edges.count()} edges")
        
        # Extract advanced features
        feature_df = self.extract_advanced_features()
        
        # Prepare ML data
        X, df, feature_cols = self.prepare_ml_data(feature_df)
        print(f"Prepared {len(feature_cols)} features for {len(df)} entities")
        
        # Train models
        results = self.train_advanced_models(X, df)
        
        # Print results
        print("\n" + "=" * 40)
        print("MODEL RESULTS")
        print("=" * 40)
        
        for task, result in results.items():
            print(f"\n{task.upper()}:")
            if 'best_score' in result:
                print(f"  Best Score: {result['best_score']:.4f}")
                print(f"  Best Params: {result['best_params']}")
            if 'mean_cv_score' in result:
                print(f"  Mean CV Score: {result['mean_cv_score']:.4f}")
        
        # Visualize results
        try:
            self.visualize_results(results, X, feature_cols)
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        return results, X, df, feature_cols
    
    def close(self):
        self.spark.stop()

if __name__ == "__main__":
    kg = AdvancedKnowledgeGraph()
    try:
        results, X, df, feature_cols = kg.run_advanced_pipeline()
        print("\nAdvanced pipeline completed successfully!")
    finally:
        kg.close()

"""
Natural Language Processing Engine
Advanced NLP capabilities for labor analytics text processing
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Advanced NLP
import spacy
from textblob import TextBlob
import gensim
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans

# Deep Learning for NLP
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# Custom imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_connectors.database_connector import DatabaseConnector
import structlog

logger = structlog.get_logger(__name__)


class NLPProcessor:
    """Advanced NLP processor for labor analytics"""
    
    def __init__(self, config_path: str = "../config/ai_config.json"):
        """Initialize NLP processor"""
        self.config = self._load_config(config_path)
        self.db_connector = DatabaseConnector()
        self.nlp_models = {}
        self.vectorizers = {}
        self._setup_nltk()
        self._setup_spacy()
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load NLP configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('natural_language_processing', {})
        except FileNotFoundError:
            return {
                "sentiment_analysis": {"enabled": True, "model": "vader"},
                "text_classification": {"enabled": True, "categories": ["urgent", "normal", "low_priority"]},
                "entity_extraction": {"enabled": True, "entities": ["project_names", "client_names", "dates"]}
            }
    
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
    
    def _setup_spacy(self):
        """Setup spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _setup_logging(self):
        """Setup structured logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP analysis"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_tokens(self, text: str) -> List[str]:
        """Extract tokens from text"""
        processed_text = self.preprocess_text(text)
        tokens = word_tokenize(processed_text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        return tokens
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text"""
        tokens = self.extract_tokens(text)
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            if not text or pd.isna(text):
                return {"sentiment": "neutral", "scores": {"pos": 0, "neg": 0, "neu": 0, "compound": 0}}
            
            # VADER Sentiment Analysis
            sia = SentimentIntensityAnalyzer()
            vader_scores = sia.polarity_scores(text)
            
            # TextBlob Sentiment Analysis
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment
            compound_score = vader_scores['compound']
            if compound_score >= 0.05:
                sentiment = "positive"
            elif compound_score <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "vader_scores": vader_scores,
                "textblob_polarity": textblob_polarity,
                "textblob_subjectivity": textblob_subjectivity,
                "confidence": abs(compound_score)
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"sentiment": "neutral", "error": str(e)}
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        try:
            if not text or pd.isna(text):
                return {"entities": {}}
            
            entities = {
                "project_names": [],
                "client_names": [],
                "dates": [],
                "amounts": [],
                "locations": [],
                "organizations": []
            }
            
            # NLTK Named Entity Recognition
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    entity_type = chunk.label()
                    
                    if entity_type == 'PERSON':
                        entities["client_names"].append(entity_text)
                    elif entity_type == 'ORGANIZATION':
                        entities["organizations"].append(entity_text)
                    elif entity_type == 'GPE':
                        entities["locations"].append(entity_text)
                    elif entity_type == 'DATE':
                        entities["dates"].append(entity_text)
            
            # spaCy Entity Recognition (if available)
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        entities["client_names"].append(ent.text)
                    elif ent.label_ == 'ORG':
                        entities["organizations"].append(ent.text)
                    elif ent.label_ == 'GPE':
                        entities["locations"].append(ent.text)
                    elif ent.label_ == 'DATE':
                        entities["dates"].append(ent.text)
                    elif ent.label_ == 'MONEY':
                        entities["amounts"].append(ent.text)
            
            # Extract project names using regex patterns
            project_patterns = [
                r'\b(?:project|prj|proj)\s*[#\-]?\s*([a-zA-Z0-9]+)',
                r'\b([A-Z]{2,}\s*(?:Project|Initiative|Program))',
                r'\b([a-zA-Z]+\s*[0-9]{4})\b'
            ]
            
            for pattern in project_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["project_names"].extend(matches)
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return {"entities": entities}
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return {"entities": {}, "error": str(e)}
    
    def classify_text(self, text: str, categories: List[str] = None) -> Dict[str, Any]:
        """Classify text into predefined categories"""
        try:
            if not text or pd.isna(text):
                return {"category": "unknown", "confidence": 0.0}
            
            if categories is None:
                categories = self.config.get("text_classification", {}).get("categories", ["urgent", "normal", "low_priority"])
            
            # Simple keyword-based classification
            text_lower = text.lower()
            
            # Define keywords for each category
            category_keywords = {
                "urgent": ["urgent", "asap", "immediate", "critical", "emergency", "deadline", "rush"],
                "normal": ["normal", "regular", "standard", "routine", "usual"],
                "low_priority": ["low", "minor", "non-critical", "optional", "when possible"]
            }
            
            scores = {}
            for category in categories:
                if category in category_keywords:
                    score = sum(1 for keyword in category_keywords[category] if keyword in text_lower)
                    scores[category] = score
            
            # Determine best category
            if scores:
                best_category = max(scores, key=scores.get)
                confidence = scores[best_category] / max(scores.values()) if max(scores.values()) > 0 else 0
            else:
                best_category = "normal"
                confidence = 0.5
            
            return {
                "category": best_category,
                "confidence": confidence,
                "scores": scores
            }
            
        except Exception as e:
            logger.error(f"Error in text classification: {e}")
            return {"category": "unknown", "confidence": 0.0, "error": str(e)}
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords from text using TF-IDF"""
        try:
            if not text or pd.isna(text):
                return []
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=top_k,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores
            scores = tfidf_matrix.toarray()[0]
            
            # Create keyword-score pairs
            keywords = list(zip(feature_names, scores))
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords[:top_k]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def topic_modeling(self, texts: List[str], n_topics: int = 5) -> Dict[str, Any]:
        """Perform topic modeling using LDA"""
        try:
            if not texts:
                return {"topics": [], "error": "No texts provided"}
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts if text and not pd.isna(text)]
            
            if not processed_texts:
                return {"topics": [], "error": "No valid texts after preprocessing"}
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
            
            lda.fit(tfidf_matrix)
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": topic[top_words_idx].tolist()
                })
            
            return {
                "topics": topics,
                "n_topics": n_topics,
                "feature_names": feature_names.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return {"topics": [], "error": str(e)}
    
    def create_text_embeddings(self, texts: List[str], method: str = "tfidf") -> np.ndarray:
        """Create text embeddings"""
        try:
            if not texts:
                return np.array([])
            
            processed_texts = [self.preprocess_text(text) for text in texts if text and not pd.isna(text)]
            
            if method == "tfidf":
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                embeddings = vectorizer.fit_transform(processed_texts).toarray()
            
            elif method == "word2vec":
                # Tokenize texts
                tokenized_texts = [self.extract_tokens(text) for text in processed_texts]
                
                # Train Word2Vec model
                model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
                
                # Create document embeddings (average of word vectors)
                embeddings = []
                for tokens in tokenized_texts:
                    if tokens:
                        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
                        if word_vectors:
                            doc_vector = np.mean(word_vectors, axis=0)
                        else:
                            doc_vector = np.zeros(100)
                    else:
                        doc_vector = np.zeros(100)
                    embeddings.append(doc_vector)
                
                embeddings = np.array(embeddings)
            
            else:
                raise ValueError(f"Unknown embedding method: {method}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating text embeddings: {e}")
            return np.array([])
    
    def analyze_task_descriptions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze task descriptions for insights"""
        try:
            if 'task_description' not in df.columns:
                return {"error": "No task_description column found"}
            
            # Filter out empty descriptions
            valid_descriptions = df[df['task_description'].notna() & (df['task_description'] != '')]
            
            if valid_descriptions.empty:
                return {"error": "No valid task descriptions found"}
            
            results = {
                "total_descriptions": len(valid_descriptions),
                "sentiment_analysis": [],
                "entity_extraction": [],
                "text_classification": [],
                "keywords": [],
                "topic_modeling": None
            }
            
            # Analyze each description
            for idx, row in valid_descriptions.iterrows():
                description = row['task_description']
                
                # Sentiment analysis
                sentiment = self.analyze_sentiment(description)
                results["sentiment_analysis"].append({
                    "index": idx,
                    "sentiment": sentiment
                })
                
                # Entity extraction
                entities = self.extract_entities(description)
                results["entity_extraction"].append({
                    "index": idx,
                    "entities": entities
                })
                
                # Text classification
                classification = self.classify_text(description)
                results["text_classification"].append({
                    "index": idx,
                    "classification": classification
                })
                
                # Keywords
                keywords = self.extract_keywords(description, top_k=5)
                results["keywords"].append({
                    "index": idx,
                    "keywords": keywords
                })
            
            # Topic modeling on all descriptions
            all_descriptions = valid_descriptions['task_description'].tolist()
            topics = self.topic_modeling(all_descriptions, n_topics=3)
            results["topic_modeling"] = topics
            
            # Summary statistics
            sentiment_distribution = {}
            category_distribution = {}
            
            for sentiment_result in results["sentiment_analysis"]:
                sentiment = sentiment_result["sentiment"]["sentiment"]
                sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
            
            for classification_result in results["text_classification"]:
                category = classification_result["classification"]["category"]
                category_distribution[category] = category_distribution.get(category, 0) + 1
            
            results["summary"] = {
                "sentiment_distribution": sentiment_distribution,
                "category_distribution": category_distribution
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing task descriptions: {e}")
            return {"error": str(e)}
    
    def generate_nlp_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive NLP analysis report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("NATURAL LANGUAGE PROCESSING ANALYSIS REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            if "error" in analysis_results:
                report.append(f"❌ Error: {analysis_results['error']}")
                return "\n".join(report)
            
            # Summary
            summary = analysis_results.get("summary", {})
            report.append("📊 SUMMARY STATISTICS")
            report.append("-" * 30)
            report.append(f"Total Descriptions Analyzed: {analysis_results.get('total_descriptions', 0)}")
            
            # Sentiment distribution
            sentiment_dist = summary.get("sentiment_distribution", {})
            if sentiment_dist:
                report.append("\n😊 SENTIMENT DISTRIBUTION:")
                for sentiment, count in sentiment_dist.items():
                    percentage = (count / analysis_results.get('total_descriptions', 1)) * 100
                    report.append(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
            
            # Category distribution
            category_dist = summary.get("category_distribution", {})
            if category_dist:
                report.append("\n🏷️  CATEGORY DISTRIBUTION:")
                for category, count in category_dist.items():
                    percentage = (count / analysis_results.get('total_descriptions', 1)) * 100
                    report.append(f"  {category.capitalize()}: {count} ({percentage:.1f}%)")
            
            # Topic modeling
            topics = analysis_results.get("topic_modeling", {})
            if topics and "topics" in topics:
                report.append("\n📝 TOPIC MODELING RESULTS:")
                for topic in topics["topics"]:
                    report.append(f"  Topic {topic['topic_id'] + 1}: {', '.join(topic['words'][:5])}")
            
            # Common entities
            all_entities = {}
            for entity_result in analysis_results.get("entity_extraction", []):
                entities = entity_result.get("entities", {}).get("entities", {})
                for entity_type, entity_list in entities.items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = {}
                    for entity in entity_list:
                        all_entities[entity_type][entity] = all_entities[entity_type].get(entity, 0) + 1
            
            if all_entities:
                report.append("\n🏢 COMMON ENTITIES:")
                for entity_type, entities in all_entities.items():
                    if entities:
                        top_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:3]
                        report.append(f"  {entity_type.replace('_', ' ').title()}: {', '.join([f'{entity} ({count})' for entity, count in top_entities])}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating NLP report: {e}")
            return f"Error generating report: {str(e)}"


if __name__ == "__main__":
    # Test NLP processor
    nlp_processor = NLPProcessor()
    
    # Test with sample data
    sample_texts = [
        "Urgent: Need to complete the website redesign project by Friday",
        "Regular maintenance and updates for the mobile app",
        "Low priority: Update documentation when possible",
        "Critical bug fix required for production system",
        "Standard weekly team meeting and progress review"
    ]
    
    print("NLP Processing Test Results:")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nText {i}: {text}")
        
        # Sentiment analysis
        sentiment = nlp_processor.analyze_sentiment(text)
        print(f"  Sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.2f})")
        
        # Entity extraction
        entities = nlp_processor.extract_entities(text)
        print(f"  Entities: {entities['entities']}")
        
        # Text classification
        classification = nlp_processor.classify_text(text)
        print(f"  Category: {classification['category']} (confidence: {classification['confidence']:.2f})")
        
        # Keywords
        keywords = nlp_processor.extract_keywords(text, top_k=3)
        print(f"  Keywords: {[kw[0] for kw in keywords]}")
    
    # Topic modeling
    print(f"\nTopic Modeling:")
    topics = nlp_processor.topic_modeling(sample_texts, n_topics=2)
    if "topics" in topics:
        for topic in topics["topics"]:
            print(f"  Topic {topic['topic_id'] + 1}: {', '.join(topic['words'][:5])}")

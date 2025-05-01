# NLP Data Cleaning and Analysis Project

## Problem Description
This project demonstrates various data cleaning and transformation techniques for NLP tasks using a product reviews dataset. The main objectives include:
- Handling missing values using multiple imputation strategies
- Detecting and treating outliers
- Cleaning and preprocessing text data
- Applying different text featurization techniques

## Project Structure
```
├── product_reviews.csv                 # Raw dataset
├── product_reviews_basic_clean.csv     # Dataset after basic cleaning
├── product_reviews_advanced_clean.csv  # Dataset after advanced cleaning
├── nlp_data_cleaning_project.ipynb     # Main Jupyter notebook with all code
├── README.md                           # Project documentation
└── visualizations/                     # Folder containing generated visualizations
    ├── missing_values.png
    ├── rating_sentiment_distribution.png
    ├── numerical_features_distribution.png
    ├── helpfulness_scores_comparison.png
    └── word_frequency_comparison.png
```

## Data Description
The dataset contains product reviews with the following features:
- `product_id`: Unique identifier for products
- `rating`: User rating (1-5 stars)
- `review`: Text of the product review
- `sentiment`: Labeled sentiment (positive, neutral, negative)
- `helpfulness_score`: Score indicating review helpfulness
- `review_length`: Number of words in the review
- `sales`: Associated product sales

## Data Cleansing Approaches

### Approach 1: Basic Data Cleaning
This approach focuses on simple statistical methods for handling missing values and outliers:

#### Missing Value Treatment:
- **Mean Imputation**: Applied to `helpfulness_score` 
- **Median Imputation**: Applied to `rating` and `sales` (better for skewed distributions)
- **Mode Imputation**: Applied to categorical variables like `sentiment`
- **Placeholder Text**: Used for missing reviews

#### Outlier Detection and Treatment:
- **IQR Method**: Used to identify outliers in `helpfulness_score`
- **Capping**: Values above upper bound or below lower bound are capped

#### Basic Text Cleaning:
```python
def clean_text_basic(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

### Approach 2: Advanced Data Cleaning
This approach uses more sophisticated techniques that leverage relationships between features:

#### Missing Value Treatment:
- **KNN Imputation**: Used for numerical features (`rating`, `helpfulness_score`, `review_length`)
- **Model-based Imputation**: Used Random Forest to predict missing `sales` values based on other features
- **Context-aware Imputation**: Generated missing reviews based on imputed rating and sentiment

#### Outlier Treatment:
- **Winsorization**: Applied percentile-based capping to handle outliers while preserving distribution shape
- **Domain-specific Constraints**: Applied reasonable bounds for `helpfulness_score` (0-100)

#### Advanced Text Cleaning:
```python
def clean_text_advanced(text):
    # Basic cleaning steps (as above)
    ...
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)
```

## Text Featurization Techniques

Three different text featurization methods were implemented to transform cleaned text data into numerical features:

### 1. Bag of Words (CountVectorizer)
```python
count_vectorizer = CountVectorizer(max_features=1000)
count_vectors = count_vectorizer.fit_transform(df_advanced_clean['clean_review_basic'])
```

### 2. TF-IDF Vectorization
```python
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_vectors = tfidf_vectorizer.fit_transform(df_advanced_clean['clean_review_basic'])
```

### 3. N-gram Features
```python
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)
ngram_vectors = ngram_vectorizer.fit_transform(df_advanced_clean['clean_review_basic'])
```

## Results and Evaluation

### Impact of Data Cleaning
- Successfully handled all missing values in the dataset
- Effectively mitigated extreme values while preserving the overall distribution
- Improved data quality with more consistent statistical properties

### Text Processing Effectiveness
- Removed noise (HTML tags, special characters) from the text
- Reduced vocabulary size by removing stopwords and lemmatizing terms
- Prepared clean text for effective feature extraction

### Comparing Approaches
The advanced cleaning approach provided better results by:
- Utilizing contextual information between features
- Providing more accurate imputation through ML-based methods
- Better preserving the statistical properties of the original data

## Key Insights
1. Model-based imputation outperforms simple statistical imputation for complex relationships
2. Text preprocessing significantly improves the quality of NLP features
3. Different text featurization techniques capture different aspects of the text:
   - Bag of Words: Simple presence of terms
   - TF-IDF: Term importance relative to the corpus
   - N-grams: Contextual patterns and phrases

## Installation and Usage
1. Clone this repository
```bash
git clone https://github.com/yourusername/nlp-data-cleaning-project.git
cd nlp-data-cleaning-project
```

2. Install required packages
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

3. Run the Jupyter notebook
```bash
jupyter notebook nlp_data_cleaning_project.ipynb
```

## Future Work
- Experiment with more sophisticated imputation methods like MICE (Multiple Imputation by Chained Equations)
- Apply deep learning-based text representations (word embeddings, BERT)
- Develop an end-to-end pipeline for sentiment analysis using the cleaned data

## References
- Brownlee, J. (2020). *Data Preparation for Machine Learning*
- Manning, C., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*
- VanderPlas, J. (2016). *Python Data Science Handbook*

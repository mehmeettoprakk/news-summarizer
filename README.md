News Summarizer and Classifier
Overview
This Python-based application fetches real-time news from multiple RSS feeds, summarizes articles using Latent Semantic Analysis (LSA), and classifies them into predefined categories using machine learning models. The interactive CustomTkinter GUI allows users to view and analyze news efficiently.

Features<br>
📰 Real-time News Retrieval: Fetches news from various categories (Economy, Sports, Technology, etc.).<br>
✂ LSA-Based Summarization: Extracts key points from news articles.<br>
🤖 Machine Learning Classification: Trains and compares Naïve Bayes, Logistic Regression, Random Forest, and SVM models.<br>
🚀 Parallel Processing: Enhances performance using ThreadPoolExecutor.<br>
🔍 Custom News Prediction: Users can input custom summaries for classification.<br>
🎨 Interactive GUI: Provides an intuitive interface with CustomTkinter for easy navigation. <br>
<br>
How It Works<br>
1.News articles are fetched from RSS feeds. <br>
2.The text is cleaned and summarized using LSA. <br>
3.A TF-IDF vectorizer transforms the text for classification. <br>
4.The best machine learning model is selected based on accuracy. <br> 
5.Users can explore random news predictions or input their own summaries. <br>

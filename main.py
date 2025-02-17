import os
import feedparser
import pandas as pd
import joblib
import random
import customtkinter as ctk
from tkinter import messagebox,simpledialog
from bs4 import BeautifulSoup
import concurrent.futures
from tqdm import tqdm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ✅ CustomTkinter Tema Ayarları
ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")

# ✅ RSS Kaynakları
RSS_FEEDS = {
    "Ekonomi": "https://www.ntv.com.tr/ekonomi.rss",
    "Dünya": "https://www.ntv.com.tr/dunya.rss",
    "Eğitim": "https://www.ntv.com.tr/egitim.rss",
    "Spor": "https://www.ntv.com.tr/spor.rss",
    "Teknoloji": "https://www.ntv.com.tr/teknoloji.rss",
    "Sağlık": "https://www.ntv.com.tr/saglik.rss",
    "Otomobil": "https://www.ntv.com.tr/otomobil.rss",
    "Sanat": "https://www.ntv.com.tr/sanat.rss",
    "Seyahat": "https://www.ntv.com.tr/seyahat.rss",
    "Magazin": "https://www.ntv.com.tr/n-life.rss",
}

# ✅ HTML Temizleme Fonksiyonu
def clean_html(text):
    """HTML etiketlerini temizler ve düz metni döndürür."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ").strip()

# ✅ LSA ile Özetleme Fonksiyonu
def summarize_text_lsa(text, num_sentences=2):  # Daha kısa özetleme için 2 cümle
    """LSA algoritması ile haber metinlerini özetler."""
    parser = PlaintextParser.from_string(text, Tokenizer("turkish"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# ✅ RSS'den Haber Çekme 
def fetch_rss_news():
    """RSS kaynaklarından haberleri çeker, özetler ve sadece 'Magazin' kategorisi için özel denetim uygular."""
    news_data = []
    total_feeds = len(RSS_FEEDS)

    print("\n📡 RSS Verisi Çekiliyor...\n")

    for i, (category, url) in enumerate(RSS_FEEDS.items(), start=1):
        print(f"⏳ {category} haberleri çekiliyor... (%{(i/total_feeds)*100:.1f})")
        feed = feedparser.parse(url)

        for entry in feed.entries[:130]:
            title = entry.get("title", "Başlık Yok")
            summary = entry.get("summary", "Özet Yok")
            clean_summary = clean_html(summary)

            # 🔍 Magazin kategorisi için özel filtreleme
            if category == "Magazin":
                entry_id = entry.get("id", "").lower() 
                if "magazin" in entry_id:  # ID içinde "magazin" geçiyorsa ekle
                    lsa_summary = summarize_text_lsa(clean_summary)
                    news_data.append((title, lsa_summary, category))
            else:
                # Diğer kategoriler için doğrudan ekleme yap
                lsa_summary = summarize_text_lsa(clean_summary)
                news_data.append((title, lsa_summary, category))

    df = pd.DataFrame(news_data, columns=["title", "summary", "category"])
    df.to_csv("news_dataset.csv", index=False)
    print("✅ Veri çekme tamamlandı!")

# ✅ Farklı Modelleri Eğitme ve Karşılaştırma 
def train_and_compare_models():
    """Farklı sınıflandırma modellerini eğitir, ilerleme yüzdesi gösterir ve doğruluk oranlarını karşılaştırır."""
    df = pd.read_csv("news_dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(df["summary"], df["category"], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(kernel="linear")
    }

    results = {}
    print("\n🎯 Model eğitimi başlıyor...\n")
    
    for name, model in tqdm(models.items(), desc="📊 Model Eğitimi", unit="model"):
        model.fit(X_train_vec, y_train)  
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

    print("\n✅ Model eğitimi tamamlandı!\n")

    # En iyi modeli seç ve kaydet
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    joblib.dump(best_model, "news_classifier.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    # En iyi model bilgilerini kaydet
    with open("model_info.txt", "w") as f:
        f.write(f"{best_model_name},{results[best_model_name]}")

    return best_model_name, results[best_model_name]

# ✅ Modeli Yükle veya Eğit
def load_or_train_model():
    """Mevcut modeli yükler veya yeni model eğitir."""
    if os.path.exists("news_classifier.pkl") and os.path.exists("model_info.txt"):
        print("📢 Kayıtlı model bulundu, yükleniyor...")

        try:
            with open("model_info.txt", "r") as f:
                best_model_name, best_accuracy = f.read().split(",")
                best_accuracy = float(best_accuracy)
        except Exception as e:
            print(f"⚠️ Model bilgisi okunamadı: {e}. Model yeniden eğitiliyor...")
            best_model_name, best_accuracy = train_and_compare_models()
    else:
        print("🔄 Model bulunamadı, eğitiliyor...")
        fetch_rss_news()
        best_model_name, best_accuracy = train_and_compare_models()

    return best_model_name, best_accuracy

# ✅ Haberleri Paralel İşleyerek Hızlandırma
def process_news(title, summary, model, vectorizer):
    """Haber metnini özetleyip kategorisini tahmin eder."""
    lsa_summary = summarize_text_lsa(summary)  # LSA ile özetleme
    vectorized_text = vectorizer.transform([lsa_summary])
    predicted_category = model.predict(vectorized_text)[0]
    
    return {
        "title": title,
        "summary": lsa_summary,
        "category": predicted_category
    }
# ✅ Rastgele 10 Haberi Tahmin Et
def fetch_random_news():
    """En iyi modeli kullanarak rastgele 10 haberin kategorisini tahmin eder."""
    if not os.path.exists("news_classifier.pkl"):
        messagebox.showerror("Hata", "Model bulunamadı! Lütfen modeli eğitin.")
        return []

    model = joblib.load("news_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    all_news = []
    for category, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title", "Başlık Yok")
            summary = entry.get("summary", "Özet Yok")
            clean_summary = clean_html(summary)
            all_news.append((title, clean_summary))

    random_news = random.sample(all_news, 10)

    news_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_news = {executor.submit(process_news, title, summary, model, vectorizer): (title, summary) for title, summary in random_news}
        for future in concurrent.futures.as_completed(future_to_news):
            news_list.append(future.result())

    return news_list

# ✅ Kullanıcının Girdiği Haberi Tahmin Et
def predict_custom_news():
    """Kullanıcının girdiği haber özetini sınıflandırır."""
    if not os.path.exists("news_classifier.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
        messagebox.showerror("Hata", "Model veya vektörleyici bulunamadı! Lütfen modeli eğitin.")
        return
    
    model = joblib.load("news_classifier.pkl")  # Modeli yükle
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Vektörleyiciyi yükle
    
    user_input = simpledialog.askstring("Haber Girişi", "Lütfen haber özetini girin:")
    if not user_input:
        return
    
    processed_text = summarize_text_lsa(user_input)  # Haberi özetle
    vectorized_text = vectorizer.transform([processed_text])  # Vektörleştir
    
    predicted_category = model.predict(vectorized_text)[0]  # Model ile tahmin et
    
    messagebox.showinfo("Tahmin", f"🏷️ Tahmin Edilen Kategori: {predicted_category}")

# ✅ En İyi Modeli Yükle veya Eğit
best_model_name, best_accuracy = load_or_train_model()

# ✅ CustomTkinter GUI
class NewsApp(ctk.CTk):
    def __init__(self, model_name, accuracy):
        super().__init__()

        self.title("📢 Haber Sınıflandırma ve Özetleme")
        self.geometry("1200x800")  # Pencere büyüklüğü artırıldı

        self.model_name = model_name
        self.accuracy = accuracy

        # Ana Menü
        self.main_menu = ctk.CTkFrame(self, width=1150, height=700)
        self.main_menu.pack(pady=20, padx=20, fill="both", expand=True)

        self.label = ctk.CTkLabel(self.main_menu, text="📢 Haber Sınıflandırma ve Özetleme", font=("Arial", 24, "bold"))
        self.label.pack(pady=10)

        self.model_label = ctk.CTkLabel(self.main_menu, text=f"🏆 En İyi Model: {self.model_name} ({self.accuracy:.4f} doğruluk)", font=("Arial", 18, "bold"))
        self.model_label.pack(pady=5)

        self.random_news_btn = ctk.CTkButton(self.main_menu, text="📩 Rastgele Haber Tahmini", font=("Arial", 16), command=self.show_random_news)
        self.random_news_btn.pack(pady=20)

        self.custom_news_btn = ctk.CTkButton(self.main_menu, text="📝 Girilen Haberi Tahmin Et", font=("Arial", 16), command=self.show_custom_news)
        self.custom_news_btn.pack(pady=20)

        ### ✅ Rastgele Haber Bölümü ###
        self.random_news_frame = ctk.CTkFrame(self, width=1150, height=700)
        self.get_news_label = ctk.CTkLabel(self.random_news_frame, text="📩 Rastgele Haberler ve Tahminleri", font=("Arial", 18, "bold"))
        self.get_news_label.pack(pady=5)

        self.get_news_btn = ctk.CTkButton(self.random_news_frame, text="📩 Rastgele Haber Tahmin Et", font=("Arial", 14), command=self.get_news)
        self.get_news_btn.pack(pady=5)

        self.random_news_text = ctk.CTkTextbox(self.random_news_frame, width=1100, height=500, font=("Arial", 12))
        self.random_news_text.pack(pady=5)

        self.back_btn1 = ctk.CTkButton(self.random_news_frame, text="🔙 Geri Dön", font=("Arial", 14), command=self.show_main_menu)
        self.back_btn1.pack(pady=10)

        ### ✅ Girilen Haberi Tahmin Etme Bölümü ###
        self.user_news_frame = ctk.CTkFrame(self, width=1150, height=700)
        self.input_label = ctk.CTkLabel(self.user_news_frame, text="📝 Haber Özetini Girin ve Kategorisini Öğrenin", font=("Arial", 18, "bold"))
        self.input_label.pack(pady=5)

        self.user_input_box = ctk.CTkTextbox(self.user_news_frame, width=1100, height=100, font=("Arial", 12))
        self.user_input_box.pack(pady=5)

        self.predict_btn = ctk.CTkButton(self.user_news_frame, text="🔍 Tahmin Et", font=("Arial", 14), command=self.predict_custom_news)
        self.predict_btn.pack(pady=5)

        self.user_news_text = ctk.CTkTextbox(self.user_news_frame, width=1100, height=400, font=("Arial", 12))
        self.user_news_text.pack(pady=5)

        self.back_btn2 = ctk.CTkButton(self.user_news_frame, text="🔙 Geri Dön", font=("Arial", 14), command=self.show_main_menu)
        self.back_btn2.pack(pady=10)

    def show_main_menu(self):
        """Ana menüyü gösterir ve diğer bölümleri gizler."""
        self.random_news_frame.pack_forget()
        self.user_news_frame.pack_forget()
        self.main_menu.pack(pady=20, padx=20, fill="both", expand=True)

    def show_random_news(self):
        """Rastgele haber bölümünü gösterir ve ana menüyü gizler."""
        self.main_menu.pack_forget()
        self.user_news_frame.pack_forget()
        self.random_news_frame.pack(pady=20, padx=20, fill="both", expand=True)

    def show_custom_news(self):
        """Kullanıcının girdiği haber tahmin bölümünü gösterir ve ana menüyü gizler."""
        self.main_menu.pack_forget()
        self.random_news_frame.pack_forget()
        self.user_news_frame.pack(pady=20, padx=20, fill="both", expand=True)

    def get_news(self):
        """Rastgele haberleri tahmin eder ve ilk kutuya ekler."""
        self.random_news_text.delete("1.0", "end")
        self.random_news_text.insert("end", "⏳ Haberler yükleniyor...\n")
        self.update()

        news_data = fetch_random_news()
        self.random_news_text.delete("1.0", "end")

        for idx, news in enumerate(news_data, 1):
            self.random_news_text.insert("end", f"\n🔹 {idx}. Haber:\n📌 Başlık: {news['title']}\n📜 Özet (LSA): {news['summary']}\n🏷️ Tahmin Edilen Kategori: {news['category']}\n\n")

    def predict_custom_news(self):
        """Kullanıcının girdiği haber özetini sınıflandırır ve sonucu gösterir."""
        if not os.path.exists("news_classifier.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
            self.user_news_text.delete("1.0", "end")
            self.user_news_text.insert("end", "⚠️ Model bulunamadı! Lütfen modeli eğitin.")
            return
        
        model = joblib.load("news_classifier.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")

        user_input = self.user_input_box.get("1.0", "end").strip()
        if not user_input:
            self.user_news_text.delete("1.0", "end")
            self.user_news_text.insert("end", "⚠️ Lütfen haber özetini girin.")
            return
        
        processed_text = summarize_text_lsa(user_input)
        vectorized_text = vectorizer.transform([processed_text])
        predicted_category = model.predict(vectorized_text)[0]

        # Sonucu kutuya ekle
        self.user_news_text.delete("1.0", "end")
        self.user_news_text.insert("end", f"📌 Girilen Haber: {user_input}\n\n📜 Özet (LSA): {processed_text}\n\n🏷️ Tahmin Edilen Kategori: {predicted_category}")

# ✅ En İyi Modeli Yükle veya Eğit
best_model_name, best_accuracy = load_or_train_model()

# ✅ Uygulamayı Başlat
app = NewsApp(best_model_name, best_accuracy)
app.mainloop()

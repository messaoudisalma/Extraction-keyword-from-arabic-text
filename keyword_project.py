import streamlit as st
import nltk
from keybert import KeyBERT
from yake import KeywordExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import re
from nltk.corpus import stopwords
import pandas as pd

# Télécharger les ressources nécessaires
nltk.download('stopwords')
nltk.download('punkt')
arabic_stopwords = stopwords.words('arabic')

# Charger AraBERT
tokenizer_arabert = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabert")
model_arabert = AutoModel.from_pretrained("aubmindlab/bert-base-arabert")

# Fonction de nettoyage
def remove_foreign_words(text):
    """Supprime les mots contenant des caractères non arabes."""
    return re.sub(r'\b[^\u0600-\u06FF]+\b', ' ', text)

def remove_diacritics(text):
    """Supprime les diacritiques de l'arabe."""
    return re.sub(r'[\u064B-\u065F]', '', text)

def remove_punctuation_and_symbols(text):
    """Supprime la ponctuation et les symboles inutiles."""
    return re.sub(r'[،.؟!:\-(){}\[\];"\'~@#$%^&*_+]', '', text)

def remove_numbers_and_symbols(text):
    """Supprime les chiffres et autres symboles."""
    return re.sub(r'[0-9\u0660-\u0669\u06F0-\u06F9%$#@!&^]', '', text)

def remove_multiple_spaces(text):
    """Supprime les espaces multiples."""
    return re.sub(r'\s+', ' ', text).strip()

custom_stopwords = [
    'و', 'في', 'على', 'من', 'إلى', 'عن', 'مع', 'إن', 'إلا', 'هذا', 'تلك', 
    'ذلك', 'التي', 'التي', 'هو', 'هي', 'هم', 'أو', 'أي', 'أيضا', 'ذلك', 
    'كان', 'تكون', 'تكون', 'عندما', 'لذلك', 'لكن', 'لأن', 'هذه', 'التي', 
    'أن', 'أنت', 'نحن', 'أنتِ', 'له', 'لها', 'هم', 'علي', 'لن', 'فيما', 
    'مما', 'منذ', 'إحدى', 'إحدى', 'لا', 'ال', 'أكثر', 'أقل', 'أيضا', 'أولا',
    'في', 'سوف', 'من', 'عند', 'الذي', 'الذين', 'ثم', 'لكن', 'لم', 'لو', 'ماذا', 
    'بين', 'إذا', 'بعد', 'قبل', 'داخل', 'خارج', 'التي', 'على', 'هذه', 'ذلك', 
    'كان', 'بعض', 'كل', 'أول', 'ثاني', 'آخر', 'الأخرى', 'أنت', 'نعم', 'لا', 
    'بينما'
]

def remove_stopwords(text, custom_stopwords):
    """Supprime les stopwords personnalisés et les stopwords arabes."""
    words = text.split()
    return ' '.join(word for word in words if word not in arabic_stopwords and word not in custom_stopwords)

def clean_text(text):
    """Applique toutes les étapes de nettoyage."""
    text = remove_foreign_words(text)
    text = remove_diacritics(text)
    text = remove_punctuation_and_symbols(text)
    text = remove_numbers_and_symbols(text)
    text = remove_multiple_spaces(text)
    return text

def calculate_tfidf(corpus, top_n=10):
    """Applique le modèle TF-IDF au texte et retourne les mots-clés et leurs scores."""
    vectorizer = TfidfVectorizer(stop_words=arabic_stopwords, max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    
    # Trier les mots-clés par score décroissant
    keywords_scores = list(zip(feature_names, scores))
    sorted_keywords_scores = sorted(keywords_scores, key=lambda x: x[1], reverse=True)
    
    return sorted_keywords_scores

# Fonction YAKE modifiée pour retourner les scores et trier par ordre croissant
def extract_with_yake(text, top_n=10):
    """Extrait des mots-clés en utilisant YAKE avec calcul du score et tri par score croissant."""
    extractor = KeywordExtractor(lan="ar", n=1, top=top_n)
    keywords_with_scores = extractor.extract_keywords(text)
    
    # Trier les mots-clés par score croissant
    sorted_keywords = sorted(keywords_with_scores, key=lambda x: x[1], reverse=False)
    
    return sorted_keywords

# Fonction KeyBERT avec DistilBERT
def extract_with_keybert_DistilBERT(text, top_n=10):
    """Extrait des mots-clés en utilisant KeyBERT avec DistilBERT."""
    kw_model = KeyBERT(model='distilbert-base-nli-mean-tokens')  # Modèle DistilBERT pour KeyBERT
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    
    # Supprimer les doublons tout en conservant le score le plus élevé pour chaque mot-clé
    unique_keywords = {}
    for keyword, score in keywords:
        if keyword not in unique_keywords or score > unique_keywords[keyword]:
            unique_keywords[keyword] = score

    # Retourner les mots-clés sans doublons, triés par score décroissant
    sorted_keywords = sorted(unique_keywords.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords

def extract_with_keybert_AraBERT(text, top_n=10):
    """Extrait des mots-clés en utilisant KeyBERT avec AraBERT."""
    # Charger le modèle AraBERT
    kw_model = KeyBERT(model='asafaya/bert-base-arabic')  # Utilisation d'AraBERT
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    
    # Supprimer les doublons tout en conservant le score le plus élevé pour chaque mot-clé
    unique_keywords = {}
    for keyword, score in keywords:
        if keyword not in unique_keywords or score > unique_keywords[keyword]:
            unique_keywords[keyword] = score

    # Retourner les mots-clés sans doublons, triés par score décroissant
    sorted_keywords = sorted(unique_keywords.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords

# Fonction pour extraire les mots-clés avec KeyBERT utilisant le modèle XLM-RoBERTa
def extract_with_keybert_XLMRoBerta(text, top_n=10):
    """Extrait des mots-clés en utilisant KeyBERT avec le modèle XLM-RoBERTa."""
    kw_model = KeyBERT(model='xlm-roberta-base')  # Utilisation du modèle XLM-RoBERTa
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    
    # Supprimer les doublons tout en conservant le score le plus élevé pour chaque mot-clé
    unique_keywords = {}
    for keyword, score in keywords:
        if keyword not in unique_keywords or score > unique_keywords[keyword]:
            unique_keywords[keyword] = score

    # Retourner les mots-clés sans doublons, triés par score décroissant
    sorted_keywords = sorted(unique_keywords.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords

# Fonction pour calculer les scores AraBERT
def calculate_arabert_scores(words):
    scores = {}
    for word in words:
        inputs = tokenizer_arabert(word, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model_arabert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # Embedding de la [CLS] token
        scores[word] = cls_embedding.mean().item()  # Retourne la moyenne des valeurs de l'embedding
    return scores

# Fonction combinée TF-IDF + AraBERT
def extract_with_tfidf_arabert(text, top_n=10):
    """Extrait des mots-clés en combinant TF-IDF et AraBERT."""
    # Nettoyage du texte
    cleaned_text = clean_text(text)
    
    # Extraction des mots-clés avec TF-IDF
    tfidf_keywords = calculate_tfidf([cleaned_text], top_n)
    
    # Extraction des scores AraBERT pour chaque mot-clé
    arabert_scores = calculate_arabert_scores([word for word, _ in tfidf_keywords])
    
    # Combinaison des scores TF-IDF et AraBERT
    combined_scores = {
        word: tfidf_score + arabert_scores.get(word, 0)
        for word, tfidf_score in tfidf_keywords
    }
    
    # Trier les mots-clés par score combiné
    sorted_keywords = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_keywords[:top_n]

def extract_with_tfidf_yake_arabert(text, top_n=10):
    """Extrait les mots-clés avec une combinaison de TF-IDF, YAKE et AraBERT."""
    # Nettoyage du texte
    cleaned_text = clean_text(text)
    
    # Extraction des mots-clés avec TF-IDF
    tfidf_keywords = calculate_tfidf([cleaned_text], top_n)
    
    # Extraction des scores AraBERT pour chaque mot-clé
    arabert_scores = calculate_arabert_scores([word for word, _ in tfidf_keywords])
    
    # Extraction des mots-clés avec YAKE
    yake_keywords = extract_with_yake(cleaned_text, top_n)
    
    # Combinaison des scores TF-IDF, YAKE et AraBERT
    combined_scores = {}
    for word, tfidf_score in tfidf_keywords:
        # Chercher le score YAKE et AraBERT pour chaque mot
        yake_score = next((score for keyword, score in yake_keywords if word in keyword), 0)
        arabert_score = arabert_scores.get(word, 0)
        combined_scores[word] = tfidf_score + yake_score + arabert_score
    
    # Trier les mots-clés par score combiné
    sorted_keywords = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_keywords[:top_n]

# Ajouter une icône en lien avec l'analyse textuelle et l'extraction de mots-clés
st.markdown('<h1 style="display: flex; align-items: center;">'
            '<img src="https://cdn-icons-png.flaticon.com/512/942/942748.png" alt="keyword-icon" width="40" style="margin-right: 10px;">'
            'Extraction de Mots-Clés</h1>',
            unsafe_allow_html=True)

st.markdown("---")

# Ajouter une partie description
st.markdown("""
### Description
Cette interface permet de comparer différentes méthodes pour l'extraction de mots-clés à partir d'un texte en arabe. 
Vous pouvez saisir un texte, choisir parmi plusieurs algorithmes comme KeyBERT et YAKE, 
et visualiser les mots-clés extraits pour mieux comprendre et analyser les résultats.
""")


# Menu horizontal avec des boutons pour sélectionner le modèle
st.subheader("Choisissez un modèle pour l'extraction des mots-clés")
# Première ligne avec 4 colonnes
col1, col2, col3, col4 = st.columns(4)

with col1:
    yake_selected = st.button("YAKE")
with col2:
    tfidf_selected = st.button("TF-IDF")
with col3:
    keybert_selected = st.button("KeyBERT + DistilBERT")
with col4:
    tfidf_arabert_selected = st.button("TF-IDF + AraBERT")

# Deuxième ligne avec 3 colonnes
col5, col6, col7 = st.columns(3)

with col5:
    keybert_arabert_selected = st.button("KeyBERT + AraBERT")
with col6:
    keybert_xlm_selected = st.button("KeyBERT + XLM-RoBERTa")
with col7:
    tfidf_yake_arabert_selected = st.button("TF-IDF + Yake + AraBERT")
# Déterminer le modèle choisi
if yake_selected:
    model_choice = "YAKE"
elif tfidf_selected:
    model_choice = "TF-IDF"
elif keybert_selected:
    model_choice = "KeyBERT + DistilBERT"
elif tfidf_arabert_selected:
    model_choice = "TF-IDF + AraBERT"
elif keybert_arabert_selected:
    model_choice = "KeyBERT + AraBERT"
elif keybert_xlm_selected:
    model_choice = "KeyBERT + XLM-RoBERTa"
elif tfidf_yake_arabert_selected:
    model_choice = "TF-IDF + Yake + AraBERT"
else:
    model_choice = None

st.markdown("---")

# Textes prédéfinis dans la barre latérale
st.sidebar.subheader("Choisissez un texte prédéfini")
texts = {
    "التعليم": "التعليم هو أساس تقدم الأمم وازدهارها. يعتبر التعليم حقًا من حقوق الإنسان الأساسية، فهو يعزز من مهارات الأفراد ويزيد من فرصهم في الحياة. كما يسهم التعليم في بناء مجتمعات متقدمة تتمتع بالاستقرار الاقتصادي والاجتماعي. من خلال التعليم، يمكن للأفراد اكتساب المعرفة اللازمة للمشاركة الفعالة في مختلف المجالات. لذلك، يجب على الحكومات أن تركز على تطوير التعليم وتوفير الفرص التعليمية لجميع المواطنين. التعليم لا يقتصر على المدارس والجامعات، بل يمتد إلى الحياة اليومية والتعلم المستمر.",
    
    "الاقتصاد": "الاقتصاد هو العلم الذي يدرس كيفية استخدام الموارد المحدودة لتلبية احتياجات الأفراد والمجتمعات. يتعامل مع الإنتاج، والتوزيع، والاستهلاك، والتبادل للسلع والخدمات. يتأثر الاقتصاد بالعديد من العوامل مثل السياسات الحكومية، والطلب والعرض، والتطورات التكنولوجية. تعد التحديات الاقتصادية مثل التضخم، والبطالة، وعدم المساواة من القضايا الهامة التي يجب معالجتها. النمو الاقتصادي يساهم في تحسين مستويات المعيشة ويوفر فرص عمل جديدة. في عصر العولمة، أصبحت الاقتصادات العالمية مترابطة بشكل أكبر.",
    
    "التكنولوجيا": "التكنولوجيا تؤثر بشكل كبير على حياتنا اليومية، حيث أصبحت جزءاً أساسياً من العمليات في جميع المجالات. من الأجهزة الذكية إلى الإنترنت، أصبحت التكنولوجيا توفر حلولًا مبتكرة للعديد من التحديات. في مجال التعليم، تمكّن التكنولوجيا من الوصول إلى المعلومات بسرعة وسهولة. في الطب، ساعدت الابتكارات التكنولوجية في تحسين تشخيص الأمراض وعلاجها. علاوة على ذلك، غيرت التكنولوجيا بشكل جذري طريقة العمل والتواصل بين الأفراد. المستقبل يحمل المزيد من الابتكارات التي ستؤثر على حياتنا بطرق غير مسبوقة.",
    
    "البيئة": "البيئة هي نظام متكامل يعتمد على توازن عناصره للحفاظ على الحياة. تعد قضايا التلوث والتغير المناخي تحديات عالمية تتطلب تعاونًا دوليًا لمواجهتها. التلوث الناتج عن النشاطات البشرية يؤثر سلبًا على الهواء والماء والتربة، مما يؤدي إلى تدهور النظام البيئي. من المهم الحفاظ على التنوع البيولوجي وحماية الأنواع المهددة بالانقراض. يعد التحول إلى الطاقة المتجددة خطوة أساسية نحو الحد من الانبعاثات الكربونية. الحفاظ على البيئة يتطلب تغييرات في السلوكيات الفردية والجماعية من أجل حماية كوكبنا للأجيال القادمة.",
    
    "الصحة": "الصحة هي أحد أعظم النعم التي يتمتع بها الإنسان. يتطلب الحفاظ على الصحة اتباع نمط حياة متوازن يشمل التغذية السليمة، ممارسة الرياضة بانتظام، والابتعاد عن العادات الضارة مثل التدخين. كما أن العناية بالصحة النفسية تعد أمرًا أساسيًا، إذ تؤثر بشكل مباشر على جودة حياة الفرد. يجب أن توفر المجتمعات أنظمة صحية قوية تضمن الرعاية الصحية للجميع. الابتكار في المجال الطبي يسهم في تطوير العلاجات والتشخيصات، ما يجعل علاج الأمراض أكثر فاعلية. الوقاية خير من العلاج، ولذلك يجب على الجميع اتخاذ التدابير اللازمة للحفاظ على صحتهم.",
    
    "الثقافة": "الثقافة هي مجموع المعارف والعادات والمعتقدات التي يتبناها مجتمع معين. تعكس الثقافة هوية الشعب وتشكل طريقة تفكيرهم وتفاعلاتهم مع العالم. تعد اللغة، الأدب، الفنون، والعادات الاجتماعية جزءًا أساسيًا من الثقافة. المحافظة على التراث الثقافي ضروري للحفاظ على الهوية الوطنية وتعزيز التفاهم بين الشعوب. الثقافة أيضًا تلعب دورًا في تحفيز الابتكار والتطوير في مجالات متعددة. من خلال تعزيز الثقافة، يمكن للفرد أن يطور تفكيره ويساهم في إغناء المجتمع.",
    
    "الرياضة": "الرياضة تعتبر جزءًا أساسيًا من حياة الإنسان، حيث تساهم في تعزيز اللياقة البدنية وتحسين الصحة العامة. الرياضة تخلق بيئة تنافسية يمكن أن تنمي من قدرات الأفراد وتعلمهم العمل الجماعي والصبر. كما أنها تلعب دورًا في تقوية العلاقات بين الدول من خلال المنافسات الرياضية الدولية. الرياضة ليست فقط للأشخاص المحترفين، بل يجب أن تكون جزءًا من روتين الحياة اليومية لكل فرد. ممارسة الرياضة تساعد في الوقاية من الأمراض المزمنة وتحسن من الصحة النفسية أيضًا.",
    
    "السفر": "السفر هو وسيلة لاكتشاف ثقافات وأماكن جديدة، ويمنح الفرصة للتعلم والتجارب الفريدة. من خلال السفر، يمكن للمرء أن يتعرف على تقاليد وأسلوب حياة شعوب أخرى، مما يساعد على توسيع الأفق الفكري. السفر يساعد أيضًا في تطوير مهارات التواصل والقدرة على التكيف مع بيئات جديدة. إن السفر لا يتعلق فقط بزيارة أماكن سياحية، بل هو أيضًا فرصة للاستراحة من الروتين اليومي واستكشاف عالم جديد. كما يمكن أن يعزز من الإبداع والابتكار من خلال التعرض لتجارب غير مألوفة.",
    
    "الفن": "الفن هو تعبير عن الإبداع والتصورات الإنسانية باستخدام وسائل متعددة مثل الرسم، النحت، والموسيقى. الفن له تأثير كبير في نقل المشاعر والأفكار، ويمكن أن يكون وسيلة للتواصل بين الأفراد من ثقافات مختلفة. يعتبر الفن جزءًا من التراث الثقافي ويعكس قيم المجتمعات عبر العصور. إضافة إلى ذلك، يمكن للفن أن يكون وسيلة للتمرد والتغيير الاجتماعي من خلال توجيه انتقادات للأنظمة أو الأحداث. يعزز الفن من الإبداع ويشجع على التفكير النقدي.",
    
    "التاريخ": "التاريخ هو سجل للأحداث والتطورات التي مرت بها البشرية، ويعكس الأحداث التي شكلت الحاضر والمستقبل. دراسة التاريخ تمنح الأفراد فهماً عميقاً للمجتمعات المختلفة وكيفية تطورها. من خلال دراسة التاريخ، يمكن للمرء أن يفهم الأخطاء الماضية ويتجنب تكرارها في المستقبل. التاريخ يساعد في بناء الهوية الوطنية ويزيد من التفاهم بين الشعوب. كما أنه يلعب دورًا في تعزيز التسامح والسلام في العالم من خلال معرفة تجارب الآخرين.",
    
    "الذكاء الاصطناعي": "الذكاء الاصطناعي هو مجال من مجالات علوم الكمبيوتر الذي يسعى إلى إنشاء أنظمة قادرة على محاكاة الذكاء البشري. يشمل الذكاء الاصطناعي التعلم الآلي، معالجة اللغة الطبيعية، الرؤية الحاسوبية، والروبوتات. أصبح الذكاء الاصطناعي جزءًا من حياتنا اليومية من خلال التطبيقات مثل المساعدات الصوتية، السيارات ذاتية القيادة، والتوصيات المخصصة. يسهم الذكاء الاصطناعي في تحسين العديد من الصناعات مثل الرعاية الصحية، التمويل، والتعليم. مع تطور الذكاء الاصطناعي، تزداد الأسئلة حول تأثيراته على سوق العمل والخصوصية.",
    
    "الفضاء": "الفضاء هو الكون الذي يحتوي على النجوم والكواكب والمجرات والعديد من الأجرام السماوية. منذ العصور القديمة، كان الفضاء مصدرًا للدهشة والفضول. عبر التقدم العلمي والتكنولوجي، تمكن الإنسان من استكشاف الفضاء، بدءًا من إرسال الأقمار الصناعية إلى المريخ إلى إرسال مركبات فضائية إلى الكواكب البعيدة. إن دراسة الفضاء توفر لنا فهمًا أعمق لكوننا والمكان الذي نعيش فيه. الفضاء ليس مجرد مجال للاكتشافات العلمية، بل يحمل أيضًا إمكانيات للابتكار والتقدم في العديد من المجالات مثل الاتصالات والملاحة.",
    
    "العدالة الاجتماعية": "العدالة الاجتماعية هي مبدأ يهدف إلى تحقيق المساواة بين الأفراد في الحقوق والفرص. يتضمن ذلك محاربة التمييز في جميع أشكاله، سواء كان عنصريًا، دينيًا، أو اقتصاديًا. العدالة الاجتماعية تدعو إلى توزيع عادل للموارد والفرص، بما في ذلك التعليم، الرعاية الصحية، والعمل. تعتبر العدالة الاجتماعية حجر الزاوية لتحقيق التنمية المستدامة، حيث تساهم في تعزيز الاستقرار الاجتماعي والتقليل من التوترات بين الطبقات المختلفة. من خلال تعزيز العدالة الاجتماعية، يمكن بناء مجتمعات أكثر إنصافًا ورفاهية."
}

# Affichage des options de texte dans la barre latérale
chosen_topic = st.sidebar.radio("Sujets disponibles :", list(texts.keys()))

# Champ de saisie pour texte personnalisé
st.subheader("Entrez un texte ou utilisez le texte sélectionné")
text_input = st.text_area(
    "Texte (si vous ne saisissez rien, le texte prédéfini sera utilisé) :", value=""
)

# Utiliser le texte saisi ou le texte prédéfini
if text_input.strip():
    selected_text = text_input
else:
    selected_text = texts[chosen_topic]

# Afficher le texte utilisé
st.write("Texte utilisé :")
st.write(selected_text)

# Nettoyage du texte
cleaned_text = clean_text(selected_text)
cleaned_text_without_stopwords = remove_stopwords(cleaned_text, custom_stopwords)

st.write("Texte nettoyé sans stopwords :")
st.write(cleaned_text_without_stopwords)

# Extraction des mots-clés selon le modèle choisi
if model_choice == "YAKE":
    yake_keywords_with_scores = extract_with_yake(cleaned_text_without_stopwords)
    st.write("Mots-clés extraits avec YAKE (triés par score décroissant) :")
    st.table(pd.DataFrame(yake_keywords_with_scores, columns=["Mot-Clé", "Score"]))

elif model_choice == "TF-IDF":
    tfidf_keywords = calculate_tfidf([cleaned_text_without_stopwords], top_n=10)
    st.write("Mots-clés extraits avec TF-IDF (les 10 premiers) :")
    st.table(pd.DataFrame(tfidf_keywords, columns=["Mot-Clé", "Score TF-IDF"]))

# Afficher les résultats avec Streamlit
elif model_choice == "KeyBERT + DistilBERT":
    keybert_DistilBERT_keywords = extract_with_keybert_DistilBERT(cleaned_text_without_stopwords)
    st.write("Mots-clés extraits avec KeyBERT utilisant DistilBERT (sans doublons):")
    st.table(pd.DataFrame(keybert_DistilBERT_keywords, columns=["Mot-Clé", "Score"]))

# Extraction des mots-clés selon le modèle choisi
elif model_choice == "TF-IDF + AraBERT":
    top_keywords = extract_with_tfidf_arabert(cleaned_text_without_stopwords, top_n=10)
    
    keywords_df = pd.DataFrame(top_keywords, columns=["Mot-Clé", "Score"])

    # Afficher le tableau dans Streamlit
    st.write("### Mots-clés extraits avec TF-IDF + AraBERT:")
    st.table(keywords_df)

elif model_choice == "KeyBERT + AraBERT":
    keybert_AraBERT_keywords = extract_with_keybert_AraBERT(cleaned_text_without_stopwords, top_n=10)
    st.write("Mots-clés extraits avec KeyBERT utilisant AraBERT (sans doublons):")
    st.table(pd.DataFrame(keybert_AraBERT_keywords, columns=["Mot-Clé", "Score"]))

# Si l'utilisateur a choisi XLM-RoBERTa, extraire les mots-clés
elif model_choice == "KeyBERT + XLM-RoBERTa":
    keybert_XLMRoBerta_keywords = extract_with_keybert_XLMRoBerta(cleaned_text_without_stopwords, top_n=10)
    st.write("Mots-clés extraits avec KeyBERT utilisant XLM-RoBERTa (sans doublons):")
    st.table(pd.DataFrame(keybert_XLMRoBerta_keywords, columns=["Mot-Clé", "Score"]))

elif model_choice == "TF-IDF + Yake + AraBERT":
    keybert_tfidf_yake_arabert_keywords = extract_with_tfidf_yake_arabert(cleaned_text_without_stopwords, top_n=10)
    st.write("Mots-clés extraits avec TF-IDF et Yake et AraBERT:")
    st.table(pd.DataFrame(keybert_tfidf_yake_arabert_keywords, columns=["Mot-Clé", "Score"]))
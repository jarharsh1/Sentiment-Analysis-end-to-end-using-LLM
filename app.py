from flask import Flask, request, render_template
import matplotlib
matplotlib.use('Agg') # Use 'Agg' for non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import patches
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import nltk
from nltk.corpus import stopwords
import string, json, re, io, base64, openai, random, torch
from datetime import datetime
import matplotlib.colors as mcolors

# Ensure you have your OpenAI API key in static/keys.py
# Example static/keys.py:
# mykey = "YOUR_OPENAI_API_KEY"
try:
    from static.keys import mykey
    client = openai.OpenAI(api_key=mykey)
except ImportError:
    print("Error: static/keys.py not found or 'mykey' not defined.")
    print("Please create static/keys.py with: mykey = 'YOUR_OPENAI_API_KEY'")
    client = None # Set client to None if key is not found to prevent errors

app = Flask(__name__)

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load sentiment model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# List of common professions/domains to exclude from word classification
PROFESSIONS_TO_EXCLUDE = set([
    "ophthalmology", "dentistry", "neurology", "cardiology", "pediatrics",
    "oncology", "radiology", "surgery", "dermatology", "psychiatry",
    "gastroenterology", "urology", "orthopedics", "endocrinology", "nephrology",
    "rheumatology", "pulmonology", "geriatrics", "anesthesiology", "pathology",
    "emergency", "medicine", "nursing", "pharmacy", "therapy", "physiotherapy",
    "paediatrics","kamal","aster","qusais","dafza","metro","al","dr","seher","gastroenterologist",
    "engineer", "doctor", "lawyer", "teacher", "artist", "developer",
    "analyst", "consultant", "manager", "accountant", "architect",
    "designer", "scientist", "researcher", "technician", "electrician",
    "plumber", "carpenter", "chef", "pilot", "driver", "police", "firefighter",
    "paramedic", "veterinarian", "librarian", "journalist", "photographer",
    "musician", "dancer", "actor", "athlete", "coach", "trainer", "salesperson",
    "marketer", "recruiter", "strategist", "economist", "historian", "philosopher",
    "geologist", "biologist", "chemist", "physicist", "mathematician", "statistician",
    "it", "developer", "software", "hardware", "coder", "programmer", "architect", "engineer",
    "clinic", "hospital", "patient", "medical", "staff", "specialist", "advisor" # Added more common service words
])


def get_sentiment(text):
    # Handle empty or very short text gracefully
    if not text or len(text.strip()) < 3: # Min length for meaningful analysis
        return "Neutral", 50.0, "😐" # Return 50% for neutral

    # Encode text for the sentiment model
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**encoded)
    probs = softmax(output.logits[0].numpy())

    # For 'cardiffnlp/twitter-roberta-base-sentiment-latest':
    # probs[0] is probability for NEGATIVE
    # probs[1] is probability for NEUTRAL
    # probs[2] is probability for POSITIVE

    neg_prob = probs[0].item()
    neu_prob = probs[1].item()
    pos_prob = probs[2].item()

    # Determine the sentiment label based on the highest probability
    if pos_prob > neg_prob and pos_prob > neu_prob:
        label = 'Positive'
    elif neg_prob > pos_prob and neg_prob > neu_prob:
        label = 'Negative'
    else:
        label = 'Neutral' # If neutral is highest, or if there's a tie/close call

    # Calculate a single score for the gauge (0-100 scale)
    score = ((pos_prob - neg_prob + 1) / 2) * 100

    # Ensure score is within 0-100 range due to floating point arithmetic
    score = max(0.0, min(100.0, score))

    emoji = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😟'}.get(label, '❓')
    
    # Return the label and the score (0-100)
    return label, score, emoji

def get_impactful_negative_neutral_words(text):
    if not client: # Check if OpenAI client is initialized
        print("OpenAI client not initialized. Cannot get word classifications.")
        return {"impactful_words": [], "negative_words": [], "neutral_words": []}

    prompt = f"""
    Analyze the following feedback and return a valid JSON with:
    - impactful_words: Strong positive or action-oriented words (e.g., improve, excellent, resolve, recommend, enhance, strong, great, valuable).
    - negative_words: Strongly negative, critical, or problem-related words (e.g., delay, poor, issue, problematic, terrible, lacking, broken, difficult, unacceptable).
    - neutral_words: Words that are not stopwords, not impactful, not negative, and not numbers, percentages, names of people/places/professions, but still carry descriptive or factual meaning about the subject (e.g., "aster","service", "report", "system", "feedback", "process", "time", "experience").
    - names: Specific names of people, places, organizations, or very specific technical/professional terms that should be excluded from impactful, negative, or neutral lists, and are not common English words (e.g., "Aster", "John", "Dr. Smith", "Ophthalmology Department").

    Important Rules:
    - Do NOT include generic stopwords (e.g., "the", "for", "and", "is", "a", "an", "it", "that", "this", "them", "their", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "not", "no", "yes", "but", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "hereby", "thereby", "wherein", "whereupon") in any list.
    - Do NOT include common professions, medical specialities, or general job titles (e.g., "doctor", "engineer", "nurse", "manager", "IT", "staff", "specialist") in any list. Use the 'names' list for highly specific, proper nouns that might be professions (e.g., "Dr. John Doe's Clinic").
    - Each word must be a single, lowercase word. Do not include duplicates across lists or within a list.
    - Output ONLY the valid JSON object. No other text.

    Text: '''{text}'''

    Format:
    {{
      "impactful_words": ["..."],
      "negative_words": ["..."],
      "neutral_words": ["..."],
      "names": ["..."]
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, # Keep low for consistent classification
            response_format={"type": "json_object"} # Force JSON output
        )
        content = response.choices[0].message.content
        data = json.loads(content)

        impactful = set(data.get("impactful_words", []))
        negative = set(data.get("negative_words", []))
        neutral = set(data.get("neutral_words", []))
        names = set(data.get("names", [])) 

        all_excluded_words = stop_words | PROFESSIONS_TO_EXCLUDE | names

        impactful = {w.strip().lower() for w in impactful if isinstance(w, str) and w.strip().lower() not in all_excluded_words}
        negative = {w.strip().lower() for w in negative if isinstance(w, str) and w.strip().lower() not in all_excluded_words}
        neutral = {w.strip().lower() for w in neutral if isinstance(w, str) and w.strip().lower() not in all_excluded_words}

        impactful -= (negative | neutral)
        negative -= (impactful | neutral)
        neutral -= (impactful | negative)

        return {
            "impactful_words": list(impactful),
            "negative_words": list(negative),
            "neutral_words": list(neutral)
        }
    except json.JSONDecodeError as je:
        print(f"[Error - JSON Parsing Word Classification] {je} - Content: {content}")
        return {"impactful_words": [], "negative_words": [], "neutral_words": []}
    except Exception as e:
        print("[Error - Word Classification]", e)
        return {"impactful_words": [], "negative_words": [], "neutral_words": []}

def generate_wordcloud_equal_size(text, impactful_words, negative_words, font_size=50):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    all_excluded_for_cloud = stop_words | PROFESSIONS_TO_EXCLUDE
    
    filtered = [w for w in words if w not in all_excluded_for_cloud and len(w) > 2]

    freq_dict = {w: 1 for w in filtered}

    impactful_set = set(impactful_words)
    negative_set = set(negative_words)
    
    impactful_set -= negative_set 

    impactful_colors = ["#4CAF50", "#8BC34A", "#66BB6A", "#7CB342", "#C8E6C9"] # Shades of Green
    negative_colors = ["#F44336", "#E57373", "#EF5350", "#D32F2F", "#FFCDD2"] # Shades of Red

    def color_func(word, **kwargs):
        if word in impactful_set:
            return random.choice(impactful_colors)
        elif word in negative_set:
            return random.choice(negative_colors)
        return "grey" # Default for neutral/unclassified words

    wc = WordCloud(
        width=650, height=320, background_color='white',
        color_func=color_func, min_font_size=font_size,
        max_font_size=font_size, font_step=1, margin=2,
        collocations=False
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    linewidth=2, edgecolor='black', facecolor='none'))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_data

def get_recommendations(text, original_sentiment_label, original_sentiment_score):
    if not client: # Check if OpenAI client is initialized
        print("OpenAI client not initialized. Cannot get recommendations.")
        return ["OpenAI API key not configured. Cannot generate recommendations."]

    original_text_length = len(text)
    
    # Define length constraints for the LLM prompt (more flexible)
    min_len_for_llm_prompt = max(20, 20) 
    max_len_for_llm_prompt = original_text_length + 10 # Allow some room for LLM creativity

    # Define strict length constraints for filtering the final recommendations
    strict_min_len_filter = max(5, 15) # Minimum practical length for a recommendation
    strict_max_len_filter = original_text_length+10 # User's rule: "0 to input_txt length only"

    prompt = f"""
    You are an expert communications specialist and a compassionate advisor.
    Your task is to review the following feedback or statement and generate 5 distinct, improved versions.
    The goal is to make each suggestion more:
    - **Emotionally intelligent:** Show empathy, understanding, or a more positive outlook.
    - **Clear and Concise:** Maintain clarity while refining language.
    - **Proactive or constructive:** If the original is negative, transform it into a solution-oriented or positive statement. If positive, enhance its impact.
    - **Natural and Human-like:** Avoid robotic or overly formal phrasing. Imagine how a skilled human would rephrase it.
    - **Action-oriented (if applicable):** Suggest ways to move forward positively.

    Constraints for each suggestion (LLM's internal guideline, we'll filter strictly later):
    - Aim for a length between {min_len_for_llm_prompt} and {max_len_for_llm_prompt} characters.
    - Preserve the core topic and intent of the original message.
    - Ensure grammatical correctness and professional tone where appropriate.
    - Provide each recommendation on a new line, **without any leading numbers or bullet points.**

    Original Text:
    '''{text}'''

    Rewritten Suggestions:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,       # Increased for more creativity and human-like variations
            top_p=1.0,             # Allows sampling from the entire probability distribution for diversity
            frequency_penalty=0.8, # Penalize new tokens based on their existing frequency to reduce repetition
            presence_penalty=0.8   # Penalize new tokens based on whether they appear in the text to encourage new ideas
        )
        raw = response.choices[0].message.content.strip()
        print("[Raw GPT Recommendations]", raw)

        lines = [line.strip() for line in raw.split('\n') if line.strip()]
        
        tier1_recos = [] # For recommendations meeting both length and sentiment criteria
        tier2_recos = [] # For recommendations meeting sentiment criteria but not length

        for line in lines:
            rec_label, rec_score, _ = get_sentiment(line)

            is_improvement = False
            # Determine if sentiment is improved
            if original_sentiment_label == 'Negative':
                if rec_score > original_sentiment_score:
                    is_improvement = True
            elif original_sentiment_label == 'Neutral':
                # An improvement if it moves towards positive, or is more confidently neutral (>= original_score)
                if rec_label == 'Positive' or (rec_label == 'Neutral' and rec_score >= original_sentiment_score):
                    is_improvement = True
            elif original_sentiment_label == 'Positive':
                if rec_score > original_sentiment_score:
                    is_improvement = True

            if is_improvement:
                meets_length_criteria = (len(line) >= strict_min_len_filter and len(line) <= strict_max_len_filter)
                cleaned_line = re.sub(r'^\s*\d+\.\s*', '', line).strip()

                if meets_length_criteria:
                    tier1_recos.append((rec_score, cleaned_line))
                else:
                    tier2_recos.append((rec_score, cleaned_line))

        # Sort both tiers by sentiment score (descending)
        tier1_recos.sort(key=lambda x: x[0], reverse=True)
        tier2_recos.sort(key=lambda x: x[0], reverse=True)

        final_recos = []
        # Get up to 3 recommendations from Tier 1 (meeting all criteria)
        for score, reco_text in tier1_recos:
            if len(final_recos) < 3:
                final_recos.append(reco_text)
            else:
                break
        
        # If less than 3 recommendations, fill from Tier 2 (sentiment improved, but length not met)
        for score, reco_text in tier2_recos:
            if len(final_recos) < 3:
                final_recos.append(reco_text)
            else:
                break

        return final_recos if final_recos else ["No improved recommendations found. Try a different input or adjust thresholds."]

    except Exception as e:
        print("[Error - Recommendations]", e)
        return ["Recommendation error: " + str(e)]

def get_theme(text):
    if not client: # Check if OpenAI client is initialized
        print("OpenAI client not initialized. Cannot get theme.")
        return {
            "theme": "OpenAI API key not configured.",
            "suggestions": ["Please set your API key in static/keys.py."]
        }

    prompt = f"""
    You are a tone and communication coach. Your task is to analyze the input below and return a concise summary of the central theme and 3 clear suggestions to improve the message.

    Guidelines:
    - Keep the theme to a 6–12 word summary of the main intent or tone.
    - Each suggestion should improve the clarity, tone, structure, or emotional impact.
    - Focus on message refinement, not rewriting.
    - Return ONLY the JSON object.

    Input:
    '''{text}'''

    Return JSON in the exact format:
    {{
      "theme": "<Concise theme>",
      "suggestions": [
        "<Tip 1>",
        "<Tip 2>",
        "<Tip 3>"
      ]
    }}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, # Moderate temperature for balanced creativity and accuracy
            response_format={"type": "json_object"} # Force JSON output
        )
        content = res.choices[0].message.content.strip()
        data = json.loads(content)
        if "theme" in data and isinstance(data.get("suggestions"), list) and len(data["suggestions"]) <= 3:
            return data
        else:
            raise ValueError(f"Invalid JSON structure received for theme and suggestions: {data}")
    except json.JSONDecodeError as je:
        print(f"[Error - JSON Parsing Theme] {je} - Content: {content}")
        return {
            "theme": "Error generating theme (JSON parsing failed)",
            "suggestions": ["Ensure valid input text.", "Check API response format."]
        }
    except Exception as e:
        print("[Error - Theme]", e)
        return {
            "theme": "Error generating theme",
            "suggestions": ["Try refining the input.", "Check server logs for details."]
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    result_data = None
    if request.method == 'POST':
        text = request.form['feedback']
        exclude_names_from_cloud = request.form.get('exclude_names') == 'on' 

        sentiment_label, sentiment_score, emoji = get_sentiment(text)

        words_classified = get_impactful_negative_neutral_words(text)
        impactful = set(words_classified["impactful_words"])
        negative = set(words_classified["negative_words"])
        neutral = set(words_classified["neutral_words"])

        cloud_img = generate_wordcloud_equal_size(text, list(impactful), list(negative))

        recos = get_recommendations(text, sentiment_label, sentiment_score)
        
        theme_data = get_theme(text)

        result_data = {
            "text": text,
            "sentiment": sentiment_label,
            "score": round(sentiment_score, 2),
            "emoji": emoji,
            "impactful": ", ".join(sorted(list(impactful))), 
            "negative": ", ".join(sorted(list(negative))),
            "neutral": ", ".join(sorted(list(neutral))),
            "cloud_img": cloud_img,
            "recos": recos,
            "theme": theme_data.get("theme", "N/A"),
            "theme_suggestions": theme_data.get("suggestions", [])
        }

    return render_template('index.html', result=result_data, date=datetime.now().strftime('%d %b %Y'))

if __name__ == '__main__':
    print("[App Started] Running on http://127.0.0.1:5000")
    app.run(debug=True)
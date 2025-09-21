import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from pdf2image import convert_from_bytes

st.set_page_config(page_title="OMR Evaluation System", layout="wide")

st.sidebar.title("OMR System Inputs")
keyset_input = st.sidebar.text_area(
    "Paste the answer key here (one per line, e.g. 1 - a):",
    height=250,
    help="Enter question number and answer choice separated by a dash. For multiple correct answers separate choices by commas."
)
uploaded_file = st.sidebar.file_uploader(
    "Upload OMR sheet image or PDF...",
    type=["jpg", "jpeg", "png", "pdf"]
)

st.title("📋 Automated OMR Evaluation System")

def show_confetti():
    confetti_html = """
    <html>
    <head>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    </head>
    <body>
    <script>
    var duration = 2 * 1000;
    var animationEnd = Date.now() + duration;
    var defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 1000 };

    function randomInRange(min, max) {
      return Math.random() * (max - min) + min;
    }

    var interval = setInterval(function() {
      var timeLeft = animationEnd - Date.now();

      if (timeLeft <= 0) {
        return clearInterval(interval);
      }

      var particleCount = 50 * (timeLeft / duration);
      confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 } }));
      confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 } }));
    }, 250);
    </script>
    </body>
    </html>
    """
    components.html(confetti_html, height=300, width=600)

def parse_keyset(keyset_str):
    choice_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    answer_key = {}
    for line in keyset_str.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(" - ")
        if len(parts) != 2:
            continue
        q_str, ans_str = parts
        try:
            q = int(q_str.strip()) - 1
            ans = ans_str.strip().lower()
            if "," in ans:
                answer_key[q] = -1
            else:
                answer_key[q] = choice_map.get(ans, -1)
        except Exception:
            continue
    return answer_key

def resize_image(image, target_width=1500):
    h, w = image.shape[:2]
    scale = target_width / w
    new_height = int(h * scale)
    resized_image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def process_pdf(file_bytes):
    pages = convert_from_bytes(file_bytes)
    page = pages[0]
    open_cv_image = np.array(page)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image

def process_omr(image, answer_key):
    image = resize_image(image, target_width=1500)
    warp_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        warp_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_cnts = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if 7 < w < 80 and 7 < h < 80 and 0.4 < ar < 1.8:
            bubble_cnts.append((x, y, w, h))

    bubble_cnts = sorted(bubble_cnts, key=lambda b: (b[1], b[0]))
    total_questions = max(answer_key.keys()) + 1 if answer_key else 0
    choices_per_question = 4
    marked_answers = []

    for q in range(total_questions):
        bubbles = bubble_cnts[q*choices_per_question:(q+1)*choices_per_question]
        if len(bubbles) < choices_per_question:
            marked_answers.append(-1)
            continue
        filled = [np.sum(thresh[y:y+h, x:x+w]) for (x,y,w,h) in bubbles]
        answer = np.argmax(filled)
        marked_answers.append(answer)

    score = 0
    for i in range(total_questions):
        correct = answer_key.get(i, -1)
        detected = marked_answers[i] if i < len(marked_answers) else -1
        if correct == -1:
            continue

        if detected == correct:
            score += 1

    st.markdown(f'<div style="background-color:#d6f0d6; padding: 15px; border-radius: 8px; color:#104910; font-weight:bold;">Parsed answer key for {len(answer_key)} questions</div>', unsafe_allow_html=True)
    show_confetti()
    st.markdown(f'<div style="background-color:#cce5ff; padding: 15px; border-radius: 8px; color:#17375e; font-weight:bold;">OMR sheet scored: {score} out of {len(answer_key)}</div>', unsafe_allow_html=True)
    show_confetti()
    st.write(f"Detected answers: {marked_answers}")

    return score, marked_answers

answer_key = None
if keyset_input.strip():
    answer_key = parse_keyset(keyset_input)
else:
    answer_key = None

if uploaded_file and answer_key:
    if uploaded_file.type == "application/pdf":
        file_bytes = uploaded_file.read()
        image = process_pdf(file_bytes)
    else:
        file_bytes = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    process_omr(image, answer_key)
else:
    if not keyset_input.strip():
        st.info("Paste the answer key in the sidebar.")
    if not uploaded_file:
        st.info("Upload the OMR sheet image or PDF in the sidebar.")


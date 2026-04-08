import requests
from collections import Counter

from streamlit import image

OPENROUTER_API_KEY = "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"


# ====================================
# COUNT
# ====================================
def count_from_boxes(boxes):
    labels = [b["class"] for b in boxes]
    return dict(Counter(labels))


# ====================================
# CENTER
# ====================================
def get_centers(boxes):
    results = []

    for b in boxes:
        x, y, w, h = b["box"]
        cx = x + w // 2
        cy = y + h // 2

        results.append({
            "class": b["class"],
            "center": (cx, cy),
            "size": (w, h)
        })

    return results

def get_direction(cx, cy, w, h):
    # sederhana: bagi jadi 4 arah
    if cy < h * 0.33:
        return "utara Indonesia"
    elif cy > h * 0.66:
        return "selatan Indonesia"
    elif cx < w * 0.5:
        return "barat Indonesia"
    else:
        return "timur Indonesia"
    
# ====================================
# BUILD PROMPT (UPDATED 🔥)
# ====================================
def build_prompt(boxes, image_shape, selected_date):
    from collections import Counter

    h, w = image_shape[:2]
    counts = Counter([b["class"] for b in boxes])

    def get_direction(cx, cy):
        if cy < h * 0.33:
            return "utara Indonesia"
        elif cy > h * 0.66:
            return "selatan Indonesia"
        elif cx < w * 0.5:
            return "barat Indonesia"
        else:
            return "timur Indonesia"

    directions = {"core": [], "impact": [], "dcc": []}

    for b in boxes:
        x, y, bw, bh = b["box"]
        cx = x + bw // 2
        cy = y + bh // 2
        directions[b["class"]].append(get_direction(cx, cy))

    if selected_date:
        date_str = selected_date.strftime("%Y-%m-%d %H:%M WIB")
        date_url = selected_date.strftime("%Y-%m-%d %H:%M:%S")
    else:
        date_str = "tidak diketahui"
        date_url = "unknown"

    zoom_url = f"https://zoom.earth/maps/satellite-hd/#view=-0.5,117.7,4.69z/date={date_url}"

    prompt = f"""
    Tuliskan analisis cuaca dalam bahasa Indonesia yang singkat, jelas, dan tidak bertele-tele (maksimal 120 kata).

    WAJIB:
    - Jangan menyebut nama provinsi atau daerah spesifik
    - Gunakan hanya arah wilayah Indonesia (utara, selatan, timur, barat)
    - Jangan menggunakan istilah aneh atau bahasa asing
    - Jangan membuat informasi tambahan di luar data

    Awali dengan:
    "Gambar ini merupakan kondisi cuaca Indonesia berdasarkan citra satelit Himawari BMKG parameter Enhanced IR pada {date_str}."

    Data:
    - Red Core: {counts.get('core', 0)} → {directions['core']}
    - Impacted Area: {counts.get('impact', 0)} → {directions['impact']}
    - DCC: {counts.get('dcc', 0)} → {directions['dcc']}

    Aturan:
    - Jika ada Red Core → potensi bibit siklon tropis
    - Jika hanya DCC → cuaca buruk, belum siklon
    - Jika tidak ada semua → cuaca cerah

    Tulis:
    1. Kondisi cuaca (cerah atau tidak, jika tidak cerah sebutkan jumlah setiap kelas dan arahnya)
    2. Potensi siklon (jika ada red core berarti ini merupakan bibit siklon tropis ataupun lintasan siklon tropis yang terdeteksi, jika tidak ada red core berarti tidak ada potensi siklon, kalau ada DCC berarti cuaca buruk tapi belum siklon)
    3. Risiko

    Terakhir:
    Tambahkan kalimat:
    "Sebagai pembanding, hasil ini dapat dibandingkan dengan citra pada laman Zoom Earth: yang dapat diklik pada tombol "Open Zoom Earth" di samping. Jika hasil interpretasi  berbeda, gunakan data resmi dari zoom earth sebagai acuan."

    Jangan menambahkan penjelasan lain.
    JANGAN:
    - Menanyakan pertanyaan ke user
    - Menggunakan kalimat interaktif seperti "Would you like..."
    - Menambahkan komentar di luar analisis
    - Jangan berupa percakapan karena output ini berupa laporan yang akan langsung ditampilkan ke user tanpa interaksi lebih lanjut

    GUNAKAN:
    - Gaya laporan formal
    - Langsung ke hasil analisis
    """
    return prompt


def translate_to_english(text):
    prompt = f"Translate this into English and don't add any extra text or questions:\n{text}"
    return ask_llm(prompt)
# ====================================
# CALL LLM
# ====================================
def ask_llm(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    models = [
        "google/gemma-3-4b-it:free",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3.6-plus:free",
        "openrouter/auto"
    ]

    for model in models:
        try:
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500
            }

            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=8  # ⏱️ biar cepat failover
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                if content and content.strip():
                    print(f"{model} SUCCESS ✅")
                    return content

            print(f"{model} FAILED ❌ ({response.status_code})")

        except Exception as e:
            print(f"{model} ERROR ❌ ({str(e)})")

    return "Semua model gagal merespon."


# ====================================
# MAIN
# ====================================
# def interpret_boxes(boxes, image_shape, selected_date):
#     prompt = build_prompt(boxes, image_shape, selected_date)
#     result = ask_llm(prompt)
#     return result

def interpret_boxes(boxes, image_shape, selected_date):
    prompt = build_prompt(boxes, image_shape, selected_date)
    indo = ask_llm(prompt)
    eng = translate_to_english(indo)

    return indo, eng



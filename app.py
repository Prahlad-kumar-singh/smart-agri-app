import os
import re
import hmac
import uuid
import pickle
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
from train_model import PureRandomForest, DecisionTree

app = Flask(__name__)

# 1. Session & Security Config
app.secret_key = os.environ.get("SECRET_KEY", "smart-agri-dev-secret-change-in-prod")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # Max 5 MB upload
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

APP_ACCESS_PIN = os.environ.get("APP_ACCESS_PIN", "").strip()
ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}

# Folders setup
UPLOAD_FOLDER = os.path.join("static", "uploads")
DB_FOLDER = "database"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
DB_PATH = os.path.join(DB_FOLDER, "khet_records.db")


def is_allowed_image(filename):
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_IMAGE_EXTS


def safe_unique_filename(filename):
    cleaned = secure_filename(filename)
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", cleaned)
    return f"{uuid.uuid4().hex[:10]}_{cleaned}"


# Secure PIN comparison
def is_valid_pin(entered_pin):
    if not APP_ACCESS_PIN:
        return False
    return hmac.compare_digest(entered_pin.encode("utf-8"), APP_ACCESS_PIN.encode("utf-8"))


# Optional PIN Protection (Jab APP_ACCESS_PIN set ho tabhi active hoga)
@app.before_request
def check_access_pin():
    if not APP_ACCESS_PIN:
        return None
    open_paths = {"/unlock", "/manifest.json", "/favicon.ico"}
    if request.path in open_paths or request.path.startswith("/static/"):
        return None
    if not session.get("agri_unlocked"):
        return redirect(url_for("unlock"))


@app.route("/unlock", methods=["GET", "POST"])
def unlock():
    error = None
    if request.method == "POST":
        entered_pin = request.form.get("pin", "").strip()
        if is_valid_pin(entered_pin):
            session["agri_unlocked"] = True
            return redirect(url_for("index"))
        error = "Galat PIN. Kripya sahi PIN daalein."

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Unlock - Smart Agri AI</title>
    </head>
    <body style="font-family:sans-serif;background:#e8f5e9;padding:20px;">
      <form method="POST" style="max-width:340px;margin:60px auto;background:#fff;padding:20px;border-radius:12px;box-shadow:0 6px 18px rgba(0,0,0,0.1);">
        <h3 style="margin-top:0;color:#1b5e20;">🔒 Smart Agri AI Access</h3>
        <p style="font-size:12.5px;color:#444;">Bina same Wi-Fi ya public link ke dauran suraksha ke liye Access PIN daalein.</p>
        <input type="password" name="pin" required placeholder="Enter Access PIN"
               style="width:100%;padding:9px;margin:8px 0 12px;border:1.5px solid #cfd8dc;border-radius:8px;box-sizing:border-box;">
        <button type="submit" style="width:100%;padding:10px;background:#2e7d32;color:#fff;border:none;border-radius:8px;font-weight:bold;cursor:pointer;">
          Unlock App
        </button>
        {f"<p style='color:#c62828;font-size:12px;margin-top:10px;'>{error}</p>" if error else ""}
      </form>
    </body>
    </html>
    """


# App ko dobara lock karne ke liye route
@app.route("/lock")
def lock_app():
    session.pop("agri_unlocked", None)
    return redirect(url_for("unlock"))


@app.errorhandler(413)
def file_too_large(e):
    return render_template(
        "plant_doctor.html",
        diagnosis={
            "status": "File Size Bahut Bada Hai (Max 5 MB Allowed)",
            "severity": "Upload Rejected",
            "remedy": "Kripya 5 MB se kam size ki JPG, PNG, ya WEBP photo upload karein.",
            "stats": "Upload Limit Exceeded"
        },
        uploaded_img_url=None,
        plant_part="",
        crop_name=""
    ), 413


# SQLite Database Initialize
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS plot_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            village TEXT,
            flood_status TEXT,
            soil_type TEXT,
            coordinates TEXT,
            npk_summary TEXT,
            recommended_crop TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_db()

# Trained ML model load karein
model_path = os.path.join("models", "crop_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)


def get_crop_tip(crop_name):
    crop_key = (crop_name or "").strip().lower()

    crop_treatments = {
        "wheat": (
            "Beej upchar ke liye 2–2.5 g Carbendazim ya Thiram prati kg beej "
            "(ya jaivik upchar ke liye 4–5 g Trichoderma viride prati kg beej) se upcharit karein. "
            "Buwai se pehle mitti ki nami check karein."
        ),
        "mustard": (
            "Beej upchar ke liye 2–2.5 g Metalaxyl (Apron 35 SD) ya Thiram prati kg beej upyog karein "
            "taaki white rust aur downy mildew se bachav ho sake. Halka-nam khet aur achha drainage rakhein."
        ),
        "sugarcane": (
            "Ganna setts (tukdon) ko buwai se pehle 0.1% Carbendazim ghol mein 10–15 minute dubokar "
            "beej/sett upchar karein taaki red rot aur sett rot se bachav ho. Sahi nami aur gahrai par ropai karein."
        ),
        "rice": (
            "Beej upchar ke liye Carbendazim 2 g/kg beej (ya jaivik Pseudomonas fluorescens / Trichoderma) "
            "se treat karein. Nursery aur khet mein paani ka star niyantrit rakhein."
        ),
        "maize": (
            "Beej upchar ke liye Thiram ya Mancozeb 2.5 g/kg beej (ya Azospirillum/PSB bio-culture) ka upyog karein. "
            "Jal-nikasi wali mitti aur uchit doori par buwai rakhein."
        ),
        "chickpea": (
            "Beej upchar ke liye Carbendazim + Thiram (1:1 ratio mein ~2.5 g/kg beej) aur uske baad "
            "Rhizobium + PSB culture se upcharit karein. Jal-bharav (waterlogging) se bachayein."
        ),
        "lentil": (
            "Beej upchar ke liye Thiram ya Carbendazim 2 g/kg beej aur Rhizobium culture ka istemal karein. "
            "Mitti mein zyada nami ya paani rukne na dein."
        ),
        "mungbean": (
            "Beej upchar ke liye Carbendazim 2 g/kg beej aur Rhizobium culture se upcharit karein. "
            "Buwai ke samay halka-nam aur bhurbhura khet rakhein."
        ),
        "blackgram": (
            "Beej upchar ke liye Thiram/Carbendazim 2–2.5 g/kg beej aur Rhizobium inoculation karein. "
            "Khet mein paani na rukne dein."
        ),
        "cotton": (
            "Beej upchar ke liye 2–3 g/kg Carboxin ya Thiram (ya Trichoderma viride 4–5 g/kg) ka upyog karein. "
            "Buwai se pehle nami aur tapman ka dhyan rakhein."
        ),
        "jute": (
            "Beej upchar ke liye Carbendazim ya Thiram 2.5 g/kg beej ka upyog karein. "
            "Achi nami aur bhurbhuri mitti mein line se buwai karein."
        ),
    }

    default_treatment = (
        "Beej upchar ke liye recommended fungicide (jaise Carbendazim/Thiram 2–2.5 g/kg beej) "
        "ya jaivik Trichoderma/bio-fertilizer ka upyog karein aur buwai se pehle mitti ki nami jaanchein."
    )

    return crop_treatments.get(crop_key, default_treatment)


def get_local_advisory(flood_status, soil_type, crop_name=""):
    if "Ganga Kinare" in flood_status:
        base_tip = "Ganga diara ki gaad aur nami ke kaaran Rabi mein Makka, Gehu, Sarson ya Parwal/Karela jaisi sabziyan bahut achhi paidawar deti hain."
    elif "Koshi Kinare" in flood_status:
        base_tip = "Koshi belt ki baluahi-gaad mitti mein paani utarne ke baad Makka, Gehu (short-medium duration), ya Mirch/Sabzi ki kheti upyukt rehti hai."
    elif "Non-Flooded Irrigated" in flood_status:
        base_tip = "Sinchit aur bina baadh wali zameen par aap poora saal fasal chakra (Dhan - Gehu - Moong/Sabzi) aasani se chala sakte hain."
    elif "Non-Flooded Upland" in flood_status:
        base_tip = "Oonchi zameen par jal-nikasi (drainage) achhi rehti hai—Dalhan (Arhar/Chana), Tilhan (Sarson), ya Makka ke liye behtar hai."
    elif "Late Water Recession" in flood_status:
        base_tip = "Der se paani utarne par zero-tillage ya late-sown varieties (jaise der se buwai wala Gehu ya Makka) chunein taaki mitti ki nami ka sahi upyog ho."
    elif "Lowland" in flood_status:
        base_tip = "Nichli ya chaur zameen mein zyada nami rehti hai—jal-sahansheel Dhan, Makhana, ya paani sukhne par Rabi fasal best rahegi."
    else:
        base_tip = f"{soil_type} mitti mein buwai se pehle mitti ki nami zaroor check karein."

    crop_tip = get_crop_tip(crop_name)
    if crop_name:
        return f"{base_tip} [{crop_name}] {crop_tip}"
    return f"{base_tip} {crop_tip}"


def validate_plot_combination(village, flood_status, soil_type):
    v_lower = (village or "").strip().lower()
    f_lower = (flood_status or "").strip().lower()
    s_lower = (soil_type or "").strip().lower()

    is_non_flooded = "non-flooded" in f_lower
    is_gaad = "gaad" in s_lower or "new alluvial silt" in s_lower
    is_koshi = "koshi" in v_lower or "koshi" in f_lower

    if is_non_flooded and is_gaad:
        return (
            "Galat Combination: 'Gaad (New Alluvial Silt)' nayi baadh/nadi ki mitti hoti hai, "
            "isliye ise 'Non-Flooded' zameen ke saath nahi chuna ja sakta. Kripya Flood Status ya Soil Type badlein."
        )

    if is_non_flooded and is_koshi:
        return (
            "Galat Combination: Koshi belt / Srikhandi kshetra mein 'Non-Flooded' zameen hona "
            "zameeni haqeeqat se mel nahi khata. Kripya sahi Flood Status chunein."
        )

    return None


def analyze_plant_health(plant_part, symptom_note, image_path):
    yellow_ratio = brown_ratio = green_ratio = 0.0
    try:
        from PIL import Image
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB").resize((100, 100))
            pixels = list(img.getdata())
            total = len(pixels)
            y_cnt = b_cnt = g_cnt = 0
            for r, g, b in pixels:
                if g > r + 15 and g > b + 15:
                    g_cnt += 1
                elif r > 140 and g > 130 and b < 100:
                    y_cnt += 1
                elif r > b + 20 and g < 130:
                    b_cnt += 1
            green_ratio = round((g_cnt / total) * 100, 1)
            yellow_ratio = round((y_cnt / total) * 100, 1)
            brown_ratio = round((b_cnt / total) * 100, 1)
    except Exception:
        pass

    note = (symptom_note or "").lower()
    if plant_part == "Leaves (Patte)":
        if "peela" in note or "yellow" in note or yellow_ratio > 25:
            return {
                "status": "Chlorosis / Nitrogen ya Sulfur ki Kami (Yellowing)",
                "severity": "Moderate (Madhyam)",
                "remedy": "1) 2% Urea ya NPK (19:19:19) ka foliar spray karein. 2) Naye patte peele hon toh Zinc Sulphate (0.5%) + bujha chuna chhidkein. 3) Khet mein zyada paani jama na hone dein.",
                "stats": f"Visual Scan: Green {green_ratio}% | Yellow {yellow_ratio}% | Spots {brown_ratio}%"
            }
        elif "dhabba" in note or "spot" in note or "rust" in note or brown_ratio > 25:
            return {
                "status": "Fungal Leaf Spot / Blight ya Rust Infection",
                "severity": "High (Turant Dhyan Dein)",
                "remedy": "1) Sankramit patton ko tod kar khet se bahar nasht karein. 2) Mancozeb 75% WP (2g/liter paani) ya Carbendazim ka chhidkav karein. 3) Shaam ke samay sinchai na karein.",
                "stats": f"Visual Scan: Green {green_ratio}% | Yellow {yellow_ratio}% | Spots {brown_ratio}%"
            }
        return {
            "status": "General Leaf Stress / Pest Sucking Check",
            "severity": "Low to Moderate",
            "remedy": "1) Patton ke neeche Mahu/Thrips check karein. 2) Neem Oil (1500 ppm @ 3 ml/liter) ka spray karein. 3) Balanced NPK aur nami banaye rakhein.",
            "stats": f"Visual Scan: Green {green_ratio}% | Yellow {yellow_ratio}% | Spots {brown_ratio}%"
        }
    elif plant_part == "Stem (Tana)":
        return {
            "status": "Stem Borer / Tana Galan (Stem Rot) Risk",
            "severity": "High",
            "remedy": "1) Tane mein chhed ya dead heart dikhe toh Cartap Hydrochloride ya Chlorantraniliprole dalein. 2) Tana galan ke liye Copper Oxychloride (3g/liter) se jad ke paas drenching karein.",
            "stats": f"Visual Scan: Green {green_ratio}% | Brown/Dry {brown_ratio}%"
        }
    elif plant_part == "Roots (Jad)":
        return {
            "status": "Root Rot (Jad Galan) / Waterlogging Stress",
            "severity": "High",
            "remedy": "1) Khet se extra paani turant bahar nikalein (drainage sudharein). 2) Trichoderma viride ko gobar ki khaad mein mila kar jadon ke paas dalein. 3) Carbendazim 1g/liter se drenching karein.",
            "stats": f"Visual Scan: Brown/Soil Tone {brown_ratio}%"
        }
    return {
        "status": "Pod Borer (Fali Chhedak) / Boron-Calcium Kami",
        "severity": "Moderate to High",
        "remedy": "1) Faliyon mein chhed dikhne par Emamectin Benzoate (0.5g/liter) ya Neem oil spray karein. 2) Achhe bharav ke liye Boron (20% @ 1g/liter) ka chhidkav karein.",
        "stats": f"Visual Scan: Green {green_ratio}% | Spot/Yellow {round(yellow_ratio + brown_ratio, 1)}%"
    }


CROP_CALC_PROFILES = {
    "Wheat (Gehu)": {
        "urea": 110.0,
        "dap": 55.0,
        "mop": 35.0,
        "seed": 45.0,
        "default_season": "Rabi",
        "seasons": {
            "Rabi": {
                "irrigation_count": "4 se 6 baar (mitti aur mitti ki nami ke hisaab se)",
                "irrigation_interval": "Har 20–25 din ke antral par",
                "basal_note": "Rabi buwai (Nov–Dec) ke samay poora DAP + MOP aur lagbhag 1/3 Urea mitti mein milayein.",
                "schedule": [
                    {
                        "stage": "1st Paani (CRI / Taj Jad Avastha — Buwai ke 20–25 din baad)",
                        "interval": "20–25 din par",
                        "fertilizer": "Lagbhag 1/3 Urea top-dressing (paani ke baad halka watt aane par chhidkav karein)."
                    },
                    {
                        "stage": "2nd Paani (Kalla Futav / Tillering — 40–45 din baad)",
                        "interval": "~20 din ke antral par",
                        "fertilizer": "Bacha hua lagbhag 1/3 Urea (zaroorat ho toh Zinc Sulphate saath mila sakte hain)."
                    },
                    {
                        "stage": "3rd Paani (Ganth Banne / Jointing — 60–65 din baad)",
                        "interval": "~20 din ke antral par",
                        "fertilizer": "Is sinchai mein alag se thos khaad zaroori nahi; mitti mein nami barkarar rakhein."
                    },
                    {
                        "stage": "4th–5th Paani (Balii Nikalne / Booting & Daana Bharav — 80–105 din baad)",
                        "interval": "18–22 din ke antral par",
                        "fertilizer": "Daane achhe bharne ke liye zaroorat padne par 1–2% NPK (19:19:19) ya MOP ka halka spray karein."
                    }
                ],
                "land_note": "Rabi mein CRI (21 din) aur balii bharav ke samay paani ki kami bilkul na hone dein."
            },
            "Kharif": {
                "irrigation_count": "Aam taur par 0–2 baar (sirf barish ka lamba gap hone par)",
                "irrigation_interval": "Barish na hone par 15–20 din ke antral par nami jaanch kar",
                "basal_note": "Kharif mausam mein Gehu ki buwai aam taur par nahi ki jaati; zyada barish aur tapman se fasal ko nuksan ho sakta hai.",
                "schedule": [
                    {
                        "stage": "Barish / Nami Jaanch (Buwai ke 20–25 din baad)",
                        "interval": "15–25 din par (mausam ke hisaab se)",
                        "fertilizer": "Khet mein paani jama na hone dein; zaroorat ho toh 1/3 Urea tabhi dein jab nikaasi (drainage) theek ho."
                    },
                    {
                        "stage": "Madhyam Avastha (40–45 din baad)",
                        "interval": "15–20 din ke antral par",
                        "fertilizer": "Bacha hua 1/3 Urea (agar fasal khadi aur swasth ho toh)."
                    }
                ],
                "land_note": "Gehu mukhya roop se Rabi ki fasal hai—Kharif mein khet ki jal-nikasi sabse zaroori hai."
            }
        }
    },
    "Maize (Makka)": {
        "urea": 130.0,
        "dap": 65.0,
        "mop": 40.0,
        "seed": 8.0,
        "default_season": "Kharif",
        "seasons": {
            "Kharif": {
                "irrigation_count": "6 baar (local kheti aur mitti ki nami/barish ke hisaab se)",
                "irrigation_interval": "Lagbhag har 12–15 din ke antral par (nami jaanch kar)",
                "basal_note": "Kharif buwai (June–July) ke samay poora DAP + MOP aur lagbhag 1/4 Urea line mein/mitti mein dein.",
                "schedule": [
                    {
                        "stage": "1st Paani (Buwai ke ~12–15 din baad)",
                        "interval": "12–15 din par",
                        "fertilizer": "Halka paani/nami; abhi thos khaad nahi, ankur/jaddon ko set hone dein."
                    },
                    {
                        "stage": "2nd Paani (Buwai ke ~25–28 din baad)",
                        "interval": "~12–14 din ke antral par",
                        "fertilizer": "1st Urea top-dressing (lagbhag 1/3 Urea paani ke baad halka watt aane par)."
                    },
                    {
                        "stage": "3rd Paani (Buwai ke ~38–42 din baad — Ghutna / Earthing Stage)",
                        "interval": "~12–15 din ke antral par",
                        "fertilizer": "2nd Urea top-dressing (bacha hua lagbhag 1/3 Urea + mitti chadhana/earthing up karein)."
                    },
                    {
                        "stage": "4th Paani (Buwai ke ~52–56 din baad)",
                        "interval": "~14 din ke antral par",
                        "fertilizer": "Bacha hua thoda Urea (agar pehle poora na diya ho) ya foliar micronutrient/NPK spray."
                    },
                    {
                        "stage": "5th Paani (Buwai ke ~65–70 din baad — Tasseling / Bhutta Nikalne par)",
                        "interval": "~13–15 din ke antral par",
                        "fertilizer": "Khaad nahi; bhutta nikalte waqt khet mein nami banaye rakhein (khada paani na rukne dein)."
                    },
                    {
                        "stage": "6th Paani (Buwai ke ~80–85 din baad — Daana Bharav Avastha)",
                        "interval": "~15 din ke antral par",
                        "fertilizer": "Khaad nahi; daane achhe bharne ke liye aakhiri halka paani (jal-bharav na hone dein)."
                    }
                ],
                "land_note": "6 sinchai ke dauran har baar paani nikalne/drainage ka dhyan rakhein taaki khet mein paani jama na ho."
            },
            "Rabi": {
                "irrigation_count": "5 se 6 baar (sardi mein niyamit sinchai)",
                "irrigation_interval": "Har 16–22 din ke antral par",
                "basal_note": "Rabi buwai (Oct–Nov) ke samay poora DAP + MOP aur lagbhag 1/4 Urea mitti mein milayein.",
                "schedule": [
                    {
                        "stage": "1st Paani (Buwai ke ~18–20 din baad)",
                        "interval": "18–20 din par",
                        "fertilizer": "Halka paani; zaroorat ho toh shuruati badhvaar ke liye halka Urea."
                    },
                    {
                        "stage": "2nd Paani (Buwai ke ~35–40 din baad)",
                        "interval": "~18 din ke antral par",
                        "fertilizer": "1st Urea top-dressing (lagbhag 1/3 Urea watt aane par)."
                    },
                    {
                        "stage": "3rd Paani (Buwai ke ~52–58 din baad — Earthing Stage)",
                        "interval": "~18 din ke antral par",
                        "fertilizer": "2nd Urea top-dressing (bacha hua lagbhag 1/3 Urea + mitti chadhana karein)."
                    },
                    {
                        "stage": "4th–6th Paani (Tasseling, Silking & Daana Bharav — ~70 se 105 din tak)",
                        "interval": "Har 16–22 din ke antral par",
                        "fertilizer": "Khaad nahi; bhutta aur daana bharne ke dauran nami ki kami na hone dein."
                    }
                ],
                "land_note": "Rabi mein paani ka interval mausam aur thand ke hisaab se adjust karein."
            }
        }
    },
    "Rice (Dhan)": {
        "urea": 100.0,
        "dap": 50.0,
        "mop": 35.0,
        "seed": 15.0,
        "default_season": "Kharif",
        "seasons": {
            "Kharif": {
                "irrigation_count": "Barish ke alawa 4–7 baar (ropai ke baad khet mein nami/halka paani)",
                "irrigation_interval": "Paani sokhne ke 2–3 din baad (lagbhag har 7–12 din par, barish na hone par)",
                "basal_note": "Kharif ropai (June–July) ke samay DAP + MOP aur lagbhag 1/3 Urea khet mein milayein.",
                "schedule": [
                    {
                        "stage": "1st Paani / Nami (Ropai ke 18–21 din baad)",
                        "interval": "18–21 din par",
                        "fertilizer": "Lagbhag 1/3 Urea + Zinc Sulphate (agar ropai ke waqt na diya ho toh)."
                    },
                    {
                        "stage": "2nd Paani / Nami (Kalla Nikalte Samay / Tillering — 35–42 din baad)",
                        "interval": "15–20 din ke antral par",
                        "fertilizer": "Bacha hua lagbhag 1/3 Urea (khet se khada paani nikaal kar chhidkav karein, 24 ghante baad paani bharein)."
                    },
                    {
                        "stage": "3rd Paani onwards (Panicle Initiation & Daana Bharav — 55–85 din baad)",
                        "interval": "7–12 din ke antral par nami",
                        "fertilizer": "Khaad nahi; daana bharne tak khet mein lagbhag 1.5–2 inch nami/paani banayein rakhein."
                    }
                ],
                "land_note": "Dhan mukhya roop se Kharif fasal hai—beech-beech mein halka paani aur nikaasi kalle achhe banati hai."
            },
            "Rabi": {
                "irrigation_count": "8 se 12 baar (garmi/sukhe mausam ke kaaran zyada sinchai)",
                "irrigation_interval": "Har 5–9 din ke antral par (mitti ki nami ke hisaab se)",
                "basal_note": "Ropai ke samay DAP + MOP aur lagbhag 1/3 Urea mitti mein milayein.",
                "schedule": [
                    {
                        "stage": "1st Paani (Ropai ke 15–20 din baad)",
                        "interval": "15–20 din par",
                        "fertilizer": "Lagbhag 1/3 Urea + Zinc Sulphate."
                    },
                    {
                        "stage": "2nd Paani (Tillering Stage — 30–38 din baad)",
                        "interval": "12–18 din ke antral par",
                        "fertilizer": "Bacha hua lagbhag 1/3 Urea (paani nikaal kar chhidkav, phir 24 ghante baad halka paani)."
                    },
                    {
                        "stage": "3rd Paani onwards (Flowering & Grain Filling)",
                        "interval": "5–9 din ke antral par",
                        "fertilizer": "Khaad nahi; paani ki kami na hone dein (paani ki lagat zyada aati hai)."
                    }
                ],
                "land_note": "Rabi/Garma Dhan mein paani aur bijli/diesel ki lagat zyada aati hai—paani ki uplabdhata pakki ho tabhi karein."
            }
        }
    },
    "Mustard (Sarson)": {
        "urea": 65.0,
        "dap": 45.0,
        "mop": 25.0,
        "seed": 2.5,
        "default_season": "Rabi",
        "seasons": {
            "Rabi": {
                "irrigation_count": "2 se 3 baar (mitti ki nami ke hisaab se)",
                "irrigation_interval": "Har 25–35 din ke antral par",
                "basal_note": "Rabi buwai (Oct–Nov) ke samay poora DAP + MOP (aur sulfur/SSP ho toh) + aadha Urea mitti mein dein.",
                "schedule": [
                    {
                        "stage": "1st Paani (Shakha Nikalte Samay / Branching — Buwai ke 25–30 din baad)",
                        "interval": "25–30 din par",
                        "fertilizer": "Bacha hua 1/2 Urea top-dressing (paani ke baad halka watt aane par)."
                    },
                    {
                        "stage": "2nd Paani (Phalli Banne ki Avastha / Pod Formation — 55–65 din baad)",
                        "interval": "25–35 din ke antral par",
                        "fertilizer": "Koi khaad nahi; halka paani dein aur khet mein paani jama na hone dein."
                    }
                ],
                "land_note": "Sarson mukhya roop se Rabi ki fasal hai—phool/phalli ke waqt halka paani aur jal-nikasi zaroori hai."
            },
            "Kharif": {
                "irrigation_count": "1 se 2 baar (sirf barish ka lamba gap hone par)",
                "irrigation_interval": "20–30 din ke antral par (nami dekh kar)",
                "basal_note": "Buwai ke samay DAP + MOP + aadha Urea dein; khet mein paani rukne na dein.",
                "schedule": [
                    {
                        "stage": "1st Nami / Paani (Buwai ke 25–30 din baad, sukha ho toh)",
                        "interval": "25–30 din par",
                        "fertilizer": "Bacha hua 1/2 Urea (khet mein paani khada na ho tabhi dein)."
                    },
                    {
                        "stage": "2nd Nami / Paani (55–65 din baad)",
                        "interval": "25–30 din ke antral par",
                        "fertilizer": "Khaad nahi; teeno-dono aur fungal rog se bachav ka dhyan rakhein."
                    }
                ],
                "land_note": "Sarson aam taur par Rabi mein hoti hai—Kharif mein zyada nami aur barish se rog badhne ka khatra rehta hai."
            }
        }
    },
    "Vegetables (Sabzi)": {
        "urea": 90.0,
        "dap": 60.0,
        "mop": 45.0,
        "seed": 1.5,
        "default_season": "Kharif",
        "seasons": {
            "Kharif": {
                "irrigation_count": "3 se 6 baar (barish aur jal-nikasi ke hisaab se)",
                "irrigation_interval": "Barish na hone par har 8–14 din ke antral par",
                "basal_note": "Kharif ropai/buwai (jaise Karela, Parwal, Baingan) ke samay gobar ki khaad + DAP + MOP + 1/3 Urea dein.",
                "schedule": [
                    {
                        "stage": "1st Paani / Nami (Ropai / Ankur Ke 12–18 din baad)",
                        "interval": "12–18 din par (halka paani)",
                        "fertilizer": "Lagbhag 1/3 Urea paudhon ki jad se thodi doori par dein."
                    },
                    {
                        "stage": "2nd Paani (Badhvaar / Vegetative Growth — 25–35 din baad)",
                        "interval": "10–15 din ke antral par",
                        "fertilizer": "Bacha hua lagbhag 1/3 Urea (baaki hissa baad mein)."
                    },
                    {
                        "stage": "3rd Paani onwards (Phool aur Phal Banne ke Dauraan)",
                        "interval": "8–14 din ke antral par (barish na ho tab)",
                        "fertilizer": "Zaroorat padne par NPK (19:19:19) ya Boron/Calcium ka halka spray karein; medh (ridge) par lagayein taaki paani na bhare."
                    }
                ],
                "land_note": "Kharif sabziyon mein medh (ridge) par lagana aur paani ki nikaasi sabse zaroori hai."
            },
            "Rabi": {
                "irrigation_count": "6 se 9 baar (sardi mein niyamit halka paani)",
                "irrigation_interval": "Har 10–16 din ke antral par",
                "basal_note": "Rabi sabziyon (jaise Tamatar, Gobhi, Matar, Aloo, Palak) ke samay gobar khaad + DAP + MOP + 1/3 Urea dein.",
                "schedule": [
                    {
                        "stage": "1st Paani (Ropai / Ankur Ke 12–18 din baad)",
                        "interval": "12–18 din par (halka paani)",
                        "fertilizer": "Lagbhag 1/3 Urea paudhon ke paas dein."
                    },
                    {
                        "stage": "2nd Paani (Badhvaar — 25–35 din baad)",
                        "interval": "12–16 din ke antral par",
                        "fertilizer": "Bacha hua lagbhag 1/3 Urea."
                    },
                    {
                        "stage": "3rd Paani onwards (Phool, Kand/Phal Banne ke Dauraan)",
                        "interval": "10–16 din ke antral par",
                        "fertilizer": "Zaroorat padne par 1–2% NPK ya Boron/Potash ka halka upyog karein."
                    }
                ],
                "land_note": "Rabi sabziyon mein subah ya dopahar ke samay halka paani dein aur raat mein zyada paani bharne se bachein."
            }
        }
    }
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    village = flood_status = soil_type = coordinates = advisory = npk_summary = ""
    saved_msg = False

    if request.method == "POST":
        try:
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])
        except (KeyError, ValueError):
            return render_template(
                "index.html",
                prediction=None,
                advisory="Kripya N, P, K, pH, Temperature, Humidity, aur Rainfall ki sahi sankhya (number) daalein."
            )

        village = request.form.get("village", "Not Specified")
        if village == "Other":
            custom_v = request.form.get("custom_village", "").strip()
            village = custom_v if custom_v else "Custom Plot"

        flood_status = request.form.get("flood_status", "Non-Flooded Irrigated (Sinchit)")
        soil_type = request.form.get("soil_type", "Gaad (New Alluvial Silt)")
        coordinates = request.form.get("coordinates", "")
        npk_summary = f"N:{int(N)}, P:{int(P)}, K:{int(K)}, pH:{ph}"

        validation_error = validate_plot_combination(village, flood_status, soil_type)
        if validation_error:
            return render_template(
                "index.html",
                prediction=None,
                village=village,
                flood_status=flood_status,
                soil_type=soil_type,
                coordinates=coordinates,
                advisory=validation_error,
                saved_msg=False
            )

        features = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = model.predict(features)[0].capitalize()
        advisory = get_local_advisory(flood_status, soil_type, prediction)

        if request.form.get("save_record") == "yes":
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''
                INSERT INTO plot_records (created_at, village, flood_status, soil_type, coordinates, npk_summary, recommended_crop)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now().strftime("%d %b %Y, %I:%M %p"), village, flood_status, soil_type, coordinates, npk_summary, prediction))
            conn.commit()
            conn.close()
            saved_msg = True

    return render_template(
        "index.html", prediction=prediction, village=village,
        flood_status=flood_status, soil_type=soil_type,
        coordinates=coordinates, advisory=advisory, saved_msg=saved_msg
    )


@app.route("/calculator", methods=["GET", "POST"])
def calculator():
    calc_result = None
    if request.method == "POST":
        crop = request.form.get("crop", "Wheat (Gehu)")
        season_choice = request.form.get("season", "Rabi")
        try:
            area = max(0.0, float(request.form.get("area", 1)))
        except ValueError:
            area = 1.0
        unit = request.form.get("unit", "Bigha")
        acres = area * 0.62 if unit == "Bigha" else (area * 0.031 if unit == "Katha" else area)

        profile = CROP_CALC_PROFILES.get(crop, CROP_CALC_PROFILES["Wheat (Gehu)"])
        season_key = season_choice if season_choice in profile["seasons"] else profile["default_season"]
        season_data = profile["seasons"][season_key]

        calc_result = {
            "crop": crop,
            "season": season_key,
            "area_label": f"{area} {unit} (~{acres:.2f} Acre)",
            "urea": round(profile["urea"] * acres, 1),
            "dap": round(profile["dap"] * acres, 1),
            "mop": round(profile["mop"] * acres, 1),
            "seed": round(profile["seed"] * acres, 1),
            "irrigation_count": season_data["irrigation_count"],
            "irrigation_interval": season_data["irrigation_interval"],
            "basal_note": season_data["basal_note"],
            "schedule": season_data["schedule"],
            "land_note": season_data["land_note"]
        }
    return render_template("calculator.html", calc_result=calc_result)


@app.route("/records")
def records():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM plot_records ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("records.html", records=rows)


@app.route("/delete-record/<int:rec_id>", methods=["POST"])
def delete_record(rec_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM plot_records WHERE id = ?", (rec_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("records"))


@app.route("/guide")
def guide():
    return render_template("guide.html")


@app.route("/plant-doctor", methods=["GET", "POST"])
def plant_doctor():
    diagnosis = uploaded_img_url = None
    plant_part = crop_name = ""
    if request.method == "POST":
        plant_part = request.form.get("plant_part", "Leaves (Patte)")
        crop_name = request.form.get("crop_name", "General Crop")
        symptom_note = request.form.get("symptom_note", "")
        file = request.files.get("plant_image")
        img_path = ""

        if file and file.filename:
            if not is_allowed_image(file.filename):
                diagnosis = {
                    "status": "Unsupported Image Format",
                    "severity": "Rejected",
                    "remedy": "Kripya sirf JPG, JPEG, PNG, ya WEBP image file upload karein.",
                    "stats": "Allowed: png, jpg, jpeg, webp"
                }
            else:
                filename = safe_unique_filename(file.filename)
                img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(img_path)
                uploaded_img_url = f"/static/uploads/{filename}"
                diagnosis = analyze_plant_health(plant_part, symptom_note, img_path)
        else:
            diagnosis = analyze_plant_health(plant_part, symptom_note, "")

    return render_template(
        "plant_doctor.html", diagnosis=diagnosis, uploaded_img_url=uploaded_img_url,
        plant_part=plant_part, crop_name=crop_name
    )


@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json")


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
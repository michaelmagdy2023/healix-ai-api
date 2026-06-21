from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# 1️⃣ تحديد الموديل بـ None كقيمة افتراضية عشان ميديناش Error
model = None

# تحميل موديل الذكاء الاصطناعي لو موجود واصداراته متوافقة
try:
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Warning: model.pkl file NOT FOUND. Will use medical rules fallback.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. استلام البيانات الأساسية من الـ Node.js
        age = data.get('age', 30)
        sex_enc = data.get('sex_enc', 1) 
        avpu = data.get('avpu', 0)
        masktype = data.get('masktype', 0)
        hr = data.get('hr', 80)
        spo2 = data.get('spo2', 98)
        temp = data.get('temp', 37.0)

        # 2. حساب المؤشرات الخطرة طبياً
        hr_score = 1 if 60 <= hr <= 100 else (4 if hr < 40 or hr > 130 else 2)
        spo2_score = 1 if 95 <= spo2 <= 100 else (4 if spo2 < 88 else 2)
        temp_score = 1 if 36.0 <= temp <= 37.5 else (4 if temp < 35.0 or temp > 39.5 else 2)
        
        total_severity = hr_score + spo2_score + temp_score
        n_abnormal = (1 if hr_score > 1 else 0) + (1 if spo2_score > 1 else 0) + (1 if temp_score > 1 else 0)
        pct_abnormal = n_abnormal / 3.0

        hr_dev = (hr - 80) / 20
        temp_dev = (temp - 36.75) / 0.75
        spo2_dev = (spo2 - 97.5) / 2.5

        status = "Normal"

        # 3. اتخاذ القرار
        if model is not None:
            # لو الموديل موجود، هنستخدمه
            features = np.array([[
                age, sex_enc, avpu, masktype,
                hr, hr_score, temp, temp_score, spo2, spo2_score,
                total_severity, n_abnormal, pct_abnormal,
                hr_dev, temp_dev, spo2_dev,
                hr, 0, 0, temp, 0, 0, spo2, 0, 0, 
                n_abnormal, total_severity
            ]])
            prediction = model.predict(features)[0]
            status = "Danger" if prediction == 1 else "Normal"
        else:
            # 🚀 الخطة البديلة: لو الموديل ضرب إيرور، هنحسبها بالمعادلة الطبية
            if total_severity >= 5:
                status = "Danger"
            else:
                status = "Normal"

        return jsonify({
            "status": "success", 
            "prediction": status,
            "severity_score": total_severity
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    return "Healix AI Model is Running! 🚀"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

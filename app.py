from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('mental_health_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [int(x) for x in request.form.values()]
        input_data = np.array(data).reshape(1, -1)

        prediction = model.predict(input_data)[0]

        # Extract values
        age, sleep, exercise, stress, social, work, screen, substance, history = data

        suggestions = []

        if sleep < 6:
            suggestions.append("😴 Improve sleep (7–8 hrs recommended)")
        if exercise < 2:
            suggestions.append("🏃 Exercise at least 3 times/week")
        if stress > 7:
            suggestions.append("⚠️ Practice meditation & stress management")
        if social < 3:
            suggestions.append("🗣️ Increase social interaction")
        if work > 10:
            suggestions.append("💼 Reduce work overload")
        if screen > 8:
            suggestions.append("📱 Limit screen time")
        if substance == 1:
            suggestions.append("🚭 Avoid substance use")
        if history == 1:
            suggestions.append("🧠 Consult a mental health professional")

        # Risk levels
        if prediction == 0:
            result = "Low Risk"
            color = "green"
        elif prediction == 1:
            result = "Medium Risk"
            color = "orange"
        else:
            result = "High Risk"
            color = "red"

        suggestion_text = "<br>".join(suggestions) if suggestions else "✅ You are maintaining a healthy lifestyle!"

        return render_template(
            'index.html',
            prediction=result,
            suggestion=suggestion_text,
            color=color,
            form_data=request.form
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import random
import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import difflib

app = Flask(__name__)

# Paths
PDF_PATH = "wellness_plan.pdf"
DIET_DATA_PATH = "diet_plan_recommendations.csv"
YOGA_DATA_PATH = "yoga.csv"

# Load datasets at startup
diet_df = pd.read_csv(DIET_DATA_PATH)
yoga_df = pd.read_csv(YOGA_DATA_PATH, encoding='cp1252')

# Preprocess diet dataset
diet_df['Calories'] = pd.to_numeric(diet_df['Calories'], errors='coerce')
diet_df = diet_df.dropna(subset=['Calories'])
diet_df['Suitable For'] = diet_df['Suitable For'].astype(str).str.lower()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_plan', methods=['POST'])
def generate_plan():
    try:
        user_data = request.json

        # Map frontend goal strings to internal keys
        goal_mapping = {
            "lose weight": "weight_loss",
            "gain weight": "muscle_gain",
            "maintain weight": "fitness"
        }
        user_data['goal'] = goal_mapping.get(user_data['goal'].lower(), 'fitness')

        meals, total_calories = generate_diet(user_data)
        yoga_poses = generate_yoga(user_data)

        export_to_pdf(user_data, meals, yoga_poses, total_calories)

        return jsonify({"status": "success", "message": "Plan generated successfully."})
    except Exception as e:
        print("Error:", e)
        return jsonify({"status": "error", "message": str(e)})

@app.route('/download/pdf', methods=['GET'])
def download_pdf():
    try:
        if os.path.exists(PDF_PATH):
            return send_file(PDF_PATH, as_attachment=True)
        else:
            return jsonify({"status": "error", "message": "PDF file not found."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Helper Functions
def calculate_bmi(weight, height):
    try:
        height_m = height / 100
        return round(weight / (height_m ** 2), 1)
    except:
        return 0

def get_calorie_target(user):
    bmi = calculate_bmi(user['weight'], user['height'])
    base_calories = 2000 if user['gender'].lower() == 'female' else 2500

    if user['goal'] == 'weight_loss':
        return base_calories - 500
    elif user['goal'] == 'muscle_gain':
        return base_calories + 300
    else:
        return base_calories

def generate_diet(user):
    cal_target = get_calorie_target(user)
    meal_targets = {
        'Breakfast': cal_target * 0.25,
        'Lunch': cal_target * 0.30,
        'Snack': cal_target * 0.15,
        'Dinner': cal_target * 0.30,
    }

    pref = user.get('diet', '').lower().strip()
    filtered = diet_df[diet_df['Suitable For'].str.contains(pref, na=False)]

    meals = {}
    total_calories = 0

    for meal, target in meal_targets.items():
        meal_items = []
        meal_total = 0
        temp_df = filtered[filtered['Meal Type'].str.lower() == meal.lower()]

        while meal_total < target and not temp_df.empty:
            item = temp_df.sample().iloc[0]
            meal_items.append({
                'food': item['Diet Plan Name'],
                'calories': item['Calories']
            })
            meal_total += item['Calories']
            temp_df = temp_df[temp_df['Diet Plan Name'] != item['Diet Plan Name']]

        meals[meal] = {'items': meal_items, 'total': meal_total}
        total_calories += meal_total

    return meals, total_calories

def generate_yoga(user):
    try:
        focus_mapping = {
            'weight_loss': 'weight loss',
            'muscle_gain': 'strength',
            'fitness': 'general fitness'
        }
        focus_area = focus_mapping.get(user['goal'], 'general fitness')

        yoga_df['Focus Area'] = yoga_df['Focus Area'].str.lower().str.strip()

        filtered = yoga_df[yoga_df['Focus Area'].str.contains(focus_area, na=False)]

        if filtered.empty:
            filtered = yoga_df

        poses = filtered.sample(n=min(5, len(filtered)))

        # Map pose names to image files manually
        image_mapping = {
            "Extended Side Angle": "Extended-Side-Angle_Andrew-Clark.avif",
            "Forearm Plank": "Forearm-Plank-Mod-3_Andrew-Clark.avif",
            "Gate Pose": "Gate-Pose-Mod-1_Andrew-Clark-scaled.avif",
            "Half Lord of the Fishes": "Half-Lord-of-the-Fishes-Mod-1_Andrew-Clark_1.avif",
            "Janu Sirsasana": "Janu-Sirsasana_Andrew-Clark_1.avif",
            "Kundalini Meditation": "kundalini-meditation-elena-brower-1.avif",
            "Lizard Pose": "Lizard-Pose-Mod-1_Andrew-Clark.avif",
            "Prenatal Pose": "prenatal-pregnant-woman-yoga-pose-1.avif",
        }

        yoga_list = []
        for idx, row in poses.iterrows():
            pose_name = row.get('Pose Name', f"Pose {idx+1}")
            image_file = image_mapping.get(pose_name, None)  # Fetch image if exists
            yoga_list.append({
                'pose_name': pose_name,
                'focus_area': row.get('Focus Area', 'General'),
                'difficulty': row.get('Difficulty', 'Normal'),
                'image_file': image_file
            })

        return yoga_list

    except Exception as e:
        print(f"Error generating yoga plan: {e}")
        return []

   

def export_to_pdf(user_data, meals, yoga_poses, total_calories):
    try:
        create_calorie_chart(meals)

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Personalized Wellness Plan", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, f"Total Daily Calories: {int(total_calories)} kcal", ln=True)
        pdf.cell(200, 8, f"Age: {user_data['age']} | Gender: {user_data['gender'].capitalize()} | Activity Level: {user_data['activity'].capitalize()}", ln=True)
        pdf.ln(5)

        for meal, content in meals.items():
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, f"{meal} (Total: {int(content['total'])} kcal):", ln=True)
            pdf.set_font("Arial", '', 11)
            for item in content['items']:
                pdf.cell(200, 8, f"- {item['food']} ({int(item['calories'])} kcal)", ln=True)
            pdf.ln(2)

        if os.path.exists("calorie_chart.png"):
            pdf.image("calorie_chart.png", x=15, y=pdf.get_y(), w=180)
            pdf.ln(60)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Recommended Yoga Poses:", ln=True)
        pdf.ln(5)

        for pose in yoga_poses:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"\u2022 {pose['pose_name']}", ln=True)

            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, f"Focus Area: {pose['focus_area']} | Difficulty: {pose['difficulty']}", ln=True)
            pdf.ln(5)

        pdf.output(PDF_PATH)

    except Exception as e:
        print(f"Error generating PDF: {e}")



def create_calorie_chart(meals):
    labels = list(meals.keys())
    sizes = [meal['total'] for meal in meals.values()]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, sizes, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
    plt.title('Calorie Distribution per Meal')
    plt.ylabel('Calories')
    plt.tight_layout()
    plt.savefig("calorie_chart.png")
    plt.close()

def export_to_pdf(user_data, meals, yoga_poses, total_calories):
    try:
        create_calorie_chart(meals)

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Personalized Wellness Plan", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, f"Total Daily Calories: {int(total_calories)} kcal", ln=True)
        pdf.cell(200, 8, f"Age: {user_data['age']} | Gender: {user_data['gender'].capitalize()} | Activity Level: {user_data['activity'].capitalize()}", ln=True)
        pdf.ln(5)

        for meal, content in meals.items():
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, f"{meal} (Total: {int(content['total'])} kcal):", ln=True)
            pdf.set_font("Arial", '', 11)
            for item in content['items']:
                pdf.cell(200, 8, f"- {item['food']} ({int(item['calories'])} kcal)", ln=True)
            pdf.ln(2)

        if os.path.exists("calorie_chart.png"):
            pdf.image("calorie_chart.png", x=15, y=pdf.get_y(), w=180)
            pdf.ln(60)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Recommended Yoga Poses:", ln=True)
        pdf.ln(5)

        for pose in yoga_poses:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"\u2022 {pose['pose_name']}", ln=True)

            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, f"Focus Area: {pose['focus_area']} | Difficulty: {pose['difficulty']}", ln=True)
            pdf.ln(5)

        pdf.output(PDF_PATH)

    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == '__main__':
    app.run(debug=True)

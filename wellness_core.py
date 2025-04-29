import pandas as pd
import random
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import difflib

# Load the new diet dataset
def load_diet_data(file_path="diet_plan_recommendations.csv"):
    try:
        diet_df = pd.read_csv(file_path)
        if diet_df.empty:
            raise ValueError("Diet plan file is empty.")

        # Force 'Calories' to numeric, invalid parsing will become NaN
        diet_df['Calories'] = pd.to_numeric(diet_df['Calories'], errors='coerce')

        # Drop rows where Calories is NaN or missing
        diet_df = diet_df.dropna(subset=['Calories'])

        # Normalize 'Suitable For' to lower case for easier matching
        if 'Suitable For' in diet_df.columns:
            diet_df['Suitable For'] = diet_df['Suitable For'].astype(str).str.lower()

        return diet_df

    except Exception as e:
        print(f"Error loading diet data: {e}")
        return pd.DataFrame()

# Load the yoga poses dataset
def load_yoga_data(file_path="yoga.csv"):
    try:
        yoga_df = pd.read_csv(file_path)
        if yoga_df.empty:
            raise ValueError("Yoga poses file is empty.")
        return yoga_df
    except Exception as e:
        print(f"Error loading yoga data: {e}")
        return pd.DataFrame()

# Calculate BMI
def calculate_bmi(weight, height):
    try:
        weight = float(weight)
        height = float(height)
        if height <= 0 or weight <= 0:
            raise ValueError("Invalid weight or height.")
        height_m = height / 100  # convert to meters
        return round(weight / (height_m ** 2), 1)
    except Exception as e:
        print(f"Error calculating BMI: {e}")
        return 0

# Get calorie target
def get_calorie_target(user):
    try:
        bmi = calculate_bmi(user['weight'], user['height'])
        base_calories = 2000 if user['gender'].lower() == 'female' else 2500
        if user['goal'] == 'weight_loss':
            return base_calories - 500
        elif user['goal'] == 'muscle_gain':
            return base_calories + 300
        else:
            return base_calories
    except Exception as e:
        print(f"Error calculating calorie target: {e}")
        return 2000

# Generate diet plan
def generate_diet(user, diet_df):
    try:
        if diet_df.empty:
            print("Error: Diet data is empty.")
            return None, 0

        cal_target = get_calorie_target(user)
        print(f"Calculated Calorie Target: {cal_target}")

        meal_targets = {
            'Breakfast': cal_target * 0.25,
            'Lunch': cal_target * 0.30,
            'Snack': cal_target * 0.15,
            'Dinner': cal_target * 0.30,
        }

        pref = user.get('diet_pref', '').lower().strip()
        if not pref:
            print("Error: Missing or invalid diet preference.")
            return None, 0

        print(f"Diet Preference: {pref}")

        filtered = diet_df[diet_df['Suitable For'].str.contains(pref, na=False)]
        if filtered.empty:
            print(f"No items found for preference '{pref}'. Using all available data.")

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

        print(f"Total Calories: {total_calories}")

        return meals, total_calories

    except Exception as e:
        print(f"Error generating diet plan: {e}")
        return None, 0

# Create calorie chart
def create_calorie_chart(meals):

    try:
        if not meals:
            return
        labels = list(meals.keys())
        sizes = [meal['total'] for meal in meals.values()]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, sizes, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
        plt.title('Calorie Distribution per Meal')
        plt.ylabel('Calories')
        plt.tight_layout()
        plt.savefig("calorie_chart.png")
        plt.close()
    except Exception as e:
        print(f"Error creating calorie chart: {e}")

# Generate yoga poses
def generate_yoga(user, yoga_df):
    try:
        if yoga_df.empty:
            return pd.DataFrame()

        # In your new yoga.csv, choose based on "Focus Area" and "Difficulty"
        goal_mapping = {
            'weight_loss': 'Weight Loss',
            'stress_relief': 'Stress Relief',
            'muscle_gain': 'Strength'
        }

        focus_area = goal_mapping.get(user['goal'], 'General Fitness')
        filtered = yoga_df[yoga_df['Focus Area'].str.contains(focus_area, na=False, case=False)]

        if filtered.empty:
            filtered = yoga_df

        poses = filtered.sample(n=min(5, len(filtered)))

        return poses
    except Exception as e:
        print(f"Error generating yoga plan: {e}")
        return pd.DataFrame()

def export_to_pdf(user, meals, yoga_df, total_cal):
    try:
        create_calorie_chart(meals)

        pose_images = {
            'Forearm Plank': '/mnt/data/Forearm-Plank-Mod-3_Andrew-Clark.avif',
            'Gate Pose': '/mnt/data/Gate-Pose-Mod-1_Andrew-Clark-scaled.avif',
            'Downward Dog': '/mnt/data/GettyImages-1274718580-scaled.avif',
            'Head-to-Knee Forward Bend': '/mnt/data/Janu-Sirsasana_Andrew-Clark_1.avif',
            'Kundalini Meditation': '/mnt/data/kundalini-meditation-elena-brower-1.avif',
            'Lizard Pose': '/mnt/data/Lizard-Pose-Mod-1_Andrew-Clark.avif',
            'Prenatal Pose': '/mnt/data/prenatal-pregnant-woman-yoga-pose-1.avif',
        }

        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Personalized Wellness Plan", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, f"Total Daily Calories: {int(total_cal)} kcal", ln=True)
        pdf.ln(5)

        # Meal Plans
        for meal, content in meals.items():
            pdf.set_font("Arial", 'B', 13)
            pdf.cell(200, 10, f"{meal} (Total: {int(content['total'])} kcal):", ln=True)
            pdf.set_font("Arial", '', 11)
            for item in content['items']:
                pdf.cell(200, 8, f"- {item['food']} ({int(item['calories'])} kcal)", ln=True)
            pdf.ln(2)

        # Insert Calorie Chart
        if os.path.exists("calorie_chart.png"):
            pdf.ln(5)
            pdf.image("calorie_chart.png", x=15, w=180)
            pdf.ln(10)

        # Yoga Poses Section
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Recommended Yoga Poses:", ln=True)
        pdf.ln(8)

        pose_names_list = list(pose_images.keys())
        yoga_poses = []

        import difflib  # In case not already imported

        # Match poses with images
        for idx, row in yoga_df.iterrows():
            pose_name = row.get('Pose Name', f"Pose {idx+1}")
            focus_area = row.get('Focus Area', 'General')
            difficulty = row.get('Difficulty', 'Normal')

            closest_matches = difflib.get_close_matches(pose_name, pose_names_list, n=1, cutoff=0.5)
            matched_pose = closest_matches[0] if closest_matches else None
            img_path = pose_images.get(matched_pose)

            yoga_poses.append({
                'pose_name': pose_name,
                'focus_area': focus_area,
                'difficulty': difficulty,
                'img_path': img_path
            })

        # Show two poses per row
        for i in range(0, len(yoga_poses), 2):
            poses_pair = yoga_poses[i:i+2]

            start_y = pdf.get_y()

            for j, pose in enumerate(poses_pair):
                x_offset = 10 + (j * 100)  # First image at x=10, second image at x=110

                # Insert Image
                if pose['img_path'] and os.path.exists(pose['img_path']):
                    try:
                        pdf.image(pose['img_path'], x=x_offset, y=start_y, w=85)
                    except Exception as img_error:
                        print(f"Could not add image for {pose['pose_name']}: {img_error}")

            # Move cursor below images
            pdf.ln(60)

            # Insert Text under each image
            for j, pose in enumerate(poses_pair):
                x_offset = 10 + (j * 100)
                pdf.set_xy(x_offset, pdf.get_y())

                pdf.set_font("Arial", 'B', 10)
                pdf.multi_cell(85, 5, f"{pose['pose_name']}", align="C")

                pdf.set_font("Arial", '', 9)
                pdf.multi_cell(85, 5, f"Focus: {pose['focus_area']}\nDifficulty: {pose['difficulty']}", align="C")

            pdf.ln(10)

        # Save final PDF
        pdf.output("wellness_plan.pdf")
        print("✅ PDF created: wellness_plan.pdf")

    except Exception as e:
        print(f"Error exporting to PDF: {e}")


# Export wellness plan to PDF
def export_to_pdf(user, meals, yoga_df, total_cal):
    try:
        create_calorie_chart(meals)

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Personalized Wellness Plan", ln=True, align="C")

        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, f"Total Daily Calories: {int(total_cal)} kcal", ln=True)

        for meal, content in meals.items():
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, f"{meal} (Total: {int(content['total'])} kcal):", ln=True)
            pdf.set_font("Arial", '', 11)
            for item in content['items']:
                pdf.cell(200, 8, f"- {item['food']} ({int(item['calories'])} kcal)", ln=True)
            pdf.ln(2)

        if os.path.exists("calorie_chart.png"):
            pdf.image("calorie_chart.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(60)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Recommended Yoga Poses:", ln=True)
        pdf.ln(5)

        for idx, row in yoga_df.iterrows():
            pose_name = row.get('Pose Name', f"Pose {idx+1}")

            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"\u2022 {pose_name}", ln=True)

            pdf.set_font("Arial", '', 10)
            focus_area = row.get('Focus Area', 'General')
            difficulty = row.get('Difficulty', 'Normal')
            pdf.cell(0, 6, f"Focus Area: {focus_area} | Difficulty: {difficulty}", ln=True)
            pdf.ln(5)

        pdf.output("wellness_plan.pdf")
        print("PDF created: wellness_plan.pdf")

    except Exception as e:
        print(f"Error exporting to PDF: {e}")

# Main function
def main(user):
    diet_df = load_diet_data()
    yoga_df = load_yoga_data()

    meals, total_calories = generate_diet(user, diet_df)
    yoga_poses = generate_yoga(user, yoga_df)

    if meals and not yoga_poses.empty:
        export_to_pdf(user, meals, yoga_poses, total_calories)
    else:
        print("Plan generation failed.")


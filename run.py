import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("text_generation") or os.getenv("text_generation")

# Initialize Gemini Client
client = genai.Client(api_key=API_KEY) if API_KEY else genai.Client()

MODEL_NAME = "gemini-3.6-flash"

def run_zero_shot(student_data: dict) -> str:
    """Zero-shot Prompting for AI Study Coach."""
    prompt = f"""
Analyze the following student's situation and create a detailed study plan.

Student Name: {student_data.get('name')}
Subject: {student_data.get('subject')}
Weak Topics: {student_data.get('weak_topics')}
Days Remaining: {student_data.get('days_remaining')} days
Study Hours per Day: {student_data.get('hours_per_day')} hours
Current Skill Level: {student_data.get('skill_level')}

Please:
1. Identify the student's weak areas.
2. Prioritize the important topics.
3. Create a daily study schedule.
4. Recommend appropriate study activities for each day.
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text


def run_few_shot(student_data: dict) -> str:
    """Few-shot Prompting with Examples for AI Study Coach."""
    prompt = f"""
You are an AI Study Coach. Analyze student information and generate a study plan.

Examples:

Example 1:
Student Information:
- Subject: SQL
- Exam In: 3 days
- Weak Topics: SQL joins
Recommended Plan:
- Day 1: Study INNER, LEFT, RIGHT, and FULL joins
- Day 2: Practice SQL join queries
- Day 3: Complete a mock test and review mistakes

Example 2:
Student Information:
- Subject: Python
- Exam In: 7 days
- Weak Topics: Object-Oriented Programming
Recommended Plan:
- Day 1-2: Classes and objects
- Day 3-4: Inheritance and polymorphism
- Day 5-6: Coding exercises
- Day 7: Final revision and mock test

Now create a study plan for the following student:

Student Name: {student_data.get('name')}
Subject: {student_data.get('subject')}
Weak Topics: {student_data.get('weak_topics')}
Days Remaining: {student_data.get('days_remaining')} days
Study Hours per Day: {student_data.get('hours_per_day')} hours
Current Skill Level: {student_data.get('skill_level')}

Recommended Plan:
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text


def run_chain_of_thought(student_data: dict) -> str:
    """Structured Reasoning / Chain of Thought Prompting."""
    prompt = f"""
Analyze the following student's situation systematically before giving the final study plan.

Student Details:
- Name: {student_data.get('name')}
- Subject: {student_data.get('subject')}
- Weak Topics: {student_data.get('weak_topics')}
- Days Remaining: {student_data.get('days_remaining')} days
- Study Hours per Day: {student_data.get('hours_per_day')} hours
- Skill Level: {student_data.get('skill_level')}

Please follow these steps in your output:
1. Identify the student's weak topics.
2. Determine which topics are most important to focus on first.
3. Consider how many days remain before the exam ({student_data.get('days_remaining')} days).
4. Consider how many hours the student can study per day ({student_data.get('hours_per_day')} hours/day).
5. Recommend suitable learning activities.
6. Generate the final day-by-day study plan.
7. Provide a short explanation/justification for why each activity was selected.
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text

print("Gemini Client initialized successfully.")
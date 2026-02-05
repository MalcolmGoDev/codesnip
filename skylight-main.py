import functions_framework
from flask import jsonify
import requests
from datetime import date, datetime
import os

SKYLIGHT_TOKEN = os.environ.get("SKYLIGHT_TOKEN")
FRAME_ID = os.environ.get("FRAME_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_URL = "https://app.ourskylight.com"

HEADERS = {
    "Authorization": SKYLIGHT_TOKEN,
    "User-Agent": "SkylightMobile/1.95.2 (ios 26.2)",
    "Accept": "application/json"
}

FAMILY_MEMBERS = ["banksy", "miles", "sophia", "malcolm", "deborah", "house"]

TIME_PERIODS = {
    "morning": (5, 12),
    "afternoon": (12, 17),
    "evening": (17, 23),
    "night": (17, 23),
    "today": (0, 24),
    "all": (0, 24)
}

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"


def call_llm(prompt):
    """Call Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": f"<s>[INST] {prompt} [/INST]",
        "parameters": {"max_new_tokens": 100, "temperature": 0.1}
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        return None
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        text = result[0].get("generated_text", "")
        if "[/INST]" in text:
            text = text.split("[/INST]")[-1].strip()
        return text
    return None


def get_todays_chores():
    today = date.today().isoformat()
    resp = requests.get(
        f"{BASE_URL}/api/frames/{FRAME_ID}/chores",
        headers=HEADERS,
        params={"after": today, "before": today, "include_late": "true"}
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    chores = data.get("data", [])
    categories = {}
    for item in data.get("included", []):
        if item.get("type") == "category":
            categories[item["id"]] = item["attributes"].get("label", "Unknown")
    chore_list = []
    for chore in chores:
        attrs = chore["attributes"]
        chore_id = chore["id"]
        name = attrs.get("summary", "")
        status = attrs.get("status", "")
        schedule_time = attrs.get("scheduled_at", "")
        cat_id = chore.get("relationships", {}).get("category", {}).get("data", {}).get("id")
        assigned_to = categories.get(cat_id, "Unassigned")
        hour = None
        if schedule_time:
            try:
                dt = datetime.fromisoformat(schedule_time.replace("Z", "+00:00"))
                hour = dt.hour
            except:
                pass
        chore_list.append({
            "id": chore_id,
            "name": name,
            "assigned_to": assigned_to,
            "status": status,
            "hour": hour
        })
    return chore_list


def ask_llm_for_chore(chore_query, chore_list):
    incomplete = [c for c in chore_list if c["status"] != "complete"]
    chores_text = "\n".join([f"ID:{c['id']} Name:{c['name']} Person:{c['assigned_to']}" for c in incomplete])
    prompt = f'''Match this request to a chore ID.

Request: "{chore_query}"

Chores:
{chores_text}

Reply with ONLY the matching ID, nothing else. If no match, reply NONE.'''
    result = call_llm(prompt)
    if result:
        result = result.strip().split()[0]
        for c in incomplete:
            if c["id"] in result:
                return c["id"]
    return None


def complete_chore(chore_id):
    resp = requests.put(
        f"{BASE_URL}/api/frames/{FRAME_ID}/chores/{chore_id}",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"status": "complete"}
    )
    return resp.status_code >= 200 and resp.status_code < 300


def parse_query(query):
    query_lower = query.lower()
    person = None
    period = None
    for name in FAMILY_MEMBERS:
        if name in query_lower:
            person = name
            break
    for p in TIME_PERIODS.keys():
        if p in query_lower:
            period = p
            break
    return person, period


def filter_by_person_and_time(chore_list, person=None, period=None):
    results = []
    if period:
        start_hour, end_hour = TIME_PERIODS.get(period, (0, 24))
    else:
        start_hour, end_hour = 0, 24
    for chore in chore_list:
        if person and chore["assigned_to"].lower() != person.lower():
            continue
        if period and period not in ["today", "all"]:
            if chore["hour"] is None:
                continue
            if not (start_hour <= chore["hour"] < end_hour):
                continue
        if chore["status"] == "complete":
            continue
        results.append(chore)
    return results


def build_speakable_response(chores, person, period):
    count = len(chores)
    if count == 0:
        if person and period:
            return f"{person.title()} has no {period} tasks left."
        elif person:
            return f"{person.title()} has no tasks left today."
        else:
            return "No tasks found."
    period_text = f" {period}" if period and period not in ["today", "all"] else ""
    if count == 1:
        task_name = chores[0]["name"]
        return f"{person.title() if person else 'There'} has 1{period_text} task left: {task_name}."
    task_names = [c["name"] for c in chores[:3]]
    if count <= 3:
        tasks_text = ", ".join(task_names[:-1]) + f" and {task_names[-1]}"
        return f"{person.title() if person else 'There'} has {count}{period_text} tasks left: {tasks_text}."
    else:
        tasks_text = ", ".join(task_names)
        return f"{person.title() if person else 'There'} has {count}{period_text} tasks left, including {tasks_text}."


@functions_framework.http
def main(request):
    data = request.get_json(silent=True) or {}
    action = data.get("action", "complete")
    if action == "query":
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "No query specified"}), 400
        chore_list = get_todays_chores()
        if chore_list is None:
            return jsonify({"error": "Failed to fetch chores"}), 500
        person, period = parse_query(query)
        if not person:
            return jsonify({"error": "Couldn't identify family member. Try: Miles morning tasks"}), 400
        filtered = filter_by_person_and_time(chore_list, person, period)
        response_text = build_speakable_response(filtered, person, period)
        return jsonify({
            "success": True,
            "person": person,
            "period": period,
            "count": len(filtered),
            "tasks": [c["name"] for c in filtered],
            "speech": response_text
        })
    else:
        chore_query = data.get("chore", "").strip()
        if not chore_query:
            return jsonify({"error": "No chore specified"}), 400
        chore_list = get_todays_chores()
        if chore_list is None:
            return jsonify({"error": "Failed to fetch chores"}), 500
        incomplete = [c for c in chore_list if c["status"] != "complete"]
        if not incomplete:
            return jsonify({"error": "No incomplete chores today"}), 404
        matched_id = ask_llm_for_chore(chore_query, chore_list)
        if not matched_id:
            return jsonify({"error": f"No match for '{chore_query}'"}), 404
        chore_name = next((c["name"] for c in chore_list if c["id"] == matched_id), "Unknown")
        assigned_to = next((c["assigned_to"] for c in chore_list if c["id"] == matched_id), "Unknown")
        if complete_chore(matched_id):
            return jsonify({"success": True, "chore": chore_name, "assigned_to": assigned_to})
        else:
            return jsonify({"error": "Failed to complete chore"}), 500


import os
import re
import json
import uuid
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from ddgs import DDGS
from groq import Groq
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL_ID = "openai/gpt-oss-20b"

BUDGET_CATEGORIES = [
    "Round-trip transport",
    "Accommodation",
    "Food",
    "Local transport",
    "Activities and entry",
    "Other expenses",
]

st.set_page_config(
    page_title="Agentic AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        font-size: 2.3rem;
        font-weight: 800;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GROQ AI FUNCTIONS
# ============================================================

def get_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    return Groq(
        api_key=api_key,
        timeout=90.0,
        max_retries=1,
    )


def ask_ai(system_prompt, user_prompt, max_tokens=4000):
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )

    if not response.choices:
        raise ValueError("The AI returned no response.")

    choice = response.choices[0]
    content = choice.message.content or ""

    if choice.finish_reason == "length":
        raise ValueError(
            "The AI reached its output token limit. "
            "Try a shorter trip or retry later."
        )

    if not content.strip():
        raise ValueError(
            "The AI returned an empty response."
        )

    return content.strip()


def format_error(error):
    message = str(error)
    lower = message.lower()

    if "429" in message or "rate_limit" in lower:
        return (
            "Groq usage limit reached. "
            "Wait for the quota to reset and try again."
        )

    if "model_not_found" in lower:
        return f"Model unavailable: {MODEL_ID}"

    if "401" in message or "authentication" in lower:
        return (
            "Authentication failed. "
            "Check GROQ_API_KEY in your .env file."
        )

    return message[:800]


def parse_json_response(text):
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])

        raise ValueError(
            "The AI returned invalid JSON. "
            "Please generate the plan again."
        )


# ============================================================
# GENERAL UTILITIES
# ============================================================

def money(value):
    if value is None:
        return "Unknown"

    try:
        if pd.isna(value):
            return "Unknown"

        return f"₹{float(value):,.0f}"

    except (TypeError, ValueError):
        return "Unknown"


def valid_number(value):
    if value is None or isinstance(value, bool):
        return None

    try:
        if isinstance(value, str):
            value = (
                value.replace(",", "")
                .replace("₹", "")
                .strip()
            )

        number = float(value)

        if not pd.notna(number) or number < 0:
            return None

        return number

    except (TypeError, ValueError):
        return None


def calculate_car_cost(
    distance,
    mileage,
    fuel_price,
    toll_parking,
):
    if distance <= 0 or mileage <= 0:
        return None

    round_trip_distance = distance * 2
    fuel_litres = round_trip_distance / mileage
    fuel_cost = fuel_litres * fuel_price

    return {
        "distance": round_trip_distance,
        "fuel_litres": fuel_litres,
        "fuel_cost": fuel_cost,
        "toll_parking": toll_parking,
        "total": fuel_cost + toll_parking,
    }


# ============================================================
# DESTINATION RESEARCH
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def search_travel(origin, destination, travel_date):
    queries = [
        f"{destination} official tourism attractions",
        f"{origin} to {destination} train bus travel",
        f"{destination} accommodation food local transport costs",
        f"{destination} attractions opening hours tourism",
    ]

    results = []

    try:
        with DDGS() as ddgs:
            for query in queries:
                try:
                    for item in ddgs.text(
                        query,
                        max_results=3,
                    ):
                        url = item.get("href", "")

                        if not url:
                            continue

                        results.append(
                            {
                                "title": item.get(
                                    "title",
                                    "Travel information",
                                ),
                                "url": url,
                                "snippet": item.get(
                                    "body",
                                    "",
                                ),
                            }
                        )

                except Exception:
                    continue

    except Exception:
        pass

    unique = {}

    for item in results:
        unique[item["url"]] = item

    return list(unique.values())[:10]


def source_context(sources):
    if not sources:
        return (
            "No research results available. "
            "Do not claim unverified details are confirmed."
        )

    return "\n\n".join(
        f"Source: {item['title']}\n"
        f"URL: {item['url']}\n"
        f"Snippet: {item['snippet'][:250]}"
        for item in sources[:8]
    )


# ============================================================
# TRIP GENERATION
# ============================================================

SYSTEM_PROMPT = """
You are a careful AI travel planning assistant.

Create realistic travel plans with sensible cost estimates.

Never invent:
- Confirmed train or flight schedules
- Live ticket prices
- Hotel availability
- Booking confirmations
- Confirmed attraction opening hours

Research snippets are unverified reference material.

Return valid JSON only when requested.

For budget calculations:
- Use numeric unit rates and quantities.
- Calculate group costs consistently.
- Use null when a meaningful estimate is unavailable.
- Never treat unknown expenses as free.
- Never force expenses to fit the user's budget.
- Clearly describe estimates as unverified.
"""


def generate_trip(details, sources, car_info):
    start_date = date.fromisoformat(
        details["departure_date"]
    )

    final_date = start_date + timedelta(
        days=details["duration"] - 1
    )

    days = int(details["duration"])
    nights = max(0, days - 1)
    travelers = int(details["travelers"])

    prompt = f"""
Create a practical travel plan.

TRIP DETAILS
Starting city: {details['origin']}
Destination: {details['destination']}
Departure date: {start_date.isoformat()}
Final date: {final_date.isoformat()}
Trip duration: {days} days
Hotel nights assumed: {nights}
Number of travelers: {travelers}
Total group budget: INR {details['budget']}
Travel style: {details['style']}
Preferred transport: {details['transport_mode']}

Train class: {details['train_class']}
Train quota: {details['train_quota']}
Bus type: {details['bus_type']}
Flight cabin: {details['flight_class']}

CAR INFORMATION
{car_info}

RESEARCH
{source_context(sources)}

TRANSPORT REQUIREMENTS
- Compare Own car, Train, Bus, and Flight where practical.
- Recommend one primary transport mode.
- Respect the user's preferred transport.
- Do not invent exact train numbers or schedules.
- All fares are estimates, not confirmed prices.
- If road distance is unknown, car fuel cost is unknown.
- Transport options are alternatives.
- Only the recommended transport goes in the main budget.
- General train quota is not a train class.

ITINERARY REQUIREMENTS
- Exactly {days} itinerary days.
- Start on {start_date.isoformat()}.
- End on {final_date.isoformat()}.
- Avoid unrealistic travel schedules.
- Group nearby attractions together.
- Avoid distant attractions in the same morning.
- Keep the return day light.
- Do not assume unverified arrival times.
- Identify weather-dependent activities.

BUDGET CALCULATION REQUIREMENTS

Every budget item must include:
- category
- group_cost
- unit_rate
- quantity
- unit
- assumptions

All monetary values are in Indian rupees.

Calculate:

group_cost = unit_rate * quantity

Use these rules:

1. Round-trip transport:
   For trains, buses, and flights:
   unit_rate = estimated one-way fare per person
   quantity = {travelers * 2}
   unit = "one-way passenger tickets"

   For own car:
   If entered road distance is available, use the
   car calculation supplied above.
   Otherwise use null.

2. Accommodation:
   unit_rate = estimated nightly cost for rooms
   needed by the whole group
   quantity = {nights}
   unit = "nights"

3. Food:
   unit_rate = estimated daily food cost for the
   entire group
   quantity = {days}
   unit = "days"

4. Local transport:
   unit_rate = estimated daily local transport
   cost for the entire group
   quantity = {days}
   unit = "days"

5. Activities and entry:
   unit_rate = estimated combined activity expense
   per traveler
   quantity = {travelers}
   unit = "travelers"

6. Other expenses:
   unit_rate = estimated miscellaneous group allowance
   quantity = 1
   unit = "group allowance"

IMPORTANT:
- group_cost must equal unit_rate * quantity.
- Use numeric values, not strings.
- If an estimate cannot reasonably be made,
  set group_cost, unit_rate and quantity to null.
- Do not use zero merely because a price is unverified.
- Do not put cost calculations only in assumptions.
- Do not invent confirmed fares.
- Do not force totals to fit the group budget.
- Explain every cost assumption.
- Keep the budget items in the six required categories.

RETURN EXACTLY ONE VALID JSON OBJECT.

The numbers below demonstrate the JSON structure only.
Do not reuse the example numbers automatically.

{{
  "overview": "Short destination summary",
  "transport_options": [
    {{
      "mode": "Train",
      "group_round_trip_cost": 4000,
      "journey_time": "Verify current train schedule",
      "advantages": "Advantages",
      "disadvantages": "Disadvantages",
      "verification": "Check fare and seat availability"
    }}
  ],
  "recommended_mode": "Train",
  "recommendation_reason": "Explain recommendation",
  "itinerary": [
    {{
      "day": 1,
      "date": "{start_date.isoformat()}",
      "morning": "Plan",
      "afternoon": "Plan",
      "evening": "Plan",
      "notes": "Feasibility caveats"
    }}
  ],
  "budget_items": [
    {{
      "category": "Round-trip transport",
      "group_cost": 4000,
      "unit_rate": 1000,
      "quantity": 4,
      "unit": "one-way passenger tickets",
      "assumptions": "Illustrative estimate only"
    }},
    {{
      "category": "Accommodation",
      "group_cost": 3000,
      "unit_rate": 1500,
      "quantity": 2,
      "unit": "nights",
      "assumptions": "Illustrative guesthouse estimate"
    }},
    {{
      "category": "Food",
      "group_cost": 3000,
      "unit_rate": 1000,
      "quantity": 3,
      "unit": "days",
      "assumptions": "Illustrative daily group food cost"
    }},
    {{
      "category": "Local transport",
      "group_cost": 1500,
      "unit_rate": 500,
      "quantity": 3,
      "unit": "days",
      "assumptions": "Illustrative daily local transport cost"
    }},
    {{
      "category": "Activities and entry",
      "group_cost": 500,
      "unit_rate": 250,
      "quantity": 2,
      "unit": "travelers",
      "assumptions": "Illustrative activity allowance"
    }},
    {{
      "category": "Other expenses",
      "group_cost": 500,
      "unit_rate": 500,
      "quantity": 1,
      "unit": "group allowance",
      "assumptions": "Illustrative miscellaneous allowance"
    }}
  ],
  "accommodation_food": [
    {{
      "item": "Accommodation",
      "location": "Suitable area",
      "price_range": "Estimated range",
      "notes": "Practical guidance"
    }}
  ],
  "booking_checklist": [
    "Check ticket availability"
  ],
  "practical_tips": [
    "Practical travel tip"
  ],
  "limitations": [
    "Details requiring verification"
  ]
}}

Return exactly {days} itinerary objects.

Return exactly six budget items.

Keep the JSON concise.
"""

    raw = ask_ai(
        SYSTEM_PROMPT,
        prompt,
        max_tokens=6000,
    )

    result = parse_json_response(raw)

    required = [
        "overview",
        "transport_options",
        "recommended_mode",
        "recommendation_reason",
        "itinerary",
        "budget_items",
        "accommodation_food",
        "booking_checklist",
        "practical_tips",
        "limitations",
    ]

    for field in required:
        if field not in result:
            raise ValueError(
                f"AI response missing required field: {field}"
            )

    if not isinstance(result["itinerary"], list):
        raise ValueError("Invalid itinerary format.")

    if len(result["itinerary"]) != days:
        raise ValueError(
            "The AI returned an incorrect number of days."
        )

    for index, item in enumerate(result["itinerary"]):
        item["day"] = index + 1
        item["date"] = (
            start_date + timedelta(days=index)
        ).isoformat()

    return result


# ============================================================
# AUTOMATIC BUDGET CALCULATION
# ============================================================

def initialize_budget(plan, details, car_info):
    """
    Calculate group costs from structured AI estimates.

    Priority:
    1. unit_rate * quantity
    2. direct group estimate
    3. unknown cost

    Python performs arithmetic instead of trusting
    inconsistent AI totals.
    """

    raw_items = {
        str(item.get("category", "")).strip().casefold(): item
        for item in plan.get("budget_items", [])
        if isinstance(item, dict)
    }

    days = int(details["duration"])
    nights = max(0, days - 1)
    travelers = int(details["travelers"])

    rows = []

    for category in BUDGET_CATEGORIES:
        item = raw_items.get(category.casefold(), {})

        assumptions = str(
            item.get("assumptions", "")
        ).strip()

        direct_cost = valid_number(
            item.get("group_cost")
        )

        rate = valid_number(
            item.get("unit_rate")
        )

        quantity = valid_number(
            item.get("quantity")
        )

        unit = str(
            item.get("unit", "")
        ).strip().lower()

        # Enforce correct quantity for standard categories.
        if category == "Round-trip transport":
            recommended = str(
                plan.get("recommended_mode", "")
            ).strip().casefold()

            if recommended not in {"own car", "car"}:
                if unit in {
                    "one-way passenger tickets",
                    "one-way tickets",
                }:
                    quantity = travelers * 2

        elif category == "Accommodation":
            if unit == "nights":
                quantity = nights

        elif category in {"Food", "Local transport"}:
            if unit == "days":
                quantity = days

        elif category == "Activities and entry":
            if unit == "travelers":
                quantity = travelers

        elif category == "Other expenses":
            if unit == "group allowance":
                quantity = 1

        calculated_cost = None
        calculation_note = ""

        if rate is not None and quantity is not None:
            calculated_cost = round(
                rate * quantity,
                2,
            )

            calculation_note = (
                f"Calculated: ₹{rate:,.0f} × "
                f"{quantity:g} {unit or 'units'} "
                f"= ₹{calculated_cost:,.0f}."
            )

        elif category == "Accommodation" and nights == 0:
            calculated_cost = 0

            calculation_note = (
                "No overnight accommodation assumed."
            )

        elif direct_cost is not None:
            calculated_cost = direct_cost

            calculation_note = (
                "Direct AI group estimate; "
                "unit calculation unavailable."
            )

        else:
            calculated_cost = None

            calculation_note = (
                "Cost unknown; enter a planning estimate."
            )

        explicitly_free = any(
            phrase in assumptions.lower()
            for phrase in [
                "free of charge",
                "no expense",
                "no cost",
                "free activity",
            ]
        )

        if calculated_cost == 0:
            zero_allowed = (
                explicitly_free
                or (
                    category == "Accommodation"
                    and nights == 0
                )
            )

            if not zero_allowed:
                calculated_cost = None
                calculation_note = (
                    "Cost unknown; zero was not justified."
                )

        if (
            direct_cost is not None
            and calculated_cost is not None
            and rate is not None
            and quantity is not None
            and abs(direct_cost - calculated_cost) > 1
        ):
            calculation_note += (
                f" AI supplied ₹{direct_cost:,.0f}, "
                "but Python used the calculated total."
            )

        description = (
            calculation_note
            + (
                " " + assumptions
                if assumptions
                else ""
            )
        )

        rows.append(
            {
                "Category": category,
                "Estimated group cost (₹)": calculated_cost,
                "Assumptions": description,
            }
        )

    # --------------------------------------------------------
    # OWN-CAR TRANSPORT OVERRIDE
    # --------------------------------------------------------

    recommended = str(
        plan.get("recommended_mode", "")
    ).strip().casefold()

    if recommended in {"own car", "car"}:
        calculated_car = None

        if details["road_distance"] > 0:
            calculated_car = calculate_car_cost(
                details["road_distance"],
                details["car_mileage"],
                details["fuel_price"],
                details["toll_parking"],
            )

        for row in rows:
            if row["Category"] == "Round-trip transport":
                if calculated_car is not None:
                    row["Estimated group cost (₹)"] = (
                        calculated_car["total"]
                    )

                    row["Assumptions"] = (
                        "Calculated from entered road distance, "
                        "mileage, fuel price, tolls and parking. "
                        "Vehicle maintenance excluded."
                    )

                else:
                    row["Estimated group cost (₹)"] = None

                    row["Assumptions"] = (
                        "Car cost unknown. Enter a verified "
                        "one-way road distance and regenerate."
                    )

    return pd.DataFrame(rows)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero">✈️ Agentic AI Travel Planner</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI Research • Smart Budgeting • Transport Comparison '
    '• Itinerary Planning'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("✈️ Plan Your Trip")

    with st.form("trip_form"):
        origin = st.text_input(
            "Starting city",
            value="Pune",
        )

        destination = st.text_input(
            "Destination",
            value="Goa",
        )

        departure_date = st.date_input(
            "Departure date",
            value=date.today(),
            min_value=date.today(),
        )

        duration = st.number_input(
            "Trip duration (days)",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
        )

        travelers = st.number_input(
            "Number of travelers",
            min_value=1,
            max_value=50,
            value=2,
            step=1,
        )

        budget = st.number_input(
            "Total group budget (₹)",
            min_value=0,
            value=15000,
            step=1000,
        )

        travel_style = st.selectbox(
            "Travel style",
            [
                "Budget",
                "Balanced",
                "Luxury",
                "Adventure",
                "Family",
                "Food and Culture",
            ],
        )

        transport_mode = st.selectbox(
            "Preferred transport",
            [
                "Compare all options",
                "Own car",
                "Train",
                "Bus",
                "Flight",
            ],
        )

        st.markdown("#### 🚗 Own-car details")

        car_mileage = st.number_input(
            "Car mileage (km/litre)",
            min_value=1.0,
            max_value=60.0,
            value=15.0,
            step=1.0,
        )

        fuel_price = st.number_input(
            "Fuel price (₹/litre)",
            min_value=1.0,
            max_value=300.0,
            value=105.0,
            step=1.0,
        )

        road_distance = st.number_input(
            "One-way road distance (km)",
            min_value=0,
            max_value=5000,
            value=0,
            step=10,
        )

        toll_parking = st.number_input(
            "Round-trip toll and parking (₹)",
            min_value=0,
            value=0,
            step=100,
        )

        st.markdown("#### 🚆 Train preferences")

        train_class = st.selectbox(
            "Preferred train class",
            [
                "Any class",
                "SL",
                "3A",
                "2A",
                "1A",
                "CC",
                "EC",
            ],
        )

        train_quota = st.selectbox(
            "Train quota",
            [
                "General",
                "Tatkal",
                "Ladies",
                "Senior Citizen",
            ],
        )

        st.markdown("#### 🚌 Bus preferences")

        bus_type = st.selectbox(
            "Preferred bus type",
            [
                "Any bus",
                "AC Seater",
                "Non-AC Seater",
                "AC Sleeper",
                "Non-AC Sleeper",
                "Volvo / Premium",
            ],
        )

        st.markdown("#### ✈️ Flight preferences")

        flight_class = st.selectbox(
            "Flight cabin",
            [
                "Economy",
                "Premium Economy",
                "Business",
            ],
        )

        generate = st.form_submit_button(
            "Generate My Trip",
            type="primary",
            use_container_width=True,
        )

    st.caption(f"AI model: {MODEL_ID}")
    st.caption(
        "Research results are cached for 30 minutes."
    )


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "trip_plan": None,
    "trip_sources": [],
    "trip_details": {},
    "trip_chat": [],
    "trip_review": "",
    "budget_table": None,
    "budget_editor_id": None,
    "destination_map": None,
    "destination_map_name": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERATE TRIP
# ============================================================

if generate:
    if not origin.strip() or not destination.strip():
        st.error(
            "Please enter both starting city and destination."
        )

    elif origin.strip().casefold() == destination.strip().casefold():
        st.error(
            "Starting city and destination must be different."
        )

    else:
        details = {
            "origin": origin.strip(),
            "destination": destination.strip(),
            "departure_date": departure_date.isoformat(),
            "duration": int(duration),
            "travelers": int(travelers),
            "budget": int(budget),
            "style": travel_style,
            "transport_mode": transport_mode,
            "car_mileage": float(car_mileage),
            "fuel_price": float(fuel_price),
            "road_distance": float(road_distance),
            "toll_parking": float(toll_parking),
            "train_class": train_class,
            "train_quota": train_quota,
            "bus_type": bus_type,
            "flight_class": flight_class,
        }

        st.session_state.trip_plan = None
        st.session_state.trip_sources = []
        st.session_state.trip_details = details
        st.session_state.trip_chat = []
        st.session_state.trip_review = ""
        st.session_state.budget_table = None
        st.session_state.budget_editor_id = uuid.uuid4().hex
        st.session_state.destination_map = None
        st.session_state.destination_map_name = ""

        with st.status(
            "Researching and building your trip...",
            expanded=True,
        ) as status:
            try:
                st.write(
                    "Step 1/2: Searching travel information."
                )

                sources = search_travel(
                    details["origin"],
                    details["destination"],
                    details["departure_date"],
                )

                st.session_state.trip_sources = sources

                st.write(
                    "Step 2/2: Generating itinerary and budget."
                )

                car_info = (
                    "Road distance unknown. "
                    "Do not invent car fuel costs."
                )

                if details["road_distance"] > 0:
                    estimate = calculate_car_cost(
                        details["road_distance"],
                        details["car_mileage"],
                        details["fuel_price"],
                        details["toll_parking"],
                    )

                    car_info = (
                        f"One-way distance: "
                        f"{details['road_distance']} km. "
                        f"Round-trip distance: "
                        f"{estimate['distance']:.0f} km. "
                        f"Fuel: {estimate['fuel_litres']:.1f} L. "
                        f"Fuel cost: ₹{estimate['fuel_cost']:.0f}. "
                        f"Tolls and parking: "
                        f"₹{estimate['toll_parking']:.0f}. "
                        f"Total: ₹{estimate['total']:.0f}."
                    )

                plan = generate_trip(
                    details,
                    sources,
                    car_info,
                )

                st.session_state.trip_plan = plan

                st.session_state.budget_table = (
                    initialize_budget(
                        plan,
                        details,
                        car_info,
                    )
                )

                status.update(
                    label="Travel plan generated successfully.",
                    state="complete",
                    expanded=False,
                )

            except Exception as error:
                status.update(
                    label="Unable to generate travel plan.",
                    state="error",
                    expanded=True,
                )

                st.error(
                    format_error(error)
                )


# ============================================================
# RESULTS
# ============================================================

plan = st.session_state.trip_plan
details = st.session_state.trip_details

if plan and details:
    start = date.fromisoformat(
        details["departure_date"]
    )

    end = start + timedelta(
        days=details["duration"] - 1
    )

    st.success("Your travel plan is ready!")

    st.title(
        f"{details['origin']} → {details['destination']}"
    )

    st.caption(
        f"{start.isoformat()} to {end.isoformat()} · "
        f"{details['duration']} days · "
        f"{details['travelers']} travelers · "
        f"{details['style']} · "
        f"{details['transport_mode']}"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Duration",
        f"{details['duration']} days",
    )

    m2.metric(
        "Travelers",
        details["travelers"],
    )

    m3.metric(
        "Group budget",
        money(details["budget"]),
    )

    m4.metric(
        "Recommended transport",
        plan["recommended_mode"],
    )

    tabs = st.tabs(
        [
            "Research",
            "Transport & Booking",
            "Budget Dashboard",
            "Itinerary",
            "Interactive Map",
            "AI Review & Chat",
            "Sources",
        ]
    )

    # ========================================================
    # TAB 1 - RESEARCH
    # ========================================================

    with tabs[0]:
        st.subheader("Destination Research")

        st.write(
            plan["overview"]
        )

        st.caption(
            "Search snippets are reference material, "
            "not confirmation of live prices or schedules."
        )

        if st.session_state.trip_sources:
            for source in st.session_state.trip_sources:
                st.markdown(
                    f"**[{source['title']}]({source['url']})**"
                )

                st.write(
                    source["snippet"][:400]
                )

        else:
            st.warning(
                "No research results were retrieved."
            )

    # ========================================================
    # TAB 2 - TRANSPORT & BOOKING
    # ========================================================

    with tabs[1]:
        st.subheader(
            "Transport Options Compared"
        )

        st.write(
            f"**Recommended:** "
            f"{plan['recommended_mode']}\n\n"
            f"{plan['recommendation_reason']}"
        )

        rows = []

        for option in plan["transport_options"]:
            cost = valid_number(
                option.get(
                    "group_round_trip_cost"
                )
            )

            rows.append(
                {
                    "Mode": option.get(
                        "mode",
                        "Unknown",
                    ),
                    "Estimated group round trip": (
                        money(cost)
                        if cost is not None and cost > 0
                        else "Requires verification"
                    ),
                    "Journey time": option.get(
                        "journey_time",
                        "",
                    ),
                    "Advantages": option.get(
                        "advantages",
                        "",
                    ),
                    "Disadvantages": option.get(
                        "disadvantages",
                        "",
                    ),
                    "Verify": option.get(
                        "verification",
                        "",
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Transport options are alternatives. "
            "Only the recommended transport is included "
            "in the main budget."
        )

        st.divider()

        st.subheader(
            "Own-car calculator"
        )

        if details["road_distance"] > 0:
            car = calculate_car_cost(
                details["road_distance"],
                details["car_mileage"],
                details["fuel_price"],
                details["toll_parking"],
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Round-trip distance",
                f"{car['distance']:,.0f} km",
            )

            c2.metric(
                "Fuel required",
                f"{car['fuel_litres']:.1f} L",
            )

            c3.metric(
                "Estimated total",
                money(car["total"]),
            )

        else:
            st.warning(
                "Enter a verified one-way road distance "
                "to calculate car costs."
            )

        origin_q = quote(
            details["origin"]
        )

        destination_q = quote(
            details["destination"]
        )

        st.divider()

        st.subheader(
            "Official and external booking links"
        )

        links = [
            (
                "Search trains on IRCTC",
                "https://www.irctc.co.in/nget/train-search",
            ),
            (
                "Indian Railways train schedule",
                "https://www.indianrail.gov.in/enquiry/SCHEDULE/TrainSchedule.jsp",
            ),
            (
                "MSRTC reservations",
                "https://npublic.msrtcors.com/reservation-home",
            ),
            (
                "Compare buses on redBus",
                "https://www.redbus.in/",
            ),
            (
                "Compare flights",
                "https://www.google.com/travel/flights",
            ),
            (
                "Driving directions",
                f"https://www.google.com/maps/dir/{origin_q}/{destination_q}",
            ),
        ]

        for label, url in links:
            st.markdown(
                f"- [{label}]({url})"
            )

        st.warning(
            "The app does not connect to live booking inventory. "
            "Verify schedules, fares and availability "
            "on the booking websites."
        )

    # ========================================================
    # TAB 3 - BUDGET DASHBOARD
    # ========================================================

    with tabs[2]:
        st.subheader(
            "Group Budget Dashboard"
        )

        st.caption(
            "Edit the amounts to update your budget. "
            "Blank costs mean unknown, not free."
        )

        editor_key = (
            "budget_editor_"
            + str(st.session_state.budget_editor_id)
        )

        edited = st.data_editor(
            st.session_state.budget_table,
            key=editor_key,
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            column_config={
                "Category": st.column_config.TextColumn(
                    "Expense category",
                    disabled=True,
                ),
                "Estimated group cost (₹)": (
                    st.column_config.NumberColumn(
                        "Group cost (₹)",
                        min_value=0,
                        step=100,
                        format="₹%.0f",
                    )
                ),
                "Assumptions": st.column_config.TextColumn(
                    "Assumptions",
                ),
            },
        )

        st.session_state.budget_table = edited

        costs = pd.to_numeric(
            edited["Estimated group cost (₹)"],
            errors="coerce",
        )

        known_costs = costs.dropna().clip(lower=0)

        total = float(
            known_costs.sum()
        )

        unknown_count = int(
            costs.isna().sum()
        )

        budget_value = float(
            details["budget"]
        )

        balance = budget_value - total

        per_person = total / max(
            1,
            details["travelers"],
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Known estimated costs",
            money(total),
        )

        c2.metric(
            "Group budget",
            money(budget_value),
        )

        c3.metric(
            "Budget balance",
            (
                money(abs(balance))
                if unknown_count == 0
                else "Incomplete"
            ),
        )

        c4.metric(
            "Known per-person estimate",
            money(per_person),
        )

        if unknown_count > 0:
            st.warning(
                f"Budget incomplete: {unknown_count} "
                "expense categories have unknown costs. "
                "The displayed total is only a subtotal. "
                "Enter planning estimates for blank amounts."
            )

        elif balance < 0:
            st.error(
                f"Estimated costs exceed your budget by "
                f"{money(abs(balance))}."
            )

        else:
            st.success(
                f"Current estimated expenses are "
                f"{money(balance)} below your group budget. "
                "Prices still require verification."
            )

        if total > 0:
            chart_data = edited.copy()

            chart_data[
                "Estimated group cost (₹)"
            ] = costs

            chart_data = chart_data[
                chart_data[
                    "Estimated group cost (₹)"
                ].notna()
                & (
                    chart_data[
                        "Estimated group cost (₹)"
                    ] > 0
                )
            ]

            fig = px.pie(
                chart_data,
                names="Category",
                values="Estimated group cost (₹)",
                hole=0.4,
                title="Estimated Expense Distribution",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.download_button(
            "Download budget CSV",
            data=edited.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="travel_budget.csv",
            mime="text/csv",
        )

        st.caption(
            "AI estimates are planning assumptions, "
            "not verified ticket or accommodation prices."
        )

    # ========================================================
    # TAB 4 - ITINERARY
    # ========================================================

    with tabs[3]:
        st.subheader(
            "Day-by-Day Itinerary"
        )

        for item in plan["itinerary"]:
            with st.expander(
                f"Day {item['day']} — {item['date']}",
                expanded=True,
            ):
                st.markdown(
                    f"**Morning**\n\n"
                    f"{item.get('morning', '')}"
                )

                st.markdown(
                    f"**Afternoon**\n\n"
                    f"{item.get('afternoon', '')}"
                )

                st.markdown(
                    f"**Evening**\n\n"
                    f"{item.get('evening', '')}"
                )

                if item.get("notes"):
                    st.caption(
                        item["notes"]
                    )

        st.divider()

        st.subheader(
            "Accommodation and Food"
        )

        for item in plan["accommodation_food"]:
            st.markdown(
                f"**{item.get('item', 'Suggestion')}** — "
                f"{item.get('location', '')}"
            )

            st.write(
                "Price: "
                + str(
                    item.get(
                        "price_range",
                        "Requires verification",
                    )
                )
            )

            st.write(
                item.get("notes", "")
            )

        st.divider()

        st.subheader(
            "Booking Checklist"
        )

        for item in plan["booking_checklist"]:
            st.markdown(
                f"- {item}"
            )

        st.subheader(
            "Practical Tips"
        )

        for item in plan["practical_tips"]:
            st.markdown(
                f"- {item}"
            )

        st.subheader(
            "Important Limitations"
        )

        for item in plan["limitations"]:
            st.markdown(
                f"- {item}"
            )

        st.download_button(
            "Download complete plan as JSON",
            data=json.dumps(
                plan,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            file_name="travel_plan.json",
            mime="application/json",
        )

    # ========================================================
    # TAB 5 - INTERACTIVE MAP
    # ========================================================

    with tabs[4]:
        st.subheader(
            "Destination Map"
        )

        st.write(
            "Load a map pin for your destination. "
            "This is not a route planner."
        )

        if st.button(
            "Load Destination Map"
        ):
            with st.spinner(
                "Looking up coordinates..."
            ):
                try:
                    geolocator = Nominatim(
                        user_agent="agentic_ai_travel_planner"
                    )

                    location = geolocator.geocode(
                        details["destination"],
                        timeout=10,
                    )

                    if location:
                        st.session_state.destination_map = (
                            pd.DataFrame(
                                {
                                    "latitude": [
                                        location.latitude
                                    ],
                                    "longitude": [
                                        location.longitude
                                    ],
                                }
                            )
                        )

                        st.session_state.destination_map_name = (
                            details["destination"]
                        )

                    else:
                        st.warning(
                            "Destination not found. "
                            "Try a more specific place name."
                        )

                except (
                    GeocoderUnavailable,
                    GeocoderTimedOut,
                ):
                    st.warning(
                        "Map service timed out. "
                        "Try again later."
                    )

                except Exception as error:
                    st.warning(
                        f"Unable to load map: {error}"
                    )

        if st.session_state.destination_map is not None:
            st.caption(
                st.session_state.destination_map_name
            )

            st.map(
                st.session_state.destination_map
            )

        else:
            st.info(
                "Click Load Destination Map "
                "to display the destination."
            )

    # ========================================================
    # TAB 6 - AI REVIEW & CHAT
    # ========================================================

    with tabs[5]:
        st.subheader(
            "AI Review & Travel Assistant"
        )

        # ----------------------------------------------------
        # REVIEW TRAVEL PLAN
        # ----------------------------------------------------

        if st.button(
            "Review My Travel Plan"
        ):
            with st.spinner(
                "Reviewing itinerary and budget..."
            ):
                try:
                    budget_records = (
                        st.session_state.budget_table
                        .where(
                            pd.notna(
                                st.session_state.budget_table
                            ),
                            None,
                        )
                        .to_dict(
                            orient="records"
                        )
                    )

                    review_data = {
                        "origin": details["origin"],
                        "destination": details["destination"],
                        "duration": details["duration"],
                        "travelers": details["travelers"],
                        "group_budget": details["budget"],
                        "recommended_transport": (
                            plan["recommended_mode"]
                        ),
                        "itinerary": plan["itinerary"],
                        "budget_items": budget_records,
                    }

                    review = ask_ai(
                        """
                        You are a careful travel plan reviewer.

                        Review:
                        - Unrealistic travel schedules
                        - Overloaded sightseeing days
                        - Accommodation-night assumptions
                        - Missing expenses
                        - Budget completeness
                        - Budget arithmetic

                        IMPORTANT:
                        - Null costs mean unknown expenses.
                        - Never treat unknown costs as free.
                        - Do not invent live fares.
                        - Do not invent entry fees.
                        - Do not suggest extra hotel nights
                          without considering trip duration.
                        - Distinguish estimates from facts.
                        - Keep the review under 250 words.
                        """,
                        json.dumps(
                            review_data,
                            ensure_ascii=False,
                            default=str,
                        ),
                        max_tokens=2000,
                    )

                    st.session_state.trip_review = review

                except Exception as error:
                    st.error(
                        format_error(error)
                    )

        if st.session_state.trip_review:
            st.markdown(
                "### AI Travel Plan Review"
            )

            st.markdown(
                st.session_state.trip_review
            )

        st.divider()

        # ----------------------------------------------------
        # AI CHAT ASSISTANT
        # ----------------------------------------------------

        st.subheader(
            "Ask Your AI Travel Assistant"
        )

        question = st.text_input(
            "Ask a question about your trip",
            placeholder=(
                "How can I reduce my Pune to Goa trip cost?"
            ),
            key="trip_question",
        )

        if st.button(
            "Ask AI"
        ):
            if not question.strip():
                st.warning(
                    "Please enter a question first."
                )

            else:
                with st.spinner(
                    "Generating an answer..."
                ):
                    try:
                        trip_summary = {
                            "origin": details["origin"],
                            "destination": details["destination"],
                            "departure_date": (
                                details["departure_date"]
                            ),
                            "duration": details["duration"],
                            "travelers": details["travelers"],
                            "group_budget": details["budget"],
                            "travel_style": details["style"],
                            "preferred_transport": (
                                details["transport_mode"]
                            ),
                            "recommended_transport": (
                                plan["recommended_mode"]
                            ),
                        }

                        answer = ask_ai(
                            """
                            You are a practical AI Travel Assistant.

                            Rules:
                            - Answer the specific question.
                            - Keep your answer under 200 words.
                            - Consider the user's budget.
                            - Give practical suggestions.
                            - Do not invent confirmed fares.
                            - Do not invent train schedules.
                            - Do not promise discounts.
                            - Clearly label unverified estimates.
                            - Do not repeat the full itinerary.
                            """,
                            (
                                "Trip information:\n"
                                + json.dumps(
                                    trip_summary,
                                    ensure_ascii=False,
                                )
                                + "\n\nQuestion:\n"
                                + question.strip()
                            ),
                            max_tokens=2000,
                        )

                        st.session_state.trip_chat.append(
                            {
                                "question": question.strip(),
                                "answer": answer,
                            }
                        )

                    except Exception as error:
                        st.error(
                            format_error(error)
                        )

        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        for entry in reversed(
            st.session_state.trip_chat
        ):
            with st.chat_message("user"):
                st.write(
                    entry["question"]
                )

            with st.chat_message("assistant"):
                st.markdown(
                    entry["answer"]
                )

    # ========================================================
    # TAB 7 - SOURCES
    # ========================================================

    with tabs[6]:
        st.subheader(
            "Research Sources"
        )

        if st.session_state.trip_sources:
            for index, source in enumerate(
                st.session_state.trip_sources,
                start=1,
            ):
                with st.expander(
                    f"{index}. {source['title']}"
                ):
                    st.write(
                        source["snippet"]
                    )

                    st.markdown(
                        f"[Open original source]({source['url']})"
                    )

        else:
            st.warning(
                "No research sources were retrieved."
            )

else:
    st.info(
        "Enter your trip details in the sidebar "
        "and click 'Generate My Trip' to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI estimates are not confirmed fares or reservations. "
    "Verify transport schedules, accommodation availability, "
    "prices, opening hours, and travel restrictions before booking."
)

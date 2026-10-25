import json
import math
import os
import re
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import streamlit as st
from ddgs import DDGS
from dotenv import load_dotenv
from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim
from groq import Groq

load_dotenv()
st.set_page_config(page_title="Agentic AI Travel Planner", page_icon="✈️", layout="wide")
MODEL = "openai/gpt-oss-20b"
CATEGORIES = ["Round-trip transport", "Accommodation", "Food", "Local transport", "Activities and entry", "Other expenses"]


def api_key():
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", ""))
    except Exception:
        return ""


def ai(system, user, max_tokens=4500):
    key = api_key()
    if not key:
        raise ValueError("Missing GROQ_API_KEY. Set it in .env locally or Streamlit Cloud app Secrets.")
    client = Groq(api_key=key, timeout=90, max_retries=1)
    response = client.chat.completions.create(
        model=MODEL, temperature=0.15, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if not response.choices or not (response.choices[0].message.content or "").strip():
        raise ValueError("AI returned an empty response. Try again.")
    if response.choices[0].finish_reason == "length":
        raise ValueError("AI response was cut off. Try a shorter trip or retry.")
    return response.choices[0].message.content.strip()


def safe_error(exc):
    msg = str(exc)
    if "429" in msg or "rate_limit" in msg.lower():
        return "Groq rate limit reached. Wait and retry."
    if "401" in msg or "authentication" in msg.lower():
        return "Groq authentication failed. Check the API key in Streamlit Secrets or .env."
    if "GROQ_API_KEY" in msg:
        return msg
    return "Operation failed: " + msg[:350]


def parse_json(raw):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("AI did not return valid JSON. Generate again.")
        data = json.loads(text[a:b + 1])
    if not isinstance(data, dict):
        raise ValueError("AI returned the wrong response structure.")
    return data


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(str(value).replace(",", "").replace("₹", "").strip())
        return x if math.isfinite(x) and x >= 0 else None
    except (ValueError, TypeError):
        return None


def money(value):
    x = number(value)
    return f"₹{x:,.0f}" if x is not None else "Unknown"


def car_cost(d):
    distance = number(d.get("road_distance")) or 0
    mileage = number(d.get("mileage")) or 0
    fuel_price = number(d.get("fuel_price")) or 0
    extras = number(d.get("tolls")) or 0
    if distance <= 0 or mileage <= 0:
        return None
    kms = distance * 2
    fuel = kms / mileage * fuel_price
    return {"distance": kms, "litres": kms / mileage, "fuel": fuel, "extras": extras, "total": round(fuel + extras, 2)}


@st.cache_data(ttl=1800, show_spinner=False)
def research(origin, destination):
    queries = [
        f"{destination} official tourism attractions", f"{origin} to {destination} travel train bus routes",
        f"{destination} budget hotels local transport costs", f"{destination} tourist places opening hours",
    ]
    found = {}
    try:
        with DDGS() as engine:
            for q in queries:
                try:
                    for result in engine.text(q, max_results=3):
                        url = str(result.get("href", ""))
                        if url.startswith("https://") or url.startswith("http://"):
                            found[url] = {"title": str(result.get("title", "Source")), "url": url, "snippet": str(result.get("body", ""))}
                except Exception:
                    continue
    except Exception:
        pass
    return list(found.values())[:10]


def generate_plan(d, sources):
    start = date.fromisoformat(d["departure"])
    days, travelers = d["days"], d["travelers"]
    nights = max(0, days - 1)
    search_context = "\n".join(f"{s['title']}: {s['snippet'][:180]} ({s['url']})" for s in sources[:7])
    car = car_cost(d)
    instructions = """You are a careful Indian travel planner. Return ONLY one concise valid JSON object, without Markdown.
Research snippets are unverified. Never claim live availability, confirmed fares, precise train schedules or verified prices.
Return keys: overview (string), transport_options (array of objects with mode, group_round_trip_cost nullable number, journey_time, advantages, disadvantages, verification), recommended_mode (one of Train, Bus, Flight, Own car), recommendation_reason (string), itinerary (array of objects with day integer, date string, morning, afternoon, evening, notes), budget_items (exactly six objects with category, unit_rate nullable number, quantity nullable number, unit string, group_cost nullable number, assumptions string), accommodation_food (array of objects with item, location, price_range, notes), booking_checklist (string array), practical_tips (string array), limitations (string array).
Budget categories EXACTLY: Round-trip transport, Accommodation, Food, Local transport, Activities and entry, Other expenses.
Rates must be estimates, never presented as verified. group_cost = unit_rate * quantity. Unknown costs use null, not zero.
Transport: for train/bus/flight, unit_rate is ONE-WAY fare PER traveler and quantity is 2*travelers, unit 'one-way passenger tickets'. For own car, use supplied car cost only when known; otherwise null.
Accommodation: unit_rate is nightly room cost FOR ENTIRE GROUP; quantity=hotel nights; unit 'nights'.
Food: unit_rate is daily expense FOR ENTIRE GROUP; quantity=days; unit 'days'.
Local transport: unit_rate is daily expense FOR ENTIRE GROUP; quantity=days; unit 'days'.
Activities: unit_rate is estimated combined activities per traveler; quantity=travelers; unit 'travelers'.
Other: unit_rate is group miscellaneous allowance; quantity=1; unit 'group allowance'.
Itinerary must have EXACTLY requested number of days. Avoid packing far-apart attractions into one morning. The last day must allow for return travel. Do not assume precise departure or arrival times. Do not force budget to fit user's limit. Keep output compact."""
    user = json.dumps({
        "origin": d["origin"], "destination": d["destination"], "start_date": d["departure"],
        "end_date": (start + timedelta(days=days - 1)).isoformat(), "days": days, "hotel_nights": nights,
        "travelers": travelers, "group_budget_inr": d["budget"], "style": d["style"],
        "transport_preference": d["mode"], "train_class": d["train_class"], "train_quota": d["train_quota"],
        "bus_type": d["bus_type"], "flight_class": d["flight_class"],
        "car_calculation": car, "sources": search_context,
    }, ensure_ascii=False)
    plan = parse_json(ai(instructions, user, max_tokens=6500))
    for field in ["overview", "transport_options", "recommended_mode", "itinerary", "budget_items"]:
        if field not in plan:
            raise ValueError(f"AI response missing {field}; generate again.")
    if not isinstance(plan["itinerary"], list) or len(plan["itinerary"]) != days:
        raise ValueError("AI returned the wrong number of itinerary days; generate again.")
    for i, entry in enumerate(plan["itinerary"]):
        if not isinstance(entry, dict):
            raise ValueError("Invalid itinerary format; generate again.")
        entry["day"] = i + 1
        entry["date"] = (start + timedelta(days=i)).isoformat()
    for field in ["transport_options", "budget_items"]:
        if not isinstance(plan[field], list):
            raise ValueError(f"Invalid {field} format; generate again.")
    for field in ["accommodation_food", "booking_checklist", "practical_tips", "limitations"]:
        if not isinstance(plan.get(field), list):
            plan[field] = []
    return plan


def budget_from_plan(plan, d):
    items = {str(x.get("category", "")).casefold(): x for x in plan.get("budget_items", []) if isinstance(x, dict)}
    days, nights, travelers = d["days"], max(0, d["days"] - 1), d["travelers"]
    recommended = str(plan.get("recommended_mode", "")).casefold().strip()
    result = []
    for category in CATEGORIES:
        entry = items.get(category.casefold(), {})
        rate = number(entry.get("unit_rate"))
        stated = number(entry.get("group_cost"))
        note = str(entry.get("assumptions") or "")
        unit = str(entry.get("unit") or "")
        expected = {
            "Round-trip transport": travelers * 2,
            "Accommodation": nights,
            "Food": days,
            "Local transport": days,
            "Activities and entry": travelers,
            "Other expenses": 1,
        }[category]
        expected_unit = {
            "Round-trip transport": "one-way passenger tickets",
            "Accommodation": "nights", "Food": "days", "Local transport": "days",
            "Activities and entry": "travelers", "Other expenses": "group allowance",
        }[category]
        if category == "Accommodation" and nights == 0:
            cost, calc = 0.0, "No hotel nights for a one-day trip."
        elif category == "Round-trip transport" and recommended in {"car", "own car"}:
            car = car_cost(d)
            cost = car["total"] if car else None
            calc = (f"Own car: {car['distance']:.0f} km round trip, {car['litres']:.1f} L fuel, "
                    f"{money(car['fuel'])} fuel + {money(car['extras'])} toll/parking.") if car else "Unknown car cost: enter one-way road distance and regenerate."
        elif rate is not None and rate > 0:
            cost = round(rate * expected, 2)
            calc = f"{money(rate)} × {expected} {expected_unit} = {money(cost)}."
            if stated is not None and abs(stated - cost) > 1:
                calc += " AI total differed; Python calculation used."
        else:
            # No fallback to inconsistent AI group totals.
            cost = None
            calc = "Rate unavailable; enter an estimated group cost."
        # Do not repeat potentially contradictory numeric prose from the model.
        result.append({"Category": category, "Estimated group cost (₹)": cost, "Calculation / basis": calc,
                       "AI notes (unverified)": note})
    return pd.DataFrame(result)


def budget_context(d, plan, table):
    rows = []
    known = 0.0
    missing = []
    for _, item in table.iterrows():
        cat = str(item["Category"])
        value = number(item["Estimated group cost (₹)"])
        if value is None:
            missing.append(cat)
        else:
            known += value
        rows.append({"category": cat, "group_cost_inr": value})
    return {
        "origin": d["origin"], "destination": d["destination"], "departure_date": d["departure"],
        "trip_days": d["days"], "hotel_nights": max(0, d["days"] - 1), "travelers": d["travelers"],
        "group_budget_inr": d["budget"], "travel_style": d["style"], "recommended_transport": plan["recommended_mode"],
        "budget_items": rows, "known_group_total_inr": round(known, 2),
        "known_per_person_inr": round(known / d["travelers"], 2),
        "remaining_after_known_costs_inr": round(d["budget"] - known, 2),
        "budget_complete": not missing, "unknown_categories": missing,
    }


for key, default in {"plan": None, "details": None, "sources": [], "budget_table": None,
                     "review": "", "chat": [], "map_data": None, "generation": 0}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("✈️ Agentic AI Travel Planner")
st.caption("Research • Transport • Budget • Itinerary • Map • AI Review & Chat")

with st.sidebar:
    st.header("Plan Your Trip")
    with st.form("trip_form"):
        origin = st.text_input("Starting city", "Pune")
        destination = st.text_input("Destination", "Goa")
        departure = st.date_input("Departure date", value=date.today(), min_value=date.today())
        days = st.number_input("Trip duration (days)", 1, 14, 3)
        travelers = st.number_input("Number of travelers", 1, 30, 2)
        budget = st.number_input("Total group budget (₹)", min_value=0, value=15000, step=1000)
        style = st.selectbox("Travel style", ["Budget", "Balanced", "Luxury", "Adventure", "Family", "Food and Culture"])
        mode = st.selectbox("Preferred transport", ["Compare all options", "Own car", "Train", "Bus", "Flight"])
        st.markdown("#### 🚗 Own-car details")
        mileage = st.number_input("Car mileage (km/litre)", min_value=1.0, value=15.0)
        fuel_price = st.number_input("Fuel price (₹/litre)", min_value=1.0, value=105.0)
        road_distance = st.number_input("One-way road distance (km)", min_value=0.0, value=0.0, step=10.0)
        tolls = st.number_input("Round-trip toll and parking (₹)", min_value=0.0, value=0.0, step=100.0)
        st.markdown("#### 🚆 Train preferences")
        train_class = st.selectbox("Preferred train class", ["Any class", "SL", "3A", "2A", "1A", "CC", "EC"])
        train_quota = st.selectbox("Train quota", ["General", "Tatkal", "Ladies", "Senior Citizen"])
        st.markdown("#### 🚌 Bus preferences")
        bus_type = st.selectbox("Bus type", ["Any bus", "AC Seater", "Non-AC Seater", "AC Sleeper", "Non-AC Sleeper", "Volvo / Premium"])
        st.markdown("#### ✈️ Flight preferences")
        flight_class = st.selectbox("Flight cabin", ["Economy", "Premium Economy", "Business"])
        submit = st.form_submit_button("Generate My Trip", type="primary", use_container_width=True)
    st.caption(f"AI model: {MODEL}")

if submit:
    if not origin.strip() or not destination.strip() or origin.strip().casefold() == destination.strip().casefold():
        st.error("Enter different, nonempty starting and destination cities.")
    else:
        d = {"origin": origin.strip(), "destination": destination.strip(), "departure": departure.isoformat(),
             "days": int(days), "travelers": int(travelers), "budget": float(budget), "style": style, "mode": mode,
             "mileage": float(mileage), "fuel_price": float(fuel_price), "road_distance": float(road_distance),
             "tolls": float(tolls), "train_class": train_class, "train_quota": train_quota,
             "bus_type": bus_type, "flight_class": flight_class}
        st.session_state.plan = None
        st.session_state.review = ""
        st.session_state.chat = []
        st.session_state.map_data = None
        with st.spinner("Researching and generating your travel plan..."):
            try:
                sources = research(d["origin"], d["destination"])
                plan = generate_plan(d, sources)
                st.session_state.details = d
                st.session_state.sources = sources
                st.session_state.plan = plan
                st.session_state.budget_table = budget_from_plan(plan, d)
                st.session_state.generation += 1
                st.success("Your travel plan is ready!")
            except Exception as exc:
                st.error(safe_error(exc))

plan, d = st.session_state.plan, st.session_state.details
if plan is None or d is None:
    st.info("Enter your details in the sidebar and click Generate My Trip.")
    st.stop()

end = date.fromisoformat(d["departure"]) + timedelta(days=d["days"] - 1)
st.subheader(f"{d['origin']} → {d['destination']}")
st.caption(f"{d['departure']} to {end.isoformat()} · {d['days']} days · {max(0, d['days']-1)} hotel nights · {d['travelers']} travelers")
a, b, c, e = st.columns(4)
a.metric("Duration", f"{d['days']} days")
b.metric("Travelers", d["travelers"])
c.metric("Group budget", money(d["budget"]))
e.metric("Recommended transport", str(plan.get("recommended_mode", "Unknown")))

tabs = st.tabs(["Research", "Transport & Booking", "Budget Dashboard", "Itinerary", "Interactive Map", "AI Review & Chat", "Sources"])

with tabs[0]:
    st.subheader("Destination Research")
    st.write(str(plan.get("overview", "")))
    st.caption("Search snippets are not verification of current fares, hours or availability.")
    if st.session_state.sources:
        for source in st.session_state.sources:
            st.markdown(f"**[{source['title']}]({source['url']})**")
            st.write(source["snippet"][:450])
    else:
        st.warning("No online research results were available. AI suggestions require independent verification.")

with tabs[1]:
    st.subheader("Transport Comparison")
    st.write(f"**Recommended:** {plan.get('recommended_mode', 'Unknown')}")
    st.write(str(plan.get("recommendation_reason", "")))
    options = []
    for item in plan.get("transport_options", []):
        if isinstance(item, dict):
            cost = number(item.get("group_round_trip_cost"))
            options.append({"Mode": item.get("mode", ""), "Group round-trip estimate": money(cost) if cost and cost > 0 else "Unverified / unknown",
                            "Journey time": item.get("journey_time", ""), "Advantages": item.get("advantages", ""),
                            "Disadvantages": item.get("disadvantages", ""), "Verification": item.get("verification", "")})
    if options:
        st.dataframe(pd.DataFrame(options), hide_index=True, use_container_width=True)
    st.info("Alternatives are compared, not added together. Only the recommended mode belongs in the main budget.")
    st.subheader("Own-car calculator")
    car = car_cost(d)
    if car:
        x, y, z = st.columns(3)
        x.metric("Round-trip distance", f"{car['distance']:,.0f} km")
        y.metric("Fuel required", f"{car['litres']:.1f} L")
        z.metric("Estimated fuel + toll/parking", money(car["total"]))
        st.caption("Excludes vehicle maintenance and unexpected expenses.")
    else:
        st.warning("Enter a verified one-way distance and regenerate to calculate car fuel costs.")
    st.subheader("External booking and route links")
    booking = [
        ("IRCTC train booking", "https://www.irctc.co.in/nget/train-search"),
        ("Indian Railways", "https://www.indianrail.gov.in/"),
        ("MSRTC", "https://msrtc.maharashtra.gov.in/"),
        ("redBus", "https://www.redbus.in/"),
        ("Google Flights", "https://www.google.com/travel/flights"),
        ("Google Maps directions", f"https://www.google.com/maps/dir/{quote(d['origin'])}/{quote(d['destination'])}"),
    ]
    for label, url in booking:
        st.markdown(f"- [{label}]({url})")
    st.warning("No live bookings or prices are confirmed by this app.")

with tabs[2]:
    st.subheader("Group Budget Dashboard")
    st.caption("Python calculates rate × quantity. Edit costs below; blank means unknown, not free. Edited amounts are used by AI Review and Chat.")
    table = st.data_editor(
        st.session_state.budget_table,
        key=f"budget_editor_{st.session_state.generation}", hide_index=True, num_rows="fixed", use_container_width=True,
        column_config={
            "Category": st.column_config.TextColumn("Expense category", disabled=True),
            "Estimated group cost (₹)": st.column_config.NumberColumn("Group cost (₹)", min_value=0, step=100, format="₹%.0f"),
            "Calculation / basis": st.column_config.TextColumn("Calculation / basis", disabled=True),
            "AI notes (unverified)": st.column_config.TextColumn("AI notes (unverified)", disabled=True),
        },
    )
    st.session_state.budget_table = table
    context = budget_context(d, plan, table)
    a, b, c, e = st.columns(4)
    a.metric("Known estimated group costs", money(context["known_group_total_inr"]))
    b.metric("Group budget", money(context["group_budget_inr"]))
    c.metric("Remaining budget", money(context["remaining_after_known_costs_inr"]) if context["budget_complete"] else "Incomplete")
    e.metric("Known cost per person", money(context["known_per_person_inr"]))
    if not context["budget_complete"]:
        st.warning("Budget incomplete. Unknown: " + ", ".join(context["unknown_categories"]))
    elif context["remaining_after_known_costs_inr"] < 0:
        st.error(f"Estimated group costs exceed the budget by {money(abs(context['remaining_after_known_costs_inr']))}.")
    else:
        st.success(f"Estimated group costs are {money(context['remaining_after_known_costs_inr'])} below budget. Prices remain unverified.")
    chart_df = table.copy()
    chart_df["Estimated group cost (₹)"] = pd.to_numeric(chart_df["Estimated group cost (₹)"], errors="coerce")
    chart_df = chart_df[chart_df["Estimated group cost (₹)"] > 0]
    if not chart_df.empty:
        fig = px.pie(chart_df, names="Category", values="Estimated group cost (₹)", hole=0.4, title="Estimated Expense Distribution")
        st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download budget CSV", table.to_csv(index=False).encode("utf-8"), "travel_budget.csv", "text/csv")

with tabs[3]:
    st.subheader("Day-by-Day Itinerary")
    for entry in plan["itinerary"]:
        with st.expander(f"Day {entry['day']} — {entry['date']}", expanded=True):
            for period in ["morning", "afternoon", "evening"]:
                st.markdown(f"**{period.title()}**")
                st.write(str(entry.get(period, "")))
            if entry.get("notes"):
                st.caption(str(entry["notes"]))
    st.subheader("Accommodation & Food Suggestions")
    for item in plan.get("accommodation_food", []):
        if isinstance(item, dict):
            st.markdown(f"**{item.get('item', 'Suggestion')}** — {item.get('location', '')}")
            st.write(f"Estimate: {item.get('price_range', 'Unknown')}. {item.get('notes', '')}")
    for label, field in [("Booking Checklist", "booking_checklist"), ("Practical Tips", "practical_tips"), ("Limitations", "limitations")]:
        st.subheader(label)
        for value in plan.get(field, []):
            st.markdown(f"- {value}")
    st.download_button("Download itinerary JSON", json.dumps(plan, indent=2, ensure_ascii=False).encode("utf-8"), "travel_plan.json", "application/json")

with tabs[4]:
    st.subheader("Destination Map")
    st.caption("A destination pin, not a verified itinerary route.")
    if st.button("Load Destination Map"):
        try:
            with st.spinner("Finding destination coordinates..."):
                result = Nominatim(user_agent="agentic_ai_travel_planner_project_2026").geocode(d["destination"], timeout=12)
            if result:
                st.session_state.map_data = pd.DataFrame({"lat": [result.latitude], "lon": [result.longitude]})
            else:
                st.warning("Destination could not be located. Try a more specific city name.")
        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError):
            st.warning("Map service is temporarily unavailable. Try again later.")
        except Exception:
            st.warning("Unable to load destination map at this time.")
    if st.session_state.map_data is not None:
        st.map(st.session_state.map_data)

with tabs[5]:
    st.subheader("AI Review & Travel Assistant")
    # Recompute on every rerun: AI always receives the latest edited budget.
    context = budget_context(d, plan, st.session_state.budget_table)
    a, b, c = st.columns(3)
    a.metric("Known group costs", money(context["known_group_total_inr"]))
    b.metric("Group budget", money(context["group_budget_inr"]))
    c.metric("Remaining", money(context["remaining_after_known_costs_inr"]) if context["budget_complete"] else "Incomplete")
    st.caption(f"{context['trip_days']} days · {context['hotel_nights']} hotel nights · {context['travelers']} travelers")
    if not context["budget_complete"]:
        st.warning("Some budget categories are unknown. Remaining funds are not a confirmed surplus.")
    review_system = """You are a careful travel itinerary and budget reviewer. Treat the supplied JSON as authoritative for trip days, hotel nights, group costs, and remaining funds. Never confuse 3 days with 3 hotel nights. All budget costs are GROUP costs. Evaluate the actual itinerary for geographical feasibility, missing costs, and unrealistic scheduling. For proposed added costs, show arithmetic and explicitly say if additions exceed the remaining funds. Never claim unverified prices are confirmed. If budget_complete=false, remaining funds are provisional. Avoid asserting that suggestions fit unless the totals do. Keep under 250 words. Use INR."""
    if st.button("Review My Travel Plan"):
        with st.spinner("Reviewing plan..."):
            try:
                payload = {**context, "itinerary": plan["itinerary"]}
                st.session_state.review = ai(review_system, json.dumps(payload, ensure_ascii=False), max_tokens=2000)
            except Exception as exc:
                st.error(safe_error(exc))
    if st.session_state.review:
        st.markdown("### AI Travel Plan Review")
        st.markdown(st.session_state.review)
    st.divider()
    st.subheader("Ask Your AI Travel Assistant")
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Question", placeholder="How can I reduce my trip cost while keeping 2 hotel nights?")
        ask = st.form_submit_button("Ask AI")
    if ask:
        if not question.strip():
            st.warning("Enter a question.")
        else:
            chat_system = """You are a practical travel budgeting assistant. Treat the supplied JSON as authoritative. trip_days is NOT hotel_nights. All group_cost_inr values are totals for the group, not per person. Start from the current six budget category amounts, including the user's edits. If suggesting a revised budget, show all six categories and sum correctly; distinguish savings from extra expenses. If unknown categories exist, never call the budget complete. Never invent confirmed train fares, live schedules, guaranteed discounts, or hotel availability. Do not say that a proposed budget fits if its sum exceeds the available budget. Use INR and keep under 200 words."""
            with st.spinner("Thinking..."):
                try:
                    answer = ai(chat_system, json.dumps({**context, "question": question.strip()}, ensure_ascii=False), max_tokens=2000)
                    st.session_state.chat.append({"question": question.strip(), "answer": answer})
                except Exception as exc:
                    st.error(safe_error(exc))
    for entry in reversed(st.session_state.chat):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

with tabs[6]:
    st.subheader("Research Sources")
    if not st.session_state.sources:
        st.info("No research sources retrieved. Independently verify prices and opening hours.")
    for i, source in enumerate(st.session_state.sources, 1):
        with st.expander(f"{i}. {source['title']}"):
            st.write(source["snippet"])
            st.markdown(f"[Open original source]({source['url']})")

st.divider()
st.caption("AI-generated travel suggestions are unverified planning estimates, not confirmed bookings, fares, schedules, or availability. Verify before paying.")

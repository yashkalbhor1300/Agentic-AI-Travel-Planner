
import re
from datetime import date
from dotenv import load_dotenv
from ddgs import DDGS
from agno.agent import Agent
from agno.models.groq import Groq

load_dotenv()

MODEL_ID = "openai/gpt-oss-120b"


# ==================================================
# 1. SEARCH FOR TRAVEL INFORMATION
# ==================================================

def search_travel_info(destination, origin, travel_dates):
    query = (
        f"{destination} official tourism attractions "
        f"transport accommodation prices {travel_dates}"
    )

    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            return "No search results found. Prices are unverified."

        lines = []

        for item in results:
            lines.append(
                f"Title: {item.get('title', 'N/A')}\n"
                f"URL: {item.get('href', 'N/A')}\n"
                f"Summary: {item.get('body', 'N/A')}"
            )

        return "\n\n".join(lines)

    except Exception as error:
        print(f"Search unavailable: {error}")
        return (
            "Live search failed. No prices, availability, or "
            "travel information can be treated as verified."
        )


# ==================================================
# 2. CREATE AN AI AGENT
# ==================================================

def create_agent(name, instructions):
    return Agent(
        name=name,
        model=Groq(id=MODEL_ID),
        instructions=instructions,
        markdown=True,
    )


# ==================================================
# 3. PYTHON BUDGET CALCULATOR
# ==================================================

def calculate_budget(costs, total_budget):
    """
    Calculate totals from numeric costs supplied by the user
    or verified from external sources.

    Unknown costs must be None, not zero.
    """

    known_total = sum(
        value
        for value in costs.values()
        if value is not None
    )

    unknown_items = [
        item
        for item, value in costs.items()
        if value is None
    ]

    print("\n" + "=" * 55)
    print("PYTHON BUDGET CALCULATION")
    print("=" * 55)

    for item, value in costs.items():
        if value is None:
            print(f"{item}: Not verified")
        else:
            print(f"{item}: INR {value:,.2f}")

    print("-" * 55)
    print(f"Known costs subtotal: INR {known_total:,.2f}")
    print(f"Total budget: INR {total_budget:,.2f}")

    remaining = total_budget - known_total

    if remaining >= 0:
        print(f"Budget remaining before unknown costs: INR {remaining:,.2f}")
    else:
        print(f"Known costs exceed budget by INR {abs(remaining):,.2f}")

    if unknown_items:
        print("\nBudget status: UNCONFIRMED")
        print("These costs still need verification:")
        for item in unknown_items:
            print(f"- {item}")
        print(
            "The known-cost subtotal is not the complete trip cost."
        )
    elif known_total <= total_budget:
        print("\nBudget status: WITHIN BUDGET")
        print("All entered cost categories have numeric values.")
        print(
            "This result is reliable only if the input prices are "
            "accurate and all necessary expenses are included."
        )
    else:
        print("\nBudget status: OVER BUDGET")

    return {
        "known_total": known_total,
        "remaining_before_unknown_costs": remaining,
        "unknown_items": unknown_items,
        "all_costs_known": len(unknown_items) == 0,
        "within_budget": (
            len(unknown_items) == 0
            and known_total <= total_budget
        ),
    }


# ==================================================
# 4. VALIDATE AND DISPLAY TRIP REQUIREMENTS
# ==================================================

def validate_trip(
    destination,
    origin,
    budget,
    days,
    travelers,
    travel_style,
    travel_dates,
):
    nights = max(0, days - 1)

    print("\n" + "=" * 55)
    print("TRIP REQUIREMENTS")
    print("=" * 55)
    print(f"Destination: {destination}")
    print(f"Starting city: {origin}")
    print(f"Total trip budget: INR {budget:,}")
    print(f"Duration: {days} day(s)")
    print(f"Travelers: {travelers}")
    print(f"Travel style: {travel_style}")
    print(f"Travel dates: {travel_dates}")
    print(f"Potential overnight stays: {nights}")
    print(
        "Note: Actual nights depend on arrival and departure times."
    )

    return nights


# ==================================================
# 5. MAIN MULTI-AGENT WORKFLOW
# ==================================================

def run_trip_planner(
    destination,
    origin,
    budget,
    days,
    travelers,
    travel_style,
    travel_dates,
):
    nights = validate_trip(
        destination,
        origin,
        budget,
        days,
        travelers,
        travel_style,
        travel_dates,
    )

    requirements = f"""
USER'S TRIP REQUIREMENTS

Destination: {destination}
Starting city: {origin}
Total budget for all travelers combined: INR {budget}
Duration: exactly {days} days
Number of travelers: {travelers}
Travel style: {travel_style}
Travel dates: {travel_dates}
Potential overnight stays: {nights}

IMPORTANT RULES FOR ALL AGENTS

1. Preserve the destination, origin, total budget, duration,
   traveler count, travel style, and dates.
2. The budget is for the entire group, not each person.
3. Calculate per-person amounts only as a separate breakdown.
4. Never invent prices, search results, sources, or bookings.
5. A price is verified only when the supplied search results
   explicitly support that price for the relevant item.
6. If a price is missing, label it "Not verified".
7. Label any illustrative estimate clearly as an estimate.
8. Do not treat an unknown cost as zero.
9. Include travel to the destination and return travel.
10. Include accommodation, food, local transport, activities,
    entry fees, and reasonable incidental expenses when applicable.
11. Do not claim that the trip fits the budget if required costs
    are unknown.
12. Never claim a booking is confirmed.
13. Keep the itinerary to exactly {days} days.
14. Clearly identify assumptions and missing information.
"""

    # Search for information once and share the same results
    # with each agent.
    print("\nSearching for travel information...")

    research = search_travel_info(
        destination,
        origin,
        travel_dates,
    )

    # ----------------------------------------------
    # AGENT 1: TRAVEL RESEARCH
    # ----------------------------------------------

    research_agent = create_agent(
        "Travel Research Agent",
        """
You are a careful travel research analyst.

Use only the supplied search results as evidence.

Tasks:
- Identify useful attractions and travel information.
- Identify relevant sources and URLs.
- Extract prices only when the result explicitly gives them.
- Separate verified information from unknown information.
- Do not invent prices, dates, opening hours, or availability.
- Do not claim to have opened a webpage if only a search snippet
  was provided.
- Explain which facts require confirmation.

Return:
1. Destination Overview
2. Search Findings
3. Explicitly Supported Prices
4. Missing Information
5. Sources
""",
    )

    print("\n" + "=" * 55)
    print("1. TRAVEL RESEARCH AGENT")
    print("=" * 55)

    research_result = research_agent.run(
        requirements
        + "\n\nSUPPLIED SEARCH RESULTS:\n"
        + research
    )

    research_text = (
        research_result.content or str(research_result)
    )

    print(research_text)

    # ----------------------------------------------
    # AGENT 2: BUDGET PLANNING
    # ----------------------------------------------

    budget_agent = create_agent(
        "Budget Planning Agent",
        """
You are a strict travel budget analyst.

Prepare a group budget using the supplied requirements
and research findings.

Rules:
- The entered budget is the total for all travelers combined.
- Do not invent verified prices.
- Only mark a price verified if the supplied evidence explicitly
  supports the price and its relevant item.
- Distinguish verified prices, unverified estimates, and unknowns.
- Include round-trip travel from the stated origin.
- Account for the number of travelers when discussing group costs.
- Clearly identify whether a quoted cost is per person or for
  the entire group.
- Do not treat unknown costs as zero.
- Do not claim the trip is affordable while required costs
  remain unknown.
- Show the arithmetic for any proposed totals.
- If the budget is insufficient, explain why honestly.

Return:
1. Total Budget
2. Itemized Cost Table
3. Verified Prices and Sources
4. Unverified Estimates
5. Unknown Expenses
6. Group and Per-Person Breakdown
7. Budget Feasibility
8. Information Still Required
""",
    )

    print("\n" + "=" * 55)
    print("2. BUDGET PLANNING AGENT")
    print("=" * 55)

    budget_result = budget_agent.run(
        requirements
        + "\n\nRESEARCH FINDINGS:\n"
        + research_text
    )

    budget_text = (
        budget_result.content or str(budget_result)
    )

    print(budget_text)

    # ----------------------------------------------
    # AGENT 3: ITINERARY PLANNING
    # ----------------------------------------------

    itinerary_agent = create_agent(
        "Itinerary Planning Agent",
        """
You are a practical travel itinerary planner.

Create a plan for exactly the requested number of days.

Rules:
- Use the exact destination, dates, group size, and travel style.
- Use the supplied research and budget assessment.
- Do not invent verified prices or booking confirmations.
- Label unknown costs as "Not verified".
- Prefer free activities when appropriate for the budget.
- Include realistic travel time between activities where known.
- Do not assume unknown expenses are free.
- Do not claim that the itinerary fits the budget if costs
  remain unknown.
- Do not change the requested trip duration.
- Explain assumptions and limitations.

Return:
1. Trip Summary
2. Day-by-Day Itinerary
3. Expected Expenses and Their Verification Status
4. Budget Feasibility
5. Booking Checklist
""",
    )

    print("\n" + "=" * 55)
    print("3. ITINERARY PLANNING AGENT")
    print("=" * 55)

    itinerary_result = itinerary_agent.run(
        requirements
        + "\n\nRESEARCH FINDINGS:\n"
        + research_text
        + "\n\nBUDGET ASSESSMENT:\n"
        + budget_text
    )

    itinerary_text = (
        itinerary_result.content or str(itinerary_result)
    )

    print(itinerary_text)

    # ----------------------------------------------
    # PYTHON ARITHMETIC CHECK
    # ----------------------------------------------

    print("\n" + "=" * 55)
    print("BUDGET ARITHMETIC CHECK")
    print("=" * 55)

    print(f"User's total group budget: INR {budget:,}")

    # This detects simple expressions such as:
    # INR 400 + INR 300 = INR 700
    # It checks arithmetic only, not price authenticity.
    arithmetic_pattern = re.compile(
        r"INR\s*([\d,]+(?:\.\d+)?)\s*\+\s*"
        r"INR\s*([\d,]+(?:\.\d+)?)\s*=\s*"
        r"INR\s*([\d,]+(?:\.\d+)?)",
        re.IGNORECASE,
    )

    matches = arithmetic_pattern.findall(budget_text)

    if matches:
        for first, second, stated_total in matches:
            first_value = float(first.replace(",", ""))
            second_value = float(second.replace(",", ""))
            stated_value = float(stated_total.replace(",", ""))

            calculated_total = first_value + second_value

            if abs(calculated_total - stated_value) < 0.01:
                print(
                    f"PASS: {first_value:,.2f} + "
                    f"{second_value:,.2f} = "
                    f"{calculated_total:,.2f}"
                )
            else:
                print(
                    f"WARNING: {first_value:,.2f} + "
                    f"{second_value:,.2f} = "
                    f"{calculated_total:,.2f}, not "
                    f"{stated_value:,.2f}"
                )
    else:
        print(
            "No simple addition expression was found to check. "
            "The budget agent's complete total has not been "
            "independently verified."
        )

    print(
        "\nReminder: checking arithmetic does not verify actual "
        "travel prices. Confirm prices with the relevant providers."
    )

    # ----------------------------------------------
    # EXPLICIT PYTHON BUDGET VALIDATION
    # ----------------------------------------------

    print("\n" + "=" * 55)
    print("ENTER VERIFIED COSTS FOR A FINAL BUDGET CHECK")
    print("=" * 55)
    print(
        "Enter a numeric amount only when you have a price to use. "
        "Press Enter when a cost is unknown."
    )

    cost_categories = [
        "Round-trip travel for the whole group",
        "Accommodation for the whole group",
        "Food for the whole group",
        "Local transport for the whole group",
        "Activities and entry fees for the whole group",
        "Other expenses for the whole group",
    ]

    costs = {}

    for category in cost_categories:
        while True:
            raw = input(f"{category} (INR, or Enter if unknown): ")
            raw = raw.replace(",", "").strip()

            if raw == "":
                costs[category] = None
                break

            try:
                amount = float(raw)

                if amount < 0:
                    print("Cost cannot be negative. Try again.")
                    continue

                costs[category] = amount
                break

            except ValueError:
                print("Enter a valid number or press Enter if unknown.")

    result = calculate_budget(costs, budget)

    if result["within_budget"]:
        print(
            "\nThe entered costs are within the budget. "
            "This assumes the amounts are accurate and complete."
        )
    else:
        print(
            "\nThe trip's complete affordability has not been "
            "confirmed. Review unknown costs or reduce expenses."
        )


# ==================================================
# 6. COLLECT USER INPUT
# ==================================================

if __name__ == "__main__":
    destination = input("Enter destination: ").strip()
    origin = input("Enter starting city: ").strip()

    budget_input = input(
        "Enter total group budget in INR: "
    ).replace(",", "").strip()

    days_input = input("Enter trip duration in days: ").strip()

    travelers_input = input(
        "Enter number of travelers: "
    ).strip()

    travel_style = (
        input("Enter travel style: ").strip() or "Budget"
    )

    travel_dates = (
        input("Enter travel dates (e.g. 10-12 Nov 2026): ").strip()
        or "Not specified"
    )

    if (
        not destination
        or not origin
        or not budget_input.isdigit()
        or not days_input.isdigit()
        or not travelers_input.isdigit()
        or int(budget_input) <= 0
        or int(days_input) <= 0
        or int(travelers_input) <= 0
    ):
        print(
            "\nInvalid input. Provide a destination, starting city, "
            "and positive whole numbers for budget, days, and travelers."
        )

    else:
        try:
            run_trip_planner(
                destination=destination,
                origin=origin,
                budget=int(budget_input),
                days=int(days_input),
                travelers=int(travelers_input),
                travel_style=travel_style,
                travel_dates=travel_dates,
            )

        except Exception as error:
            print(f"\nThe trip planner encountered an error: {error}")

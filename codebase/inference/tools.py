import json

# -----------------------------
# TOOL 1: SearchKB
# -----------------------------
def SearchKB(query, retrieve_fn):
    results = retrieve_fn(query)
    return {
        "tool": "SearchKB",
        "results": results
    }

# -----------------------------
# TOOL 2: GetPolicy
# -----------------------------
def GetPolicy(section_id):
    # dummy policy (you can improve later)
    policies = {
        "loan_policy": "Loans must be repaid with interest. Eligibility depends on financial need.",
        "social_security": "Eligibility depends on earned credits through work."
    }

    return {
        "tool": "GetPolicy",
        "policy": policies.get(section_id, "Policy not found")
    }

# -----------------------------
# TOOL 3: CreateTicket
# -----------------------------
def CreateTicket(summary, category="general", severity="medium"):
    ticket_id = f"TICKET-{hash(summary) % 10000}"

    return {
        "tool": "CreateTicket",
        "ticket_id": ticket_id,
        "status": "created",
        "category": category,
        "severity": severity
    }
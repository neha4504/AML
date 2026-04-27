# Maps your ML features to plain English AML concepts
SHAP_GLOSSARY = {
    "Payment Format_target_enc": "Payment was made via ACH, a channel with historically high fraud rates in our system.",
    "Receiving Currency_target_enc": "Transaction involved a high-risk receiving currency.",
    "Payment Currency_target_enc": "Transaction involved a high-risk payment currency.",
    "anomaly_score": "Transaction behavior is statistically anomalous compared to this account's normal baseline.",
    "txns_in_directed_pair": "Abnormally high volume of transactions with this specific counterparty, suggesting a coordinated loop.",
    "is_toxic_corridor": "Funds moved through a known high-risk banking corridor.",
    "burst_score_1h": "Sudden burst of rapid transactions within a single hour, indicative of automated bot activity (Smurfing).",
    "txn_in_hour": "Unusually high number of transactions processed within a single hour.",
    "amount_vs_baseline_ratio": "Transaction amount is significantly higher than this account's historical average.",
    "counterparty_diversity_28d": "Account has been interacting with an unusually high number of distinct counterparties.",
    "flag_heavy_structuring": "Triggered the heavy structuring rule based on amount patterns.",
    "From Bank_freq_enc": "Originating bank has a very low transaction frequency, potentially indicating a shell or newly created institution.",
    "To Bank_freq_enc": "Receiving bank has a very low transaction frequency.",
    "burst_score_1h": "Sudden burst of rapid transactions within a single hour, indicative of automated bot activity or Smurfing.",
    "hour_sin": "Circular time encoding representing the time of day (part 1).",
    "hour_cos": "Circular time encoding representing the time of day (part 2). On its own, this just denotes when the transaction occurred, not necessarily suspicious behavior.",
    "day_of_week_sin": "Circular time encoding representing the day of the week (part 1).",
    "day_of_week_cos": "Circular time encoding representing the day of the week (part 2).",
}

def translate_shap_for_llm(shap_explanation_list: list, raw_feat: dict) -> str:
    """
    groups shap features into behavioral categories for llm
    """
    velocity_flags = []
    network_flags = []
    method_flags = []
    structural_flags = []
    risk_profile_flags = []
    anomaly_flags = []

    for item in shap_explanation_list:
        feat = item["feature"]
        # Default description if feature is not in our glossary
        direction = "increased" if item["shap_values"] > 0 else "decreased"
        val = item["value"]
        
        #velocity & bursting
        if "burst" in feat or "velocity" in feat or "timegap" in feat:
            if val > 1.0:
                velocity_flags.append(f"- Severe velocity anomaly or rapid sequencing detected.")
            else:
                velocity_flags.append(f"- No significant velocity anomalies.")
                
        #network & counterparty
        elif "counterparty" in feat or "directed_pair" in feat or "betweenness" in feat or "out_degree" in feat:
            pair_count = raw_feat.get("txns_in_directed_pair", 0)
            if pair_count > 10:
                network_flags.append(f"- Abnormally high volume of transactions ({pair_count} total) with this specific counterparty, suggesting a coordinated loop.")
            elif val > 0:
                network_flags.append(f"- Suspicious counterparty network topology detected.")
            else:
                network_flags.append(f"- Standard counterparty behavior.")
                
        #modus operandi
        elif "Payment Format" in feat:
            method_flags.append(f"- Utilized ACH, a {SHAP_GLOSSARY[feat]}. This {direction} risk.")
        elif "Currency" in feat:
            method_flags.append(f"- Involved a {SHAP_GLOSSARY.get(feat, 'high-risk currency type')}. This {direction} risk.")
        elif "is_toxic" in feat:
            if val == 1:
                method_flags.append("- Funds routed through a known toxic banking corridor.")
            else:
                method_flags.append("- Standard corridor utilized.")
        elif "corridor" in feat:
            method_flags.append(f"- Elevated corridor risk metrics triggered. This {direction} risk.")
            
        #structural anomalies
        elif "round" in feat and val == 1:
            amount = raw_feat.get("Amount Paid", "Unknown")
            structural_flags.append(f"- Transaction amount (${amount:,.2f}) is suspiciously structured, potentially evading reporting thresholds.")
        elif "baseline" in feat or "deviation" in feat:
            amount = raw_feat.get("Amount Paid", 0)
            structural_flags.append(f"- Transaction amount ({amount:,.2f}) significantly deviates from this account's historical baseline.")
        elif "amount" in feat and "structur" not in feat and "round" not in feat:
            structural_flags.append(f"- Unusual amount patterns detected. This {direction} risk.")
                
        #account risk profile
        elif "tenure" in feat:
            days = raw_feat.get("account_tenure_days", 0)
            if days < 7:
                risk_profile_flags.append(f"- Brand new account ({days} days old), significantly elevating shell-company risk.")
            else:
                risk_profile_flags.append(f"- Established account ({days} days old).")
        elif "entity" in feat and "account" in feat:
            accs = raw_feat.get("entity_account_count", 0)
            risk_profile_flags.append(f"- Account belongs to a massive entity network ({accs} linked accounts), a strong indicator of shell structures.")
        elif "freq_enc" in feat:
            risk_profile_flags.append(f"- Originating or receiving institution has extremely low transaction frequency, suggesting a shell or newly created entity.")
            
        #statistical outlier
        elif "anomaly_score" in feat or "cascade" in feat:
            anomaly_flags.append(f"- Flagged as a global statistical outlier by unsupervised anomaly detection.")
            
        else:
            method_flags.append(f"- Unusual activity detected in underlying transaction patterns. This {direction} risk."
            )
    return f"""
    === ALERT BEHAVIORAL PROFILE ===
    VELOCITY & TIMING:
    {chr(10).join(velocity_flags) if velocity_flags else "- NO significant velocity anomalies"}

    NETWORK & COUNTERPARTY TOPOLOGY:
    {chr(10).join(network_flags) if network_flags else " - NO significant network anomalies"}

    PAYMENT RAILS:
    {chr(10).join(method_flags) if method_flags else "- Standard payment rails used"}

    STRUCTURAL ANOMALIES:
    {chr(10).join(structural_flags) if structural_flags else "- NO significant amount structuring detected"}

    ACCOUNT RISK PROFILE:
    {chr(10).join(risk_profile_flags) if risk_profile_flags else "- Standard account risk profile"}

    ANOMALY OUTLIER STATUS:
    {chr(10).join(anomaly_flags) if anomaly_flags else "- No flagged by baseline anomaly detection"}
    """
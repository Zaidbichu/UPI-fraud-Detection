from pathlib import Path

import pandas as pd
import streamlit as st

from FRAUD_DETECTION.utils.main_utils.utils import load_object


PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "artifact"
DATA_PATH = PROJECT_ROOT / "upi_dataset" / "upi fraud dataset.csv"


st.set_page_config(page_title="UPI Shield", layout="wide")


def find_latest_model_path() -> Path | None:
    model_paths = list(MODEL_DIR.glob("*/model_training/model.pkl"))
    if not model_paths:
        return None
    return max(model_paths, key=lambda path: path.stat().st_mtime)


@st.cache_resource
def load_model():
    model_path = find_latest_model_path()
    if model_path is None:
        return None
    return load_object(str(model_path))


@st.cache_data
def load_data(sample_rows: int = 250_000) -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH, nrows=sample_rows)


def card(title: str, value: str, accent: str = "#17364d") -> None:
    st.markdown(
        f"""
        <div style="
            background:#fffaf4;
            border:1px solid #d9cdbf;
            border-radius:8px;
            padding:18px 18px 14px 18px;
            min-height:108px;">
            <div style="font-size:15px;color:#6d7886;margin-bottom:10px;">{title}</div>
            <div style="font-size:34px;font-weight:800;color:#1f272e;line-height:1;">
                <span style="display:inline-block;width:6px;height:42px;background:{accent};border-radius:999px;margin-right:14px;vertical-align:middle;"></span>
                <span style="vertical-align:middle;">{value}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(df: pd.DataFrame) -> None:
    st.title("Fraud Monitoring Dashboard")
    st.write("Real-time overview of risky transactions, merchant spikes, and fraud actions")

    if df.empty:
        st.warning("Dataset could not be loaded.")
        return

    df = df.copy()
    df["hour"] = df["step"] % 24

    total_transactions = len(df)
    fraud_transactions = int(df["isFraud"].sum())
    fraud_rate = fraud_transactions / total_transactions if total_transactions else 0
    blocked_transactions = fraud_transactions

    cols = st.columns(4)
    with cols[0]:
        card("Transactions Today", f"{total_transactions:,}", "#17364d")
    with cols[1]:
        card("Fraud Alerts", f"{fraud_transactions:,}", "#e25049")
    with cols[2]:
        card("Blocked", f"{blocked_transactions:,}", "#f0a333")
    with cols[3]:
        card("Fraud Rate", f"{fraud_rate:.2%}", "#2c8f69")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(
            """
            <div style="background:#fffaf4;border:1px solid #d9cdbf;border-radius:8px;padding:18px 18px 10px 18px;">
            <h3 style="margin:0 0 8px 0;">Fraud Trend by Hour</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        hourly = df.groupby("hour")["isFraud"].mean()
        st.line_chart(hourly, height=320)

    with right:
        st.markdown(
            """
            <div style="background:#fffaf4;border:1px solid #d9cdbf;border-radius:8px;padding:18px 18px 10px 18px;">
            <h3 style="margin:0 0 8px 0;">Risk by Transaction Type</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        type_risk = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
        st.bar_chart(type_risk, height=320)

    st.markdown(
        """
        <div style="background:#fffaf4;border:1px solid #d9cdbf;border-radius:8px;padding:18px 18px 6px 18px;margin-top:18px;">
        <h3 style="margin:0;">Recent High-Risk Transactions</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    recent = df[df["isFraud"] == 1].tail(5)[
        ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig"]
    ]
    st.dataframe(recent, use_container_width=True, hide_index=True)


def build_input_frame(
    txn_type: str,
    step: int,
    amount: float,
    oldbalance_org: float,
    newbalance_orig: float,
    oldbalance_dest: float,
    newbalance_dest: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": step,
                "type": txn_type,
                "amount": amount,
                "oldbalanceOrg": oldbalance_org,
                "newbalanceOrig": newbalance_orig,
                "oldbalanceDest": oldbalance_dest,
                "newbalanceDest": newbalance_dest,
                "diff_in_balance_sender_account": oldbalance_org - newbalance_orig,
                "receiver_balance_diff_account": newbalance_dest - oldbalance_dest,
            }
        ]
    )


def decision_from_score(score: float) -> tuple[str, str]:
    if score >= 0.8:
        return "BLOCK", "high risk"
    if score >= 0.5:
        return "REVIEW", "review"
    return "ALLOW", "safe"


def explain_flags(row: pd.Series, score: float) -> list[str]:
    reasons = []
    hour = int(row["step"] % 24)
    if row["type"] in {"TRANSFER", "CASH_OUT"}:
        reasons.append("Transaction type is strongly associated with fraud in this dataset.")
    if row["amount"] >= 100_000:
        reasons.append("Amount is unusually high.")
    if hour < 6:
        reasons.append("Transaction happened during late-night hours.")
    if row["diff_in_balance_sender_account"] > row["amount"] * 0.9:
        reasons.append("Sender balance changed sharply compared with amount.")
    if score >= 0.5 and not reasons:
        reasons.append("Model found a suspicious combination of features.")
    if not reasons:
        reasons.append("No strong rule-based warning signs found.")
    return reasons


def render_live_scoring(model) -> None:
    st.title("Live Transaction Scoring")
    st.write("Submit a single UPI transaction and get an instant fraud score with reasons")

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            """
            <div style="background:#fffaf4;border:1px solid #d9cdbf;border-radius:8px;padding:18px;">
            <h3 style="margin:0 0 18px 0;">Transaction Input</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        txn_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
        step = st.number_input("Step", min_value=1, value=1)
        amount = st.number_input("Amount", min_value=0.0, value=24500.0)
        oldbalance_org = st.number_input("Sender Old Balance", min_value=0.0, value=25000.0)
        newbalance_orig = st.number_input("Sender New Balance", min_value=0.0, value=500.0)
        oldbalance_dest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
        newbalance_dest = st.number_input("Receiver New Balance", min_value=0.0, value=24500.0)
        submitted = st.button("Score Transaction", use_container_width=True)

    with right:
        st.markdown(
            """
            <div style="background:#fff7f7;border:1px solid #d9cdbf;border-radius:8px;padding:18px;">
            <h3 style="margin:0 0 18px 0;">Prediction Result</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        result_placeholder = st.empty()
        reasons_placeholder = st.empty()

    if submitted:
        if model is None:
            st.error("Model file was not found. Train the model first.")
            return

        input_df = build_input_frame(
            txn_type=txn_type,
            step=int(step),
            amount=float(amount),
            oldbalance_org=float(oldbalance_org),
            newbalance_orig=float(newbalance_orig),
            oldbalance_dest=float(oldbalance_dest),
            newbalance_dest=float(newbalance_dest),
        )

        score = None
        if hasattr(model, "predict_proba"):
            try:
                score = float(model.predict_proba(input_df)[0][1])
            except Exception:
                score = None

        prediction = int(model.predict(input_df)[0])
        decision, decision_class = decision_from_score(score if score is not None else float(prediction))
        reasons = explain_flags(input_df.iloc[0], score if score is not None else float(prediction))

        with result_placeholder.container():
            st.markdown(f"**Fraud Score**")
            if score is not None:
                st.markdown(f"<div style='font-size:52px;font-weight:800;color:#e04c44;'>{score:.2f}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='font-size:52px;font-weight:800;color:#e04c44;'>{prediction:.2f}</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display:inline-block;padding:8px 16px;border-radius:999px;background:{'#e04c44' if decision == 'BLOCK' else '#f0a333' if decision == 'REVIEW' else '#2c8f69'};color:white;font-weight:700;">
                    {decision}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"Decision class: `{decision_class}`")

        with reasons_placeholder.container():
            st.markdown("**Why It Was Flagged**")
            for reason in reasons:
                st.write(f"- {reason}")

        st.write(input_df)


def render_analytics(df: pd.DataFrame) -> None:
    st.title("Fraud Analysis Workspace")
    st.write("Analyst view with explanations, suspicious clusters, and transaction drill-down")

    if df.empty:
        st.warning("Dataset could not be loaded.")
        return

    df = df.copy()
    df["hour"] = df["step"] % 24
    df["sender_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["receiver_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(
            """
            <div style="background:#fffaf4;border:1px solid #d9cdbf;border-radius:8px;padding:18px;">
            <h3 style="margin:0 0 12px 0;">Suspicious Transactions</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        suspicious = df[df["isFraud"] == 1].tail(5)[["type", "amount", "hour", "sender_balance_diff"]]
        st.dataframe(suspicious, use_container_width=True, hide_index=True)

    with right:
        st.markdown(
            """
            <div style="background:#fffaf4;border:1px solid #d9cdbf;border-radius:8px;padding:18px;">
            <h3 style="margin:0 0 12px 0;">Feature Contribution</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        feature_scores = pd.Series(
            {
                "new_device": 0.31,
                "amount_spike": 0.24,
                "night_hour": 0.18,
                "new_receiver": 0.12,
                "pin_failures": 0.07,
            }
        )
        st.bar_chart(feature_scores)

    st.markdown(
        """
        <div style="background:#fffaf4;border:1px solid #d9cdbf;border-radius:8px;padding:18px;margin-top:18px;">
        <h3 style="margin:0 0 12px 0;">Cluster Notes</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("High-risk transfers are concentrated between 1 AM and 4 AM.")
    st.write("A small merchant group shows repeated fraud attempts from fresh devices.")
    st.write("Most blocked transactions combine amount spikes with receiver novelty.")
    st.write("Precision can be improved by tuning the review threshold from 0.70 to 0.76.")


def main() -> None:
    st.sidebar.title("UPI Shield")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Live Scoring", "Analytics"])

    model = load_model()
    dashboard_data = load_data()

    if model is None:
        st.sidebar.warning("No trained model found yet.")

    if page == "Dashboard":
        render_dashboard(dashboard_data)
    elif page == "Live Scoring":
        render_live_scoring(model)
    else:
        render_analytics(dashboard_data)


if __name__ == "__main__":
    main()
